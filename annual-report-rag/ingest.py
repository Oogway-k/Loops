"""Step 1: ingest the annual-report PDFs into the stores of our design sketch.

    PDF --PyMuPDF, layout-aware--+--> page store   pages.jsonl   raw text per PDF page (verifier re-reads quotes here)
                                 +--> chunks ----> FAISS (dense, one index per company) + BM25 (pickled)      -> R2
                                 +--> tables ----> structured store tables.jsonl (header + rows per table)     -> R1

Parser, in order of what it fixes:
  * multi-column pages are read column by column (PyMuPDF's raw order interleaves the columns);
    number-heavy pages are read row by row instead, so a table row keeps its label next to its figures
  * running headers/footers (lines repeated in the page margins) are dropped from chunks, not from the page store
  * every page gets a section heading (largest font on the page, carried forward), stored on each chunk and
    prepended to the text that is embedded and BM25-indexed
  * tables are detected only on number-heavy pages, which keeps the slow table finder off prose pages
  * the PDFs are parsed in parallel, one process per file
Chunks never cross a page boundary, so every chunk still has exactly one page number.

Usage:
    python ingest.py               # all PDFs in data/reports/
    python ingest.py --no-tables   # skip table detection (faster)
    python ingest.py --tfidf       # skip the dense model (no download; no FAISS either, TF-IDF vectors are sparse)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf

import config
from textutil import clean_text, company_for_file

_YEAR = re.compile(r"\b(?:FY\s?)?(?:19|20)\d{2}(?:\s?[-/]\s?(?:19|20)?\d{2})?\b", re.I)
_KEEP_IN_MARGIN = re.compile(r"crore|lakh|million|billion|₹|%|march|\bfy", re.I)  # units/dates are never boilerplate


def split_page(text: str, size: int = config.CHUNK_CHARS, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split one page's text into overlapping chunks, preferring paragraph and sentence boundaries."""
    if len(text) <= size:
        return [text] if len(text) >= 40 else []
    pieces, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            window = start + size // 2
            cut = max(text.rfind("\n", window, end), text.rfind(". ", window, end))
            if cut != -1:
                end = cut + 1
        piece = text[start:end].strip()
        if len(piece) >= 40:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def digit_share(text: str) -> float:
    return sum(ch.isdigit() for ch in text) / max(1, len(text))


def norm_line(line: str) -> str:
    """'Page 12 | Annual Report' and 'Page 13 | Annual Report' normalise to the same line."""
    return re.sub(r"\d+", "#", " ".join(line.lower().split()))


def read_page(page) -> tuple[str, list[str], list[str]]:
    """Layout-aware page text. Returns (text, heading candidates best first, lines in the top/bottom margin)."""
    rect = page.rect
    width, height = rect.width or 1.0, rect.height or 1.0
    blocks, sizes, candidates, edges = [], Counter(), [], []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:  # 0 = text, 1 = image
            continue
        x0, y0, x1, y1 = block["bbox"]
        lines = []
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = clean_text("".join(s["text"] for s in line["spans"]))
            if not text:
                continue
            size = round(max(s["size"] for s in spans), 1)
            sizes[size] += len(text)
            candidates.append((size, line["bbox"][1], text))
            lines.append(text)
        if not lines:
            continue
        if y1 < rect.y0 + config.MARGIN_SHARE * height or y0 > rect.y1 - config.MARGIN_SHARE * height:
            edges += lines
        blocks.append((x0, y0, x1, "\n".join(lines)))
    if not blocks:
        return "", [], []

    tabular = digit_share("".join(b[3] for b in blocks)) >= config.TABLE_MIN_DIGIT_SHARE

    def order(b):
        x0, y0, x1, _ = b
        if tabular:  # row by row: blocks within ~3 pt vertically form one row, read left to right
            return (0, round(y0 / 3), x0)
        column = 0 if (x1 - x0) > 0.55 * width else min(2, int(3 * (x0 - rect.x0) / width))
        return (column, y0, x0)

    text = clean_text("\n\n".join(b[3] for b in sorted(blocks, key=order)))

    body = sizes.most_common(1)[0][0]  # body font size = the size with the most characters
    headings: list[str] = []
    for size, _, t in sorted(candidates, key=lambda c: (-c[0], c[1])):
        if size < body * 1.2:
            break
        if 3 <= len(t) <= 90 and sum(ch.isalpha() for ch in t) >= 0.6 * len(t) and t not in headings:
            headings.append(t)
    return text, headings[:5], edges


def extract_tables(page, base: dict, stem: str, page_no: int) -> list[dict]:
    """Structured tables (header + rows) via PyMuPDF's table finder; markdown keeps year headers on their figures."""
    try:
        found = page.find_tables().tables
    except Exception:
        return []
    out = []
    for t, table in enumerate(found):
        try:
            rows = [[clean_text(c or "") for c in row] for row in table.extract()]
            markdown = clean_text(table.to_markdown())
        except Exception:
            continue
        rows = [r for r in rows if any(r)]
        if len(rows) < 2 or len(markdown) < 80:
            continue
        try:
            header = [clean_text(h or "") for h in table.header.names]
        except Exception:
            header = rows[0]
        years = sorted({m.group() for h in header for m in _YEAR.finditer(h)})
        out.append({**base, "id": f"{stem}|p{page_no}|t{t}", "header": header, "rows": rows, "years": years,
                    "bbox": [round(v, 1) for v in table.bbox], "markdown": markdown[:4000]})
    return out


def report_title(pdf: Path) -> str:
    return config.REPORTS.get(pdf.name, {}).get("title") or pdf.stem.replace("_", " ")


def parse_pdf(path: str, tables: bool = True) -> dict:
    """Parse one PDF (runs in a worker process). Returns its pages, chunks and tables."""
    pdf = Path(path)
    company = company_for_file(pdf.name)
    if company is None:
        return {"file": pdf.name, "company": None}
    t0 = time.perf_counter()
    title = report_title(pdf)
    pages, table_recs = [], []
    with pymupdf.open(pdf) as doc:
        n_pages = len(doc)
        for page_no, page in enumerate(doc, start=1):
            text, headings, edges = read_page(page)
            base = {"company": company, "report": title, "file": pdf.name, "page": page_no,
                    "label": page.get_label() or ""}
            pages.append({**base, "section": "", "text": text, "_headings": headings, "_edges": edges})
            if tables and digit_share(text) >= config.TABLE_MIN_DIGIT_SHARE:
                table_recs += extract_tables(page, base, pdf.stem, page_no)

    # Running headers/footers: margin lines with real words that repeat on many pages of this report
    counts = Counter(n for p in pages for n in {norm_line(l) for l in p["_edges"]
                                                if len(l) <= 100 and sum(ch.isalpha() for ch in l) >= 3
                                                and not _KEEP_IN_MARGIN.search(l)})
    boiler = {n for n, c in counts.items() if c >= max(3, config.BOILERPLATE_MIN_SHARE * n_pages)}

    chunks, section = [], ""
    for p in pages:
        heads = [h for h in p.pop("_headings") if norm_line(h) not in boiler]
        p.pop("_edges")
        section = heads[0] if heads else section  # carry the last heading forward
        p["section"] = section
        body = "\n".join(l for l in p["text"].split("\n") if norm_line(l) not in boiler)
        base = {k: p[k] for k in ("company", "report", "file", "page", "label", "section")}
        for j, piece in enumerate(split_page(body)):
            chunks.append({**base, "id": f"{pdf.stem}|p{p['page']}|c{j}", "kind": "text", "text": piece})

    sections = {p["page"]: p["section"] for p in pages}
    for rec in table_recs:
        rec["section"] = sections[rec["page"]]
        chunks.append({k: rec[k] for k in ("company", "report", "file", "page", "label", "section", "id")}
                      | {"kind": "table", "text": rec["markdown"]})

    return {"file": pdf.name, "company": company, "report": title, "n_pages": n_pages, "pages": pages,
            "chunks": chunks, "tables": table_recs, "boiler": len(boiler),
            "empty": sum(len(p["text"]) < 40 for p in pages), "seconds": time.perf_counter() - t0}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ingest(reports_dir: Path, index_dir: Path, tables: bool = True, tfidf: bool = False, workers: int = 0) -> None:
    pdfs = sorted(Path(reports_dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {reports_dir}. Download the annual reports and put them there.")
    workers = workers or min(len(pdfs), os.cpu_count() or 1, 4)
    print(f"Parsing {len(pdfs)} PDF(s) with {workers} worker(s)...")
    results = None
    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(parse_pdf, [str(p) for p in pdfs], [tables] * len(pdfs)))
        except Exception as exc:
            print(f"  ! parallel parsing failed ({type(exc).__name__}: {exc}); parsing one file at a time")
    if results is None:
        results = [parse_pdf(str(p), tables) for p in pdfs]

    chunks, pages, table_rows, files = [], [], [], []
    for r in results:
        if r["company"] is None:
            print(f"  ! skipped {r['file']}: the file name matches no company alias (rename it or add it to config.REPORTS)")
            continue
        chunks += r["chunks"]
        pages += r["pages"]
        table_rows += r["tables"]
        files.append({"company": r["company"], "file": r["file"], "report": r["report"], "pages": r["n_pages"],
                      "chunks": len(r["chunks"]), "tables": len(r["tables"])})
        print(f"  {r['company']:8s} {r['file']}: {r['n_pages']} pages, {len(r['chunks'])} chunks, "
              f"{len(r['tables'])} tables, {r['boiler']} header/footer lines dropped ({r['seconds']:.1f}s)")
        if r["empty"] > 0.3 * r["n_pages"]:
            print(f"  ! {r['empty']}/{r['n_pages']} pages have almost no text. This PDF may be scanned.")

    if not chunks:
        raise SystemExit("No text extracted. Are the PDFs scanned images? Use machine-readable reports.")
    missing = sorted(set(config.COMPANIES) - {f["company"] for f in files})
    if missing:
        print(f"  ! no report found for: {', '.join(missing)}")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(index_dir / "chunks.jsonl", chunks)
    write_jsonl(index_dir / "pages.jsonl", pages)
    write_jsonl(index_dir / "tables.jsonl", table_rows)

    from index_store import build_stores  # imported here so worker processes don't load faiss/torch

    print(f"Embedding {len(chunks)} chunks and building FAISS + BM25...")
    t0 = time.perf_counter()
    info = build_stores(chunks, index_dir, force_tfidf=tfidf)
    meta = {"created": time.strftime("%Y-%m-%d %H:%M"), **info, "chunks": len(chunks), "tables": len(table_rows),
            "files": files}
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Index ready in {index_dir} ({info['vector_store']} / {info['model']} + BM25, "
          f"{time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", type=Path, default=config.REPORTS_DIR)
    ap.add_argument("--index", type=Path, default=config.INDEX_DIR)
    ap.add_argument("--no-tables", action="store_true", help="skip table detection")
    ap.add_argument("--tables", action="store_true", help="(kept for old commands; tables are on by default)")
    ap.add_argument("--tfidf", action="store_true", help="use TF-IDF instead of the dense embedding model")
    ap.add_argument("--workers", type=int, default=0, help="parser processes (default: one per PDF, max 4)")
    args = ap.parse_args()
    ingest(args.reports, args.index, tables=not args.no_tables, tfidf=args.tfidf, workers=args.workers)