"""Step 3: data for the R1 chart -> data/metrics.csv

One row per (company, metric): value as printed, unit, fiscal year, PDF page and a verbatim quote.
Every row is then checked against the stored text of that PDF page (auto_check). A person opens the page and
writes "yes" in human_check; rows marked "yes" are never overwritten by a later extraction run.

    python extract_metrics.py            LLM-assisted extraction + automatic page check
    python extract_metrics.py --verify   only re-check the rows in metrics.csv (use after editing it by hand)
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import config
from llm import get_llm
from qa import parse_json
from retriever import Retriever, load_pages
from textutil import numbers_in
from verifier import quote_in_text

FIELDS = config.METRIC_FIELDS

PROMPT = """You extract one figure from excerpts of the {company} annual report. Use only the excerpts.
Figure wanted: {hint}. Prefer consolidated over standalone figures and the latest fiscal year in the report.
Return JSON only:
{{"found": true, "value_text": "<figure exactly as printed, e.g. 2,55,324>", "unit": "<e.g. ₹ crore, %, employees>", "fiscal_year": "<e.g. FY2025-26>", "id": "<excerpt id>", "quote": "<verbatim text from that excerpt, 6-40 words, containing the figure>", "note": "<definition caveat, or empty>"}}
If the excerpts don't state it, return {{"found": false, "value_text": "", "unit": "", "fiscal_year": "", "id": "", "quote": "", "note": ""}}."""

SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "boolean" if k == "found" else "string"}
                   for k in ("found", "value_text", "unit", "fiscal_year", "id", "quote", "note")},
    "required": ["found", "value_text", "unit", "fiscal_year", "id", "quote", "note"],
    "additionalProperties": False,
}


def parse_value(text: str) -> str:
    match = re.search(r"\d[\d,]*(?:\.\d+)?", text or "")
    return match.group().replace(",", "") if match else ""


def check_row(row: dict, pages: dict) -> str:
    """'pass' only if the figure (and the quote, when given) can be found on the cited PDF page."""
    page_no = str(row.get("pdf_page", "")).strip()
    page = pages.get((row.get("file", ""), int(page_no))) if page_no.isdigit() else None
    if page is None:
        return "fail: file/page not in the index"
    figure = numbers_in(row.get("value_text") or row.get("value") or "", skip_small=False)
    if not figure:
        return "fail: no figure"
    if row.get("quote"):
        if not quote_in_text(row["quote"], page["text"]):
            return "fail: quote not found on that page"
        if not figure <= numbers_in(row["quote"], skip_small=False):
            return "fail: figure not inside the quote"
    elif not figure <= numbers_in(page["text"], skip_small=False):
        return "fail: figure not found on that page"
    return "pass"


def read_rows(path: Path = config.METRICS_CSV) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [{k: (row.get(k) or "").strip() for k in FIELDS} for row in csv.DictReader(f)]


def write_rows(rows: list[dict], path: Path = config.METRICS_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summary(rows: list[dict]) -> None:
    for row in rows:
        if row.get("value_text") or row.get("value"):
            print(f"  {row['company']:8s} {row['metric']:17s} {row['value_text'] or row['value']:>12s} {row['unit']:10s} "
                  f"p.{row['pdf_page']:>4s}  {row['auto_check']}  human_check={row['human_check'] or '-'}")


def verify() -> None:
    pages = load_pages()
    rows = read_rows()
    for row in rows:
        if not (row["value_text"] or row["value"]):
            continue
        if row["value_text"] and not row["value"]:
            row["value"] = parse_value(row["value_text"])
        page = pages.get((row["file"], int(row["pdf_page"]))) if row["pdf_page"].isdigit() else None
        if page:
            row["page_label"] = row["page_label"] or page.get("label", "")
            row["report"] = row["report"] or page.get("report", "")
        row["auto_check"] = check_row(row, pages)
    write_rows(rows)
    summary(rows)


def extract() -> None:
    llm = get_llm()
    if llm is None:
        raise SystemExit("No LLM configured. Fill data/metrics.csv by hand (see README), then run: "
                         "python extract_metrics.py --verify")
    retriever = Retriever()
    existing = {(r["company"], r["metric"]): r for r in read_rows()}
    rows = []
    for company in retriever.companies:
        for key, metric in config.METRICS.items():
            old = existing.pop((company, key), None)
            if old and old["human_check"].lower() == "yes":  # never overwrite a human-checked value
                rows.append(old)
                continue
            hits = retriever.search(metric["query"], 8, [company])
            excerpts = {f"E{i}": h.chunk for i, h in enumerate(hits, start=1)}
            user = "\n\n".join(f"[{eid}] {chunk['text']}" for eid, chunk in excerpts.items())
            try:
                out = parse_json(llm.complete(PROMPT.format(company=company, hint=metric["hint"]), user, 400, schema=SCHEMA))
            except Exception as exc:
                print(f"  ! {company} {key}: LLM error {exc}")
                out = {"found": False}
            row = {f: "" for f in FIELDS} | {"company": company, "metric": key}
            chunk = excerpts.get(str(out.get("id", "")).strip("[] "))
            if out.get("found") and chunk:
                row.update(value_text=str(out.get("value_text", "")).strip(), unit=str(out.get("unit", "")).strip(),
                           fiscal_year=str(out.get("fiscal_year", "")).strip(), report=chunk["report"],
                           file=chunk["file"], pdf_page=str(chunk["page"]), page_label=chunk.get("label", ""),
                           quote=str(out.get("quote", "")).strip(), note=str(out.get("note", "")).strip())
                row["value"] = parse_value(row["value_text"])
                row["auto_check"] = check_row(row, retriever.pages)
            else:
                row["auto_check"] = "not found by extractor: fill by hand"
            rows.append(row)
    rows += list(existing.values())  # keep any custom rows
    write_rows(rows)
    summary(rows)
    print("\nNext: open each cited PDF page, confirm the value, and write 'yes' in the human_check column.")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="only re-check the rows already in metrics.csv")
    args = ap.parse_args()
    verify() if args.verify else extract()
