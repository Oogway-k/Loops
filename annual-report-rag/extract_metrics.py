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

# Verified FY2025-26 values (read from each report and checked by hand). Used when no LLM is loaded.
# company, metric, value as printed, unit, file in data/reports, PDF page, printed page, quote, note
_PL = "Consolidated statement of profit and loss"
VERIFIED = [
    ("HCLTech", "revenue", "130,144", "₹ crore", "HCLTech.pdf", 375, "372", "Revenue from operations 3.19 130,144 117,055", _PL),
    ("HCLTech", "net_profit", "16,652", "₹ crore", "HCLTech.pdf", 375, "372", "Profit for the year 16,652 17,399", "Profit for the year, consolidated"),
    ("HCLTech", "headcount", "237,098", "employees", "HCLTech.pdf", 216, "213", "Total employees (E+F) 237,098 166,670 70.29% 69,897 29.48% 531 0.22%", "BRSR: permanent + other employees"),
    ("HCLTech", "attrition", "12.51", "%", "HCLTech.pdf", 217, "214", "Permanent 12.61% 12.28% 8.17% 12.51% 13.06% 12.81% 5.66% 12.99%", "Voluntary attrition, LTM, IT services (BRSR)"),
    ("Infosys", "revenue", "1,78,650", "₹ crore", "infosys-ar-26.pdf", 287, "318", "Revenue from operations 2.18 1,78,650 1,62,990", _PL),
    ("Infosys", "net_profit", "29,474", "₹ crore", "infosys-ar-26.pdf", 287, "318", "Profit for the year 29,474 26,750", "Profit for the year, consolidated"),
    ("Infosys", "headcount", "3,52,095", "employees", "infosys-ar-26.pdf", 143, "174", "Total employees (D + E) 3,52,095 2,17,818 61.9 1,34,277 38.1", "BRSR: permanent + other employees"),
    ("Infosys", "attrition", "12.6", "%", "infosys-ar-26.pdf", 144, "175", "Permanent employees 13.2 11.6 12.6 14.5 13.6 14.1", "Voluntary attrition, IT services excl. BPM (BRSR)"),
    ("TCS", "revenue", "2,67,021", "₹ crore", "TCS_Annual_Report_2025-26.pdf", 167, "197", "Revenue from operations 12 2,67,021 2,55,324", _PL),
    ("TCS", "net_profit", "49,454", "₹ crore", "TCS_Annual_Report_2025-26.pdf", 167, "197", "PROFIT FOR THE YEAR 49,454 48,797", "Profit for the year, consolidated"),
    ("TCS", "headcount", "6,17,437", "employees", "TCS_Annual_Report_2025-26.pdf", 108, "138", "Total employees (D + E) 6,17,437 3,99,559 64.7 2,17,878 35.3", "BRSR: permanent + other employees"),
    ("TCS", "attrition", "13.7", "%", "TCS_Annual_Report_2025-26.pdf", 108, "138", "Permanent Employees 13.8 13.4 13.7 13.2 13.6 13.3", "Voluntary attrition, LTM, IT services (BRSR)"),
]


def seed_verified() -> None:
    """No-LLM path: write the verified values into metrics.csv, keep human-checked rows, check every page."""
    try:
        pages = load_pages()
    except FileNotFoundError:
        pages = {}
        print("  ! no index yet (run python ingest.py); rows are written but the page check will fail")
    existing = {(r["company"], r["metric"]): r for r in read_rows()}
    rows = []
    for company, metric, value_text, unit, file, page, printed, quote, note in VERIFIED:
        old = existing.pop((company, metric), None)
        if old and old["human_check"].lower() == "yes":
            rows.append(old)
            continue
        row = {f: "" for f in FIELDS} | {
            "company": company, "metric": metric, "value": parse_value(value_text), "value_text": value_text,
            "unit": unit, "fiscal_year": "FY2025-26", "report": f"{company} Annual Report 2025-26", "file": file,
            "pdf_page": str(page), "page_label": printed, "quote": quote, "note": note}
        row["auto_check"] = check_row(row, pages)
        rows.append(row)
    rows += list(existing.values())
    write_rows(rows)
    summary(rows)


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
        print("No LLM loaded: using the verified FY2025-26 values.")
        seed_verified()
        return
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
