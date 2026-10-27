"""End-to-end smoke test on three tiny SYNTHETIC PDFs (fake numbers, not real report data).
Checks ingestion, retrieval, the evidence gate, the verifier + single retry, refusals, metrics check and chart.
Runs in a temp folder in ~10 s with no API key and no model download.

    python tests/smoke_test.py
"""
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="ar_smoke_"))
os.environ.update(REPORTS_DIR=str(TMP / "reports"), INDEX_DIR=str(TMP / "index"), METRICS_CSV=str(TMP / "metrics.csv"),
                  OUTPUT_DIR=str(TMP / "outputs"), LLM_PROVIDER="none", PRELOAD_RERANKER="0")
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402

FAKE = {  # company: (attrition %, employees, revenue Rs crore, margin %)
    "HCLTech": ("12.3", "1,11,111", "11,111", "18.1"),
    "Infosys": ("13.4", "2,22,222", "22,222", "21.2"),
    "TCS": ("14.5", "3,33,333", "33,333", "24.3"),
}


def make_pdfs() -> None:
    (TMP / "reports").mkdir(parents=True)
    for company, (attr, emp, rev, margin) in FAKE.items():
        doc = pymupdf.open()
        pages = [
            f"SYNTHETIC TEST DOCUMENT - NOT REAL DATA. {company} annual report test fixture for the smoke test.",
            f"Human resources. The attrition rate for the year was {attr}% (LTM, IT services). "
            f"Total employees as on March 31 were {emp}, across all geographies.",
            f"Financial performance. Consolidated revenue from operations was Rs. {rev} crore for the year, "
            f"an increase of 5.5% over the previous year. Operating margin was {margin}%.",
            "Artificial intelligence. We invested in AI platforms and trained 22,345 employees in generative AI tools.",
        ]
        for text in pages:
            doc.new_page().insert_textbox(pymupdf.Rect(60, 60, 540, 780), text, fontsize=11, fontname="helv")
        doc.save(TMP / "reports" / f"{company}_Synthetic_AR.pdf")


class FakeLLM:
    """Answers attrition questions; its FIRST answer carries a wrong figure, to exercise the verifier + retry."""
    provider, model = "fake", "fake-llm"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, max_tokens=800, schema=None):
        self.calls += 1
        for eid, company, text in re.findall(r"\[(E\d+)\] (\w+), [^\n]*\n(.*?)(?=\n\n\[E\d+\]|\n\n(?:Return|Your previous))", user, re.S):
            m = re.search(r"attrition rate for the year was\s+([\d.]+)%", text)
            if m:
                figure = "99.9" if self.calls == 1 else m.group(1)
                return json.dumps({"found": True, "claims": [{"text": f"{company} reported an attrition rate of {figure}%.",
                                   "evidence": [{"id": eid, "quote": f"The attrition rate for the year was {m.group(1)}%"}]}]})
        return json.dumps({"found": False, "claims": []})


def main() -> None:
    make_pdfs()
    from ingest import ingest
    ingest(TMP / "reports", TMP / "index", tfidf=True)

    from qa import QAEngine, format_answer
    from retriever import Retriever

    retriever = Retriever(TMP / "index")
    engine = QAEngine(retriever, llm=None)  # extractive mode
    cases = [("What attrition rate did HCLTech report?", True),
             ("Compare the attrition rates of HCLTech, Infosys and TCS.", True),
             ("What will TCS's share price be next year?", False),
             ("What was Wipro's operating margin?", False),
             ("What did HCLTech spend on its office cafeteria in Antarctica?", False)]
    for question, should_answer in cases:
        ans = engine.ask(question)
        print(format_answer(ans), "\n")
        assert ans.found == should_answer, question
    a = engine.ask("What attrition rate did HCLTech report?")
    assert a.claims[0].evidence[0]["chunk"]["page"] == 2 and a.claims[0].evidence[0]["chunk"]["company"] == "HCLTech"
    comparison = engine.ask("Compare the attrition rates of HCLTech, Infosys and TCS.")
    assert {e["chunk"]["company"] for c in comparison.claims for e in c.evidence} == set(FAKE)

    fake = FakeLLM()
    llm_engine = QAEngine(retriever, llm=fake)
    ans = llm_engine.ask("What attrition rate did Infosys report?")
    print(format_answer(ans))
    print("\n".join(ans.trace), "\n")
    assert ans.found and fake.calls == 2 and "13.4" in ans.claims[0].text, "verifier should reject 99.9 and the retry fix it"

    from extract_metrics import verify, write_rows
    import config
    rows = []
    for company, (attr, *_rest) in FAKE.items():
        row = {f: "" for f in config.METRIC_FIELDS}
        row.update(company=company, metric="attrition", value_text=attr, unit="%", fiscal_year="FY2025-26",
                   file=f"{company}_Synthetic_AR.pdf", pdf_page="2", quote=f"The attrition rate for the year was {attr}%",
                   note="LTM, IT services", human_check="yes")
        rows.append(row)
    rows[2] = {**rows[2], "pdf_page": "3"}  # wrong page on purpose
    write_rows(rows)
    verify()
    from extract_metrics import read_rows
    checks = [r["auto_check"] for r in read_rows()]
    assert checks[:2] == ["pass", "pass"] and checks[2].startswith("fail"), checks

    import chart
    chart.main()
    assert (TMP / "outputs" / "r1_attrition.png").exists()
    print(f"\nSmoke test passed (temp files in {TMP}).")


if __name__ == "__main__":
    try:
        main()
    finally:
        if "--keep" not in sys.argv:
            shutil.rmtree(TMP, ignore_errors=True)
