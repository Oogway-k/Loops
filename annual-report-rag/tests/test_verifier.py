"""Unit tests for the static verifier and text helpers (no PDFs or API keys needed).

    python -m pytest tests        or        python tests/test_verifier.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa import make_plan  # noqa: E402
from textutil import companies_in, key_terms, numbers_in  # noqa: E402
from verifier import verify_claim  # noqa: E402

PAGE = ("Highlights\nRevenue from operations grew 6.0% to Rs. 2,55,324 crore in FY 2025-26.\n"
        "Workforce strength stood at 6,07,979 as on March 31, 2026.")
TCS = {"id": "tcs|p12|c0", "company": "TCS", "report": "TCS AR", "file": "tcs.pdf", "page": 12, "label": "",
       "kind": "text", "text": PAGE}
INFY = {**TCS, "id": "infy|p40|c0", "company": "Infosys", "file": "infy.pdf", "page": 40,
        "text": "Infosys revenue from operations was Rs. 1,62,990 crore for the year."}
EXCERPTS = {"E1": TCS, "E2": INFY}
PAGES = {("tcs.pdf", 12): PAGE, ("infy.pdf", 40): INFY["text"]}


def page_text(file, page):
    return PAGES.get((file, page), "")


def claim(text, *evidence):
    return {"text": text, "evidence": [{"id": i, "quote": q} for i, q in evidence]}


def check(c):
    return verify_claim(c, EXCERPTS, page_text)


def test_numbers_are_normalised():
    assert numbers_in("Rs. 2,55,324 crore in FY2025-26, up 6.0% as on March 31, 2026") == {"255324", "6"}
    assert numbers_in("255,324") == numbers_in("2,55,324") == {"255324"}
    assert numbers_in("13.50%") == {"13.5"}


def test_good_claim_passes():
    c = check(claim("TCS reported revenue from operations of Rs. 2,55,324 crore in FY 2025-26.",
                    ("E1", "Revenue from operations grew 6.0% to Rs. 2,55,324 crore")))
    assert c.ok, c.reasons


def test_spacing_and_quote_marks_do_not_matter():
    c = check(claim("TCS had a workforce of 6,07,979 as on March 31, 2026.",
                    ("E1", "Workforce  strength stood at 6,07,979 as on “March 31, 2026”")))
    assert c.ok, c.reasons


def test_check1_unknown_excerpt():
    c = check(claim("TCS revenue was Rs. 2,55,324 crore.", ("E9", "Revenue from operations grew 6.0% to Rs. 2,55,324 crore")))
    assert not c.checks["source"]


def test_check1_company_misattribution():
    c = check(claim("Infosys reported revenue of Rs. 2,55,324 crore.",
                    ("E1", "Revenue from operations grew 6.0% to Rs. 2,55,324 crore")))
    assert not c.checks["source"]


def test_check2_wrong_page_in_text():
    c = check(claim("TCS had 6,07,979 employees, per page 99.", ("E1", "Workforce strength stood at 6,07,979 as on March 31")))
    assert not c.checks["page"]


def test_check3_figure_not_in_quote():
    c = check(claim("TCS revenue was Rs. 2,60,000 crore.", ("E1", "Revenue from operations grew 6.0% to Rs. 2,55,324 crore")))
    assert not c.checks["support"]


def test_check3_fabricated_quote():
    c = check(claim("TCS revenue rose 9% to a record.", ("E1", "Revenue rose 9% to a record high this year")))
    assert not c.checks["support"] and not c.checks["page"]


def test_check4_claim_without_evidence():
    c = check({"text": "TCS is the largest of the three.", "evidence": []})
    assert not c.checks["coverage"]


def test_question_routing():
    assert companies_in("TCS vs Infosys attrition") == ["Infosys", "TCS"]
    assert companies_in("How did HCL Technologies do?") == ["HCLTech"]
    assert key_terms("What was TCS's attrition rate in FY26?") == ["attrition", "rate"]
    assert make_plan("What will TCS's share price be next year?").advice
    assert not make_plan("What were TCS's monthly share price high and low?").advice
    assert make_plan("What is Wipro's revenue?").outside == ["wipro"]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
