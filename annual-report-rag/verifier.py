"""Static verifier: deterministic checks, no LLM call, so it costs milliseconds rather than seconds.

check 1  source    every cited excerpt exists, and each company named in a claim is backed by that company's report
check 2  page      every quote is found again on the PDF page recorded for its excerpt (re-read from the page
                   store), and any page number written in the claim is one of the cited pages
check 3  support   every quote is verbatim in its excerpt, and every figure in the claim appears in a quote
check 4  coverage  every claim carries at least one piece of evidence

A claim is shown only if all four pass. Limits: these checks prove the claim is anchored to real text on the
cited page; they cannot judge comparative wording such as "highest" (see REPORT.md, constraints).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from textutil import companies_in, compact, numbers_in

try:
    from rapidfuzz import fuzz
except ImportError:  # still works, just without tolerance for one-character slips
    fuzz = None

MIN_QUOTE_CHARS = 12   # shorter quotes ("13.3%") could match anywhere
FUZZY_THRESHOLD = 92   # tolerate tiny transcription slips in words, never in figures

PageLookup = Callable[[str, int], str]


@dataclass
class ClaimCheck:
    text: str
    evidence: list[dict] = field(default_factory=list)  # {"id", "chunk", "quote"}
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def quote_in_text(quote: str, text: str) -> bool:
    parts = [p for p in re.split(r"\.\.\.|\u2026", quote or "") if p.strip()]
    if not parts:
        return False
    target = compact(text)
    for part in parts:
        q = compact(part)
        if len(q) < MIN_QUOTE_CHARS:
            return False
        if q in target:
            continue
        if fuzz is not None and fuzz.partial_ratio(q, target) >= FUZZY_THRESHOLD \
                and numbers_in(part, skip_small=False) <= numbers_in(text, skip_small=False):
            continue
        return False
    return True


def quote_on_page(quote: str, page_text: str, chunk: dict) -> bool:
    if quote_in_text(quote, page_text):
        return True
    if chunk.get("kind") == "table":  # the table finder re-orders cells, so match the figures instead
        figures = numbers_in(quote)
        return bool(figures) and figures <= numbers_in(page_text)
    return False


def verify_claim(claim: dict, excerpts: dict[str, dict], page_text: PageLookup) -> ClaimCheck:
    result = ClaimCheck(text=str(claim.get("text", "")).strip())
    raw = claim.get("evidence") or []
    raw = [raw] if isinstance(raw, dict) else list(raw)

    # check 1: source
    source_ok = True
    for ev in raw:
        if not isinstance(ev, dict):
            source_ok = False
            result.reasons.append("malformed evidence item")
            continue
        eid = str(ev.get("id", "")).strip().strip("[]")
        chunk = excerpts.get(eid)
        if chunk is None:
            source_ok = False
            result.reasons.append(f"cites unknown excerpt {eid!r}")
            continue
        result.evidence.append({"id": eid, "chunk": chunk, "quote": str(ev.get("quote", "")).strip()})
    cited_companies = {e["chunk"]["company"] for e in result.evidence}
    for company in companies_in(result.text):
        if company not in cited_companies:
            source_ok = False
            result.reasons.append(f"names {company} but cites no {company} excerpt")

    # check 2: page
    page_ok = bool(result.evidence)
    for e in result.evidence:
        chunk = e["chunk"]
        if not quote_on_page(e["quote"], page_text(chunk["file"], chunk["page"]), chunk):
            page_ok = False
            result.reasons.append(f"quote for {e['id']} not found on PDF page {chunk['page']}")
    cited_pages = {int(e["chunk"]["page"]) for e in result.evidence}
    for page in re.findall(r"\b(?:page|pg|p)\.?\s*(\d{1,4})\b", result.text, flags=re.I):
        if int(page) not in cited_pages:
            page_ok = False
            result.reasons.append(f"claim mentions page {page}, which is not a cited page")

    # check 3: support
    support_ok = bool(result.evidence)
    for e in result.evidence:
        if not quote_in_text(e["quote"], e["chunk"]["text"]):
            support_ok = False
            result.reasons.append(f"quote for {e['id']} is not verbatim in that excerpt")
    quoted = set().union(*(numbers_in(e["quote"]) for e in result.evidence)) if result.evidence else set()
    missing = numbers_in(result.text) - quoted
    if missing:
        support_ok = False
        result.reasons.append("figures not in the quoted evidence: " + ", ".join(sorted(missing)))

    # check 4: coverage
    covered = bool(result.text) and bool(raw)
    if not covered:
        result.reasons.append("claim has no evidence")

    result.checks = {"source": source_ok, "page": page_ok, "support": support_ok, "coverage": covered}
    return result


def verify_answer(parsed: dict, excerpts: dict[str, dict], page_text: PageLookup) -> tuple[list[ClaimCheck], list[ClaimCheck]]:
    """Returns (claims that passed every check, all checked claims)."""
    checked = [verify_claim(c, excerpts, page_text) for c in (parsed.get("claims") or []) if isinstance(c, dict)]
    return [c for c in checked if c.ok], checked
