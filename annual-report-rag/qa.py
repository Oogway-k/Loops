"""Step 2: question answering over the reports (R2).

question
  -> scope guard       refuse share-price/investment questions and companies outside the corpus (no LLM call)
  -> hybrid retrieval  cosine + BM25, routed per company
  -> evidence gate     pre-generation: do the retrieved pages contain the question's key terms at all?
  -> generation        one LLM call -> JSON claims, each with an excerpt id and a verbatim quote
  -> static verifier   4 deterministic checks (verifier.py); only claims that pass are shown
  -> single retry      if the gate or verifier fails: wider pool + reranker + the verifier's feedback
  -> answer with page citations, or "Not found in the reports."
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field, replace

import config
from llm import get_llm
from retriever import Hit, Retriever
from textutil import companies_in, key_terms, numbers_in, outside_companies_in, stem, term_in
from verifier import ClaimCheck, verify_answer

NOT_FOUND = "Not found in the reports."

SYSTEM_PROMPT = """You answer questions about the annual reports of HCLTech, Infosys and TCS, using ONLY the numbered excerpts provided. They are your entire world: no outside knowledge, no assumptions, no estimates.

Return JSON only, in exactly this shape:
{"found": true, "claims": [{"text": "<one sentence>", "evidence": [{"id": "E2", "quote": "<verbatim text from E2>"}]}]}

Rules:
1. Each claim is one factual sentence. Read in order, the claims form the answer.
2. Each claim cites at least one excerpt id with a quote copied character-for-character from that excerpt: 6 to 40 words, no ellipses, no paraphrasing.
3. Copy figures exactly as printed (for example "2,55,324" or "13.5%") with their unit and fiscal year. Never compute new numbers, convert units or round.
4. Name the company in every claim, and cite only that company's excerpts for its facts.
5. For comparisons, give each company's figure in its own claim.
6. Never write page numbers; citations are added automatically.
7. If the excerpts do not answer the question, return {"found": false, "claims": []}. Never guess. Share-price predictions and investment advice are never answerable.
8. If only part of the question is answerable, answer that part only."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "evidence": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "quote": {"type": "string"}},
                    "required": ["id", "quote"], "additionalProperties": False}},
            },
            "required": ["text", "evidence"], "additionalProperties": False}},
    },
    "required": ["found", "claims"],
    "additionalProperties": False,
}

_COMPARE = re.compile(r"\b(compare|compared|comparison|versus|vs\.?|all three|each of|across|which (?:company|one|of)"
                      r"|higher|lower|highest|lowest|more than|less than|difference|between|rank)\b", re.I)
_ADVICE = re.compile(r"should i (?:buy|sell|invest|hold)|buy or sell|worth (?:buying|investing)|investment advice"
                     r"|(?:target|future) (?:share |stock )?price|price target"
                     r"|(?:share|stock) price (?:prediction|forecast|target|next)"
                     r"|predict\w*\b[\w\s']{0,30}(?:share|stock|price)"
                     r"|will\b[\w\s']{0,30}(?:share|stock) price"
                     r"|(?:share|stock) price\b[\w\s']{0,20}\b(?:be|go|reach)\b", re.I)
_DISPLAY = {"ltts": "LTTS", "kpit": "KPIT", "ltimindtree": "LTIMindtree", "lti mindtree": "LTIMindtree"}
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])|\n{2,}|\n(?=[A-Z\u2022-])")


@dataclass
class Plan:
    companies: list[str]
    comparative: bool
    outside: list[str]
    advice: bool


@dataclass
class Answer:
    question: str
    found: bool
    claims: list[ClaimCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reason: str = ""
    trace: list[str] = field(default_factory=list)
    mode: str = ""
    seconds: float = 0.0


def make_plan(question: str) -> Plan:
    companies = companies_in(question)
    return Plan(companies=companies,
                comparative=bool(_COMPARE.search(question)) or len(companies) > 1,
                outside=outside_companies_in(question),
                advice=bool(_ADVICE.search(question)))


# def cite(chunk: dict) -> str:
#     report = chunk.get("report") or chunk.get("file", "")
#     head = report if chunk["company"].lower() in report.lower() else f"{chunk['company']}, {report}"
#     label = str(chunk.get("label") or "")
#     printed = f" (printed page {label})" if label and label != str(chunk["page"]) else ""
#     return f"{head}, PDF page {chunk['page']}{printed}"


def cite(chunk: dict) -> str:
    report = chunk.get("report") or chunk.get("file", "")
    head = report if chunk["company"].lower() in report.lower() else f"{chunk['company']}, {report}"
    label = str(chunk.get("label") or "")
    printed = f" (printed page {label})" if label and label != str(chunk["page"]) else ""
    section = str(chunk.get("section") or "")
    return f"{head}, PDF page {chunk['page']}{printed}" + (f", section “{section}”" if section else "")


def build_prompt(question: str, excerpts: dict[str, dict], feedback: str = "") -> str:
    blocks = [f"[{eid}] {chunk['company']}, {chunk['report']}\n{chunk['text']}" for eid, chunk in excerpts.items()]
    retry = (f"\n\nYour previous answer failed verification: {feedback}\n"
             "Fix it: copy quotes exactly and keep every figure of a claim inside its quote.") if feedback else ""
    return f"Question: {question}\n\nExcerpts:\n\n" + "\n\n".join(blocks) + retry + "\n\nReturn the JSON now."


def parse_json(raw: str) -> dict:
    text = re.sub(r"```(?:json)?", "", raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {"found": False, "claims": [], "error": "no JSON object in the model output"}
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        return {"found": False, "claims": [], "error": f"invalid JSON ({exc.msg})"}
    if not isinstance(data, dict):
        return {"found": False, "claims": [], "error": "JSON is not an object"}
    found = data.get("found")
    data["found"] = found.strip().lower() == "true" if isinstance(found, str) else bool(found)
    return data


def evidence_gate(question: str, hits: list[Hit]) -> tuple[bool, str]:
    """Pre-generation check. Cheap and lenient: it only stops questions whose key terms appear nowhere in the
    retrieved pages. Subtler cases are left to the LLM's 'found: false' and to the static verifier."""
    if not hits:
        return False, "no excerpts retrieved"
    terms = key_terms(question)
    if not terms:
        return True, "no key terms to check"
    text = " ".join(h.chunk["text"] for h in hits).lower()
    present = [t for t in terms if term_in(t, text)]
    missing = [t for t in terms if t not in present]
    need = max(1, math.ceil(config.MIN_TERM_COVERAGE * len(terms)))
    if len(present) < need:
        return False, f"missing key terms: {', '.join(missing)}"
    return True, f"{len(present)}/{len(terms)} key terms present" + (f" (missing: {', '.join(missing)})" if missing else "")


def extractive_answer(question: str, excerpts: dict[str, dict], idf: dict, max_claims: int = 3) -> dict:
    """No-LLM fallback (assumption 3 of the brief): verbatim report sentences containing the question's key terms."""
    terms = key_terms(question)
    if not terms or any(stem(t) not in idf for t in terms):  # a word the reports never use: play safe
        return {"found": False, "claims": []}
    rarest = max(terms, key=lambda t: idf.get(stem(t), 0.0))
    need = max(1, math.ceil(0.5 * len(terms)))
    candidates = []
    for eid, chunk in excerpts.items():
        for sentence in _SENTENCE.split(chunk["text"]):
            s = " ".join(sentence.split())
            low = s.lower()
            if not 20 <= len(s) <= 450 or not term_in(rarest, low):
                continue
            hits = sum(term_in(t, low) for t in terms)
            if hits >= need:
                candidates.append((hits, bool(numbers_in(s)), -len(s), eid, s))
    if not candidates:
        return {"found": False, "claims": []}
    candidates.sort(reverse=True)
    chosen, companies = [], set()
    for cand in candidates:  # best sentence per company first, so comparisons stay balanced
        company = excerpts[cand[3]]["company"]
        if company not in companies:
            chosen.append(cand)
            companies.add(company)
    for cand in candidates:
        if len(chosen) >= max_claims:
            break
        if cand not in chosen:
            chosen.append(cand)
    return {"found": True,
            "claims": [{"text": s, "evidence": [{"id": eid, "quote": s}]} for *_, eid, s in chosen[:max_claims]]}


class QAEngine:
    def __init__(self, retriever: Retriever | None = None, llm="auto"):
        self.r = retriever or Retriever()
        self.llm = get_llm() if llm == "auto" else llm
        self.mode = f"{self.llm.provider} / {self.llm.model}" if self.llm else "extractive mode (no LLM configured)"
        self._cache: dict[str, Answer] = {}
        if config.PRELOAD_RERANKER:
            self.r._get_reranker()

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        question = " ".join((question or "").split())
        key = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
        if key in self._cache:
            cached = self._cache[key]
            return replace(cached, trace=cached.trace + ["(repeated question: served from cache)"],
                           seconds=time.perf_counter() - t0)
        answer = self._answer(question, t0)
        answer.seconds = time.perf_counter() - t0
        if not any("LLM error" in line for line in answer.trace):  # never cache a transient failure
            self._cache[key] = answer
        return answer

    def _answer(self, q: str, t0: float) -> Answer:
        trace: list[str] = []

        def log(msg: str) -> None:
            trace.append(f"{time.perf_counter() - t0:5.2f}s  {msg}")

        if not q:
            return Answer(q, False, reason="Type a question first.", trace=trace, mode=self.mode)
        plan = make_plan(q)
        log(f"plan: companies={', '.join(plan.companies) or 'all'}; comparison={plan.comparative}")
        if plan.advice:
            log("scope guard: share-price prediction or investment advice")
            return self._not_found(q, "The reports contain no share-price predictions or investment advice, "
                                      "and this system answers only from the reports.", trace)
        if plan.outside and not plan.companies:
            names = ", ".join(_DISPLAY.get(n, n.title()) for n in plan.outside)
            log(f"scope guard: {names} is not in the corpus")
            return self._not_found(q, f"Only the HCLTech, Infosys and TCS annual reports are indexed, so questions "
                                      f"about {names} can't be answered.", trace)
        absent = [c for c in plan.companies if c not in self.r.companies]
        if plan.companies and len(absent) == len(plan.companies):
            return self._not_found(q, f"The {', '.join(absent)} report has not been ingested.", trace)
        targets = [c for c in plan.companies if c in self.r.companies] or self.r.companies

        seen: set[str] = set()
        feedback, reason = "", "No relevant evidence was retrieved."
        for attempt in (1, 2):
            hits = self.retrieve(q, targets, attempt)
            extra = f" (reranker: {self.r.reranker_status})" if attempt == 2 else ""
            log(f"attempt {attempt}: retrieved {len(hits)} excerpts{extra}")
            passed, why = evidence_gate(q, hits)
            log(f"evidence gate: {'pass' if passed else 'fail'}, {why}")
            if not passed:
                reason = f"The retrieved report pages don't mention the question's key terms ({why})."
                continue
            ids = {h.chunk["id"] for h in hits}
            if attempt == 2 and ids <= seen and not feedback:  # same pages, nothing to fix: don't pay for a 2nd call
                log("retry skipped: reranking surfaced no new pages")
                break
            seen |= ids
            excerpts = {f"E{i}": h.chunk for i, h in enumerate(hits, start=1)}
            parsed, mode = self.generate(q, excerpts, feedback, log)
            if not parsed.get("found") or not parsed.get("claims"):
                if parsed.get("error"):
                    log(f"generator ({mode}): {parsed['error']}")
                    feedback, reason = "your output was not valid JSON; return the JSON object only", \
                        "The model's output could not be read."
                else:
                    log(f"generator ({mode}): the excerpts don't answer the question")
                    feedback, reason = "", "The retrieved report pages don't contain this information."
                    if not config.RETRY_ON_NOT_FOUND:
                        break
                continue
            verified, checked = verify_answer(parsed, excerpts, self.r.page_text)
            log(f"static verifier: {len(verified)}/{len(checked)} claims passed")
            for c in checked:
                if not c.ok:
                    log("  failed: " + "; ".join(c.reasons))
            if verified:
                return Answer(q, True, claims=verified, notes=self._notes(plan, targets, verified, checked),
                              trace=trace, mode=self.mode if mode == "llm" else "extractive (verbatim report sentences)")
            feedback = " | ".join("; ".join(c.reasons) for c in checked if not c.ok)[:800]
            reason = "A candidate answer was produced, but its citations failed verification, so it isn't shown."
        log("exit: not found")
        return self._not_found(q, reason, trace)

    def retrieve(self, q: str, targets: list[str], attempt: int) -> list[Hit]:
        if attempt == 1:
            if len(targets) == 1:
                return self.r.search(q, config.TOP_K, targets)
            return self.r.search_per_company(q, targets, config.TOP_K_PER_COMPANY)
        # single retry: wider candidate pool, then rerank (cross-encoder if available)
        if len(targets) == 1:
            return self.r.rerank(q, self.r.search(q, config.CANDIDATE_POOL, targets), config.RETRY_K)
        k_each = max(2, math.ceil(config.RETRY_K / len(targets)))
        pool = max(10, config.CANDIDATE_POOL // len(targets))
        hits: list[Hit] = []
        for company in targets:
            hits += self.r.rerank(q, self.r.search(q, pool, [company], pool=pool), k_each)
        return hits

    def generate(self, q: str, excerpts: dict[str, dict], feedback: str, log) -> tuple[dict, str]:
        if self.llm is not None:
            try:
                raw = self.llm.complete(SYSTEM_PROMPT, build_prompt(q, excerpts, feedback),
                                        config.MAX_ANSWER_TOKENS, schema=ANSWER_SCHEMA)
                return parse_json(raw), "llm"
            except Exception as exc:  # rate limit, network, bad key: answer extractively instead of failing
                log(f"LLM error ({type(exc).__name__}: {str(exc)[:150]}); falling back to extractive mode")
        return extractive_answer(q, excerpts, self.r.bm25.idf), "extractive"

    @staticmethod
    def _notes(plan: Plan, targets: list[str], verified: list[ClaimCheck], checked: list[ClaimCheck]) -> list[str]:
        notes = []
        dropped = len(checked) - len(verified)
        if dropped:
            notes.append(f"{dropped} statement(s) were left out because their citations failed verification.")
        if plan.comparative:
            covered = {e["chunk"]["company"] for c in verified for e in c.evidence}
            notes += [f"{company}: not found in the reports for this question." for company in targets
                      if company not in covered]
        return notes

    def _not_found(self, q: str, reason: str, trace: list[str]) -> Answer:
        return Answer(q, False, reason=reason, trace=trace, mode=self.mode)


def format_answer(ans: Answer) -> str:
    lines = [f"Q: {ans.question}"]
    if ans.found:
        for c in ans.claims:
            sources = "; ".join(dict.fromkeys(cite(e["chunk"]) for e in c.evidence))
            lines.append(f"  - {c.text}\n      [{sources}]")
        lines += [f"  note: {n}" for n in ans.notes]
    else:
        lines.append(f"  {NOT_FOUND} {ans.reason}")
    lines.append(f"  ({ans.seconds:.2f}s, {ans.mode})")
    return "\n".join(lines)
