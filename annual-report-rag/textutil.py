"""Text helpers shared by ingestion, retrieval and the verifier."""
from __future__ import annotations

import re
import unicodedata

from config import COMPANIES, OUT_OF_CORPUS, REPORTS

_CHAR_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u00a0": " ", "\u2009": " ", "\u202f": " ", "\u00ad": "", "\u200b": "",
})


def clean_text(text: str) -> str:
    """Normalise PDF text: ligatures (NFKC), smart quotes and dashes, soft/line-break hyphens, runs of spaces."""
    text = unicodedata.normalize("NFKC", text or "").translate(_CHAR_MAP)
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# def compact(text: str) -> str:
#     """Lower-case and drop whitespace and quote marks, so spacing differences don't break quote matching."""
#     return re.sub(r"[\s\"'`]+", "", clean_text(text).lower())

def compact(text: str) -> str:
    """Lower-case and drop whitespace, quote marks and table pipes, so spacing differences don't break quote matching."""
    return re.sub(r"[\s\"'`|]+", "", clean_text(text).lower())

# ---------------------------------------------------------------- figures
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
          r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?")
_FISCAL = re.compile(
    r"\b(?:fy\s?)?(?:19|20)\d{2}\s?[-/]\s?(?:19|20)?\d{2}\b"  # 2025-26, FY 2025-2026, 2025/26
    r"|\bfy\s?'?\d{2,4}\b"                                     # FY26, FY'26, FY2026
    r"|\b[qh][1-4]\b",                                         # Q4, H1
    re.I,
)
_DATE = re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}|\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?\b", re.I)
_NUMBER = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|(?<![\w.,])\d+(?:\.\d+)?")


def numbers_in(text: str, skip_small: bool = True) -> set[str]:
    """Figures in a string, normalised so '2,55,324' == '255,324' == '255324' and '13.50' == '13.5'.

    Fiscal-year labels and dates are ignored; by default so are bare years and single digits.
    """
    text = _DATE.sub(" ", _FISCAL.sub(" ", clean_text(text)))
    found: set[str] = set()
    for match in _NUMBER.finditer(text):
        raw = match.group()
        value = raw.replace(",", "")
        if "." in value:
            value = value.rstrip("0").rstrip(".") or "0"
        if skip_small and "," not in raw and "." not in raw:
            n = int(value)
            if n < 10 or 1900 <= n <= 2100:
                continue
        found.add(value)
    return found


# ---------------------------------------------------------------- companies
def _alias_regex(aliases) -> re.Pattern:
    alternatives = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])(?:{alternatives})(?![a-z0-9])")


_COMPANY_RE = {name: _alias_regex(aliases) for name, aliases in COMPANIES.items()}
_OUTSIDE_RE = _alias_regex(OUT_OF_CORPUS)


def companies_in(text: str) -> list[str]:
    """Corpus companies named in the text, e.g. 'TCS vs Infosys' -> ['Infosys', 'TCS']."""
    low = (text or "").lower()
    return [name for name, rx in _COMPANY_RE.items() if rx.search(low)]


def outside_companies_in(text: str) -> list[str]:
    return sorted(set(_OUTSIDE_RE.findall((text or "").lower())))


def company_for_file(filename: str) -> str | None:
    if REPORTS.get(filename, {}).get("company"):
        return REPORTS[filename]["company"]
    squashed = re.sub(r"[^a-z0-9]", "", filename.lower())
    for name, aliases in COMPANIES.items():
        if any(re.sub(r"[^a-z0-9]", "", alias) in squashed for alias in aliases):
            return name
    return None


# ---------------------------------------------------------------- terms
STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does doing for from had has have how i if in into is
it its me my of on or our over per should so than that the their them then there these they this those to under
up us was we were what when where which while who whom why will with would you your
""".split())

GENERIC_TERMS = frozenset("""
annual report reports company companies fiscal year years fy mention mentioned mentions say says said tell
according disclose disclosed disclosure state stated compare comparison versus vs between among three two both
each all across highest lowest higher lower most least total latest current recent give list show describe
explain please value number amount figure much many respectively
""".split())

COMPANY_WORDS = frozenset({"hcltech", "hcl", "infosys", "infy", "tcs", "tata", "consultancy", "technologies",
                           "ltd", "limited"})

_TOKEN = re.compile(r"[a-z0-9](?:[a-z0-9&%.\-]*[a-z0-9%])?")
_NOT_A_TERM = re.compile(r"[\d.,%\-]+|(?:fy|q|h)'?\d[\d\-]*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def stem(token: str) -> str:
    """Tiny plural stripper: 'employees'->'employee', 'margins'->'margin', 'companies'->'company'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def stem_tokens(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]


def key_terms(question: str) -> list[str]:
    """Content words of a question: no stopwords, generic words, company names, years or bare numbers."""
    out: list[str] = []
    for tok in tokenize(question):
        if (tok in STOPWORDS or tok in GENERIC_TERMS or tok in COMPANY_WORDS or len(tok) < 2
                or _NOT_A_TERM.fullmatch(tok) or tok in out):
            continue
        out.append(tok)
    return out


def term_in(term: str, text_lower: str) -> bool:
    """Loose match of a question term in lower-cased text: whole word for short terms, stem prefix otherwise."""
    if len(term) <= 3:
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text_lower) is not None
    root = stem(term)
    return root[: max(4, len(root) - 1)] in text_lower
