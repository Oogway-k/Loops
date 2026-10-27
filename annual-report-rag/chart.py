"""R1: comparison chart from data/metrics.csv, with a strict check of every value against its PDF page.
Every bar carries the PDF page its value came from, and any mismatch in units, fiscal years or
definitions is stated next to the chart. A value gets a bar only if it passes the check.

    python chart.py             fill empty rows of metrics.csv with the verified FY2025-26 values (only rows
                                with no value, only when the cited PDF is in data/reports), check every value
                                against its PDF page, then save outputs/r1_<metric>.png
    python chart.py --check     fill and check metrics.csv only; draw no charts
    python chart.py --no-fill   check and chart only what metrics.csv already holds

The check re-reads the cited PDF page and rebuilds table rows from word positions (also on rotated
landscape pages). Its four parts use the same names as the answer verifier:

  1 source    the PDF is that company's report and covers the row's fiscal year
  2 page      the quote is on the cited page (if not, the reason names the page it is on)
  3 support   the figure is on the metric's own table row, or in the same sentence as its name; in a
              table it sits in the right column: the row's fiscal year, the consolidated block when
              standalone and consolidated stand side by side, the Total column when a year is split by
              gender; the unit matches the page's own unit line; money figures come from a consolidated page
  4 coverage  one row per company, metric and year; companies without a passing value are listed

The result goes into the auto_check column ("pass" or "fail: <reason>"), the details into
outputs/metrics_validation.md. human_check is never touched: a person opens each cited page and writes
"yes". extract_metrics.py --verify writes a looser check into auto_check, so run chart.py after it.

app.py uses load_metrics, chart_data and make_figure; their names, arguments and results are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import textwrap
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

import config  # noqa: E402

FIELDS = config.METRIC_FIELDS
COMPANY_COLORS = {"HCLTech": "#1F5A96", "Infosys": "#2A9D8F", "TCS": "#6A4C93"}
YEAR_COLORS = ["#1F5A96", "#2A9D8F", "#6A4C93", "#E9A23B", "#8D99AE"]
_INR_UNITS = [("crore", 1.0), ("cr", 1.0), ("lakh", 0.01), ("million", 0.1), ("mn", 0.1), ("billion", 100.0), ("bn", 100.0)]

# ---------------------------------------------------------------- verified values (FY2025-26)
# Read from each company's own annual report and checked against the page. PDF page = page index in a
# PDF viewer; printed page = the number in the page footer. Quotes are copied from the page.
VERIFIED_FY = "FY2025-26"
VERIFIED_REPORTS = {  # company: (report title, file in data/reports)
    "HCLTech": ("HCLTech Integrated Annual Report 2025-26", "HCLTech.pdf"),
    "Infosys": ("Infosys Integrated Annual Report 2025-26", "infosys-ar-26.pdf"),
    "TCS": ("TCS Integrated Annual Report 2025-26", "TCS_Annual_Report_2025-26.pdf"),
}
_PL = "Consolidated statement of profit and loss"
_PROFIT = "Profit for the year (consolidated, including non-controlling interests)"
_EMPLOYEES = "BRSR table 20(a): permanent + other-than-permanent employees at fiscal year end"
_ATTRITION = "Voluntary attrition, LTM, IT services (BRSR, permanent employees)"
VERIFIED = [  # company, metric, value as printed, unit, PDF page, printed page, quote, basis
    ("HCLTech", "revenue", "130,144", "₹ crore", 375, "372", "Revenue from operations 3.19 130,144 117,055", _PL),
    ("HCLTech", "net_profit", "16,652", "₹ crore", 375, "372", "Profit for the year 16,652 17,399", _PROFIT),
    ("HCLTech", "headcount", "237,098", "employees", 216, "213",
     "Total employees (E+F) 237,098 166,670 70.29% 69,897 29.48% 531 0.22%", _EMPLOYEES),
    ("HCLTech", "attrition", "12.51", "%", 217, "214",
     "Permanent 12.61% 12.28% 8.17% 12.51% 13.06% 12.81% 5.66% 12.99%", _ATTRITION),
    ("Infosys", "revenue", "1,78,650", "₹ crore", 287, "318", "Revenue from operations 2.18 1,78,650 1,62,990", _PL),
    ("Infosys", "net_profit", "29,474", "₹ crore", 287, "318", "Profit for the year 29,474 26,750", _PROFIT),
    ("Infosys", "headcount", "3,52,095", "employees", 143, "174",
     "Total employees (D + E) 3,52,095 2,17,818 61.9 1,34,277 38.1", _EMPLOYEES),
    ("Infosys", "attrition", "12.6", "%", 144, "175", "Permanent employees 13.2 11.6 12.6 14.5 13.6 14.1",
     "Voluntary attrition, IT services, excluding BPM, products and platforms (BRSR, permanent employees)"),
    ("TCS", "revenue", "2,67,021", "₹ crore", 167, "197", "Revenue from operations 12 2,67,021 2,55,324", _PL),
    ("TCS", "net_profit", "49,454", "₹ crore", 167, "197", "PROFIT FOR THE YEAR 49,454 48,797", _PROFIT),
    ("TCS", "headcount", "6,17,437", "employees", 108, "138",
     "Total employees (D + E) 6,17,437 3,99,559 64.7 2,17,878 35.3", _EMPLOYEES),
    ("TCS", "attrition", "13.7", "%", 108, "138", "Permanent Employees 13.8 13.4 13.7 13.2 13.6 13.3", _ATTRITION),
]
# Written into the empty operating_margin rows: why that metric has no bars.
OPERATING_MARGIN_NOTE = ("Not charted: no like-for-like figure. HCLTech reports an EBIT margin adjusted for a one-time "
                         "charge (17.9%, PDF p.83); Infosys reports 20.3% consolidated (PDF p.81) and a separate "
                         "adjusted 21%; TCS reports 25.0% (PDF p.24).")


# ---------------------------------------------------------------- chart
def load_metrics(path=config.METRICS_CSV) -> pd.DataFrame:
    """Rows of metrics.csv that have a value, with a numeric column added."""
    if not path.exists():
        return pd.DataFrame(columns=FIELDS + ["value_num"])
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    for col in FIELDS:
        if col not in df.columns:
            df[col] = ""
    df["value_num"] = pd.to_numeric(df["value"].str.replace(",", "", regex=False), errors="coerce")
    return df[df["value_num"].notna()].reset_index(drop=True)


def normalise_unit(value: float, unit: str) -> tuple[float, str]:
    """Bring INR amounts to ₹ crore; leave %, counts and foreign currencies alone."""
    u = (unit or "").lower()
    if "%" in u or "per cent" in u or "percent" in u:
        return value, "%"
    if re.search(r"us\$|usd|\$", u):
        return value, unit.strip()
    if re.search(r"₹|inr|\brs\b|rupee|crore|lakh", u):
        for word, factor in _INR_UNITS:
            if re.search(rf"\b{word}\b", u):
                return value * factor, "₹ crore"
    return value, unit.strip()


def indian(n: float) -> str:
    """2,55,324-style grouping, as printed in Indian annual reports."""
    whole, frac = f"{abs(n):.2f}".split(".")
    head, tail, groups = whole[:-3], whole[-3:], []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    frac = frac.rstrip("0")
    return ("-" if n < 0 else "") + ",".join(groups + [tail]) + (f".{frac}" if frac else "")


def chart_data(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, list[str]]:
    d = df[df["metric"] == metric].copy()
    if d.empty:
        return d, ["No values for this metric yet."]
    converted = [normalise_unit(v, u) for v, u in zip(d["value_num"], d["unit"])]
    d["plot_value"] = [c[0] for c in converted]
    d["plot_unit"] = [c[1] for c in converted]
    d["fiscal_year"] = d["fiscal_year"].replace("", "FY not stated")
    order = {c: i for i, c in enumerate(config.COMPANIES)}
    d = d.sort_values(["company", "fiscal_year"], key=lambda s: s.map(order) if s.name == "company" else s)

    warnings = []
    missing = [c for c in config.COMPANIES if c not in set(d["company"])]
    if missing:
        warnings.append("No value yet for " + ", ".join(missing) + ".")
    if d["plot_unit"].nunique() > 1:
        warnings.append("Units differ (" + ", ".join(sorted(d["plot_unit"].unique())) + "), so the bars are not directly comparable.")
    years = d.groupby("company")["fiscal_year"].apply(lambda s: ", ".join(sorted(set(s))))
    if years.nunique() > 1:
        warnings.append("Fiscal years don't line up: " + "; ".join(f"{c} {y}" for c, y in years.items()) + ".")
    notes = [f"{r.company}: {r.note}" for r in d.itertuples() if r.note]
    if notes:
        warnings.append("Definitions: " + "; ".join(notes) + ".")
    failed = sorted(set(d.loc[d["auto_check"] != "pass", "company"]))
    if failed:
        warnings.append("Page check not passed for " + ", ".join(failed) + ", so no bar is drawn. Fix the row in "
                        "data/metrics.csv and run python chart.py --check.")
    unchecked = sorted(set(d.loc[d["human_check"].str.lower() != "yes", "company"]))
    if unchecked:
        warnings.append("Not yet confirmed by a person against the PDF: " + ", ".join(unchecked) + ".")
    return d, warnings


def make_figure(d: pd.DataFrame, metric: str):
    label = config.METRICS.get(metric, {}).get("label", metric)
    companies = list(dict.fromkeys(d["company"]))
    years = sorted(set(d["fiscal_year"]))
    unit = d["plot_unit"].iloc[0] if d["plot_unit"].nunique() == 1 else "mixed units"
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=120)
    width = 0.7 / len(years)
    for j, year in enumerate(years):
        sub = d[d["fiscal_year"] == year].drop_duplicates("company").set_index("company")
        for i, company in enumerate(companies):
            if company not in sub.index:
                continue
            row = sub.loc[company]
            x = i - 0.35 + width * (j + 0.5)
            color = COMPANY_COLORS.get(company, "#8D99AE") if len(years) == 1 else YEAR_COLORS[j % len(YEAR_COLORS)]
            if row["auto_check"] != "pass":  # unverified values get no bar, only a marker
                ax.annotate("not drawn:\npage check failed", (x, 0), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, color="#B23B3B")
                continue
            ax.bar(x, row["plot_value"], width * 0.9, color=color, edgecolor=color,
                   label=year if len(years) > 1 and i == 0 else None)
            shown = f"{row['plot_value']:g}%" if row["plot_unit"] == "%" else indian(row["plot_value"])
            ax.annotate(f"{shown}\nPDF p. {row['pdf_page']}", (x, row["plot_value"]), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(companies)))
    ax.set_xticklabels(companies, fontsize=10)
    ax.set_ylabel(unit)
    if unit != "%":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: indian(v)))
    ax.set_title(f"{label}, {years[0]}" if len(years) == 1 else f"{label} by fiscal year", loc="left", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.18)
    if len(years) > 1:
        ax.legend(frameon=False, fontsize=8)
    notes = {r.company: r.note for r in d.itertuples() if r.note}
    basis = ""
    if notes:
        basis = ("Basis: " + next(iter(notes.values())) if len(set(notes.values())) == 1
                 else "Basis: " + "; ".join(f"{c}: {n}" for c, n in notes.items())) + "."
    foot = "\n".join(["Source: company annual reports. Page numbers are PDF pages; see metrics.csv for quotes."]
                     + textwrap.wrap(basis, 125))
    fig.text(0.01, 0.01, foot, fontsize=7.5, color="#555555", va="bottom")
    fig.tight_layout(rect=(0, 0.02 + 0.035 * (foot.count("\n") + 1), 1, 1))
    return fig



# ---------------------------------------------------------------- strict check of chart values
# How each metric is recognised. "row": labels of its table row. "words": names that may stand
# next to the figure in a sentence. "kind": money (converted to crore), percent or count.
RULES = {
    "revenue": {"kind": "money", "row": ["revenue from operations"], "words": ["revenue"]},
    "net_profit": {"kind": "money",
                   "row": ["profit for the year", "net profit", "profit attributable", "profit after tax"],
                   "words": ["profit"]},
    "operating_margin": {"kind": "percent",
                         "row": ["operating margin", "ebit margin", "operating profit margin"],
                         "words": ["margin"]},
    "headcount": {"kind": "count", "row": ["total employees", "total headcount"],
                  "words": ["employee", "headcount", "workforce"]},
    "attrition": {"kind": "percent", "row": ["attrition", "turnover rate", "permanent employees", "permanent"],
                  "words": ["attrition", "turnover"]},
}

SCALES = {"crore": "crore", "crores": "crore", "cr": "crore", "million": "million", "mn": "million",
          "lakh": "lakh", "lakhs": "lakh", "billion": "billion", "bn": "billion"}
_SCALE = r"(crores?|cr|million|mn|lakhs?|billion|bn)\b"
# ₹ often extracts as another glyph: ` in some fonts (Infosys), H or J in others (TCS)
PAGE_UNIT = re.compile(rf"(?:₹|\brs\.?|\binr\b|`|\bh\b|\bj\b)\s*(?:in\s+)?{_SCALE}"
                       r"|\bin\s+(crores?|million|lakhs?|billion)\b")
FIGURE_UNIT = re.compile(rf"\d[\d,.]*\s*{_SCALE}")
NUM = re.compile(r"\(?\d[\d,]*(?:\.\d+)?\)?%?")
TABLE_RUN = re.compile(r"(?:\(?\d[\d,]*(?:\.\d+)?\)?%?\s+){2}\(?\d[\d,]*(?:\.\d+)?\)?%?")
YEARS = re.compile(r"(?:fy\s*)?(20\d{2})\s*[-/]\s*(\d{2}|20\d{2})(?!\d)"      # 2025-26, FY2025-2026
                   r"|\bfy\s*'?(\d{2})\s*[-/]\s*(\d{2})(?!\d)"                 # FY24-25
                   r"|\bfy\s*'?(\d{2})(?!\d)"                                   # FY26
                   r"|(?<![\d,.])(20\d{2})(?![\d,.])")                           # 2026 (March 31, 2026)
_CHARS = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2010": "-",
                        "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
                        "\u00ad": "", "\u200b": ""})


class Stop(Exception):
    """A failed check that makes the remaining checks of the row meaningless."""


# ------------------------------------------------------------------ text and numbers
def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).translate(_CHARS).lower()
    return re.sub(r"\s+", " ", text).strip()


def compact(text: str) -> str:
    """For quote matching: no whitespace or quote marks, line-break hyphens joined (as ingest.py)."""
    text = unicodedata.normalize("NFKC", str(text or "")).translate(_CHARS).lower()
    text = re.sub(r"([a-z])-\s*\n\s*([a-z])", r"\1\2", text)
    return re.sub(r"[\s\"'`]+", "", text)


def to_number(token: str) -> float | None:
    t = token.strip().rstrip(",.").rstrip("%")
    negative = t.startswith("(") and t.endswith(")")
    try:
        value = float(t.strip("()").replace(",", ""))
    except ValueError:
        return None
    return -value if negative else value


def numbers(text: str) -> list[float]:
    return [v for v in (to_number(t) for t in NUM.findall(norm(text))) if v is not None]


def figure_of(text: str) -> float | None:
    found = numbers(text)
    return found[0] if found else None


def same(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(abs(a) - abs(b)) < 1e-6


# ------------------------------------------------------------------ fiscal years
def canonical_fy(text: str) -> str:
    """'FY26', 'FY2025-26', '2025-26', 'FY 2025–2026' -> 'FY2025-26'; '' if not understood."""
    t = norm(text).replace(" ", "")
    m = re.search(r"(20\d{2})[-/](\d{2}|20\d{2})(?!\d)", t)
    if m:
        start = int(m.group(1))
        return f"FY{start}-{str(start + 1)[2:]}"
    m = re.fullmatch(r"(?:fy)?'?(\d{2})[-/](\d{2})", t)  # FY24-25, 24-25
    if m:
        start = 2000 + int(m.group(1))
        return f"FY{start}-{str(start + 1)[2:]}"
    m = re.fullmatch(r"(?:fy)?'?(\d{2}|20\d{2})", t)
    if m:
        end = int(m.group(1))
        end += 2000 if end < 100 else 0
        return f"FY{end - 1}-{str(end)[2:]}"
    return ""


def fy_end(fy: str) -> int:
    return int(fy[2:6]) + 1


def fy_name(end_year: int) -> str:
    return f"FY{end_year - 1}-{str(end_year)[2:]}"


def previous_fy(fy: str) -> str:
    return fy_name(fy_end(fy) - 1) if fy else ""


def header_years(row: str) -> list[int]:
    """Fiscal-year end years named in a table header row, left to right, e.g. [2026, 2025].
    A header holds no figures: every number in it must be a year or a day of the month."""
    t = norm(row)
    if any(not (2000 <= v <= 2100 or (v == int(v) and 0 < v <= 31)) for v in numbers(t)):
        return []
    out: list[int] = []
    for m in YEARS.finditer(t):
        if m.group(1):
            end = int(m.group(2))
            end += 2000 if end < 100 else 0
        elif m.group(3):
            end = 2000 + int(m.group(4))
        elif m.group(5):
            end = 2000 + int(m.group(5))
        else:
            end = int(m.group(6))
        if not out or out[-1] != end:
            out.append(end)
    return out


# ------------------------------------------------------------------ units
def unit_scale(text: str) -> str:
    for word in re.findall(r"[a-z]+", norm(text)):
        if word in SCALES:
            return SCALES[word]
    return ""


def stated_scale(quote: str, page_text: str) -> str:
    """Scale the evidence itself states: next to the figure in the quote, else the page's unit line."""
    m = FIGURE_UNIT.search(norm(quote))
    if m:
        return SCALES[m.group(1)]
    m = PAGE_UNIT.search(norm(page_text))
    return SCALES[m.group(1) or m.group(2)] if m else ""


def kind_from_unit(unit: str) -> str:
    u = norm(unit)
    if "%" in u or "percent" in u or "per cent" in u:
        return "percent"
    return "money" if unit_scale(u) or re.search(r"₹|\binr\b|\brs\b|rupee", u) else "count"


# ------------------------------------------------------------------ the PDFs
def page_rows(page, tol: float = 3) -> list[str]:
    """Table rows rebuilt from word positions, so a row's label and its figures share one line."""
    words = page.get_text("words")
    if page.rotation:  # landscape table on a rotated page: map words to the displayed orientation
        import pymupdf
        m = page.rotation_matrix
        words = [(*tuple(pymupdf.Rect(w[:4]) * m), w[4]) for w in words]
    words = sorted(words, key=lambda w: (w[3], w[0]))
    rows, current, last_y = [], [], None
    for w in words:
        if last_y is not None and abs(w[3] - last_y) > tol:
            rows.append(current)
            current = []
        current.append(w)
        last_y = w[3]
    rows.append(current)
    return [" ".join(x[4] for x in sorted(r, key=lambda x: x[0])) for r in rows if r]


class Reports:
    """The annual-report PDFs, opened lazily, plus the page text ingest.py stored (if present)."""

    def __init__(self, reports_dir: Path, index_dir: Path = config.INDEX_DIR):
        self.dir = Path(reports_dir)
        self._docs, self._pages, self._compact, self._fy = {}, {}, {}, {}
        self.index_text, self.titles = {}, {}
        pages_file = Path(index_dir) / "pages.jsonl"  # written by ingest.py; optional here
        if pages_file.exists():
            with open(pages_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        p = json.loads(line)
                        self.index_text[(p.get("file", ""), int(p.get("page", 0)))] = p.get("text", "")
                        self.titles.setdefault(p.get("file", ""), p.get("report", ""))

    def has(self, file: str) -> bool:
        return (self.dir / file).is_file()

    def doc(self, file: str):
        if file not in self._docs:
            import pymupdf
            self._docs[file] = pymupdf.open(self.dir / file)
        return self._docs[file]

    def page_count(self, file: str) -> int:
        return self.doc(file).page_count

    def page(self, file: str, page: int) -> tuple[str, list[str], str]:
        """(plain text, rebuilt rows, printed page label) of a 1-based page."""
        key = (file, page)
        if key not in self._pages:
            p = self.doc(file)[page - 1]
            self._pages[key] = (p.get_text(), page_rows(p), p.get_label() or "")
        return self._pages[key]

    def has_quote(self, quote: str, file: str, page: int) -> bool:
        key = (file, page)
        if key not in self._compact:
            plain, rows, _ = self.page(file, page)
            self._compact[key] = [compact(plain), compact("\n".join(rows)),
                                  compact(self.index_text.get(key, ""))]
        parts = [compact(p) for p in re.split(r"\.\.\.|\u2026", quote) if p.strip()]
        return bool(parts) and all(len(p) >= 8 and any(p in t for t in self._compact[key] if t)
                                   for p in parts)

    def report_fy(self, file: str) -> str:
        """The fiscal year a report covers: config.REPORTS, the file name, else its first pages."""
        if file not in self._fy:
            fy = canonical_fy(config.REPORTS.get(file, {}).get("fiscal_year", ""))
            if not fy:
                m = re.search(r"(20\d{2})[-_](\d{2})(?!\d)|fy[-_]?(\d{2})(?!\d)", file.lower())
                if m:
                    fy = canonical_fy(f"{m.group(1)}-{m.group(2)}" if m.group(1) else f"fy{m.group(3)}")
            if not fy:
                doc = self.doc(file)
                spans = Counter()
                for i in range(min(5, doc.page_count)):
                    for m in re.finditer(r"(20\d{2})\s*[-–/]\s*(\d{2})(?!\d)", doc[i].get_text()):
                        spans[canonical_fy(f"{m.group(1)}-{m.group(2)}")] += 1
                fy = spans.most_common(1)[0][0] if spans else ""
            self._fy[file] = fy
        return self._fy[file]

    def title(self, file: str) -> str:
        return (config.REPORTS.get(file, {}).get("title") or self.titles.get(file)
                or Path(file).stem.replace("_", " "))


def company_for_file(file: str) -> str | None:
    if config.REPORTS.get(file, {}).get("company"):
        return config.REPORTS[file]["company"]
    squashed = re.sub(r"[^a-z0-9]", "", file.lower())
    for name, aliases in config.COMPANIES.items():
        if any(re.sub(r"[^a-z0-9]", "", a) in squashed for a in aliases):
            return name
    return None


# ------------------------------------------------------------------ check 3 helpers
def label_end(row: str, labels: list[str]) -> int:
    """Index just after the metric label in a row: earliest match, and the longest label there."""
    t = norm(row)
    hits = [(t.find(k), -len(k), k) for k in labels if k in t]
    if not hits:
        return -1
    start, _, k = min(hits)
    return start + len(k)


def numbers_after(row: str, labels: list[str]) -> list[float]:
    end = label_end(row, labels)
    return numbers(norm(row)[end:]) if end >= 0 else []


def in_sentence(quote: str, names: list[str], value: float) -> bool:
    """True if one sentence of the quote names the metric and states the figure."""
    for sentence in re.split(r"(?<=[.;])\s+(?=[A-Z])|\n", quote):
        s = norm(sentence)
        if any(n in s for n in names) and any(same(value, x) for x in numbers(s)):
            return True
    return False


def is_table_row(row: str, labels: list[str]) -> bool:
    """True if only figures follow the label (a table row), not words (a sentence)."""
    end = label_end(row, labels)
    tail = re.sub(r"\([a-z][a-z0-9 +/&.-]*\)", " ", norm(row)[end:]) if end >= 0 else ""
    return bool(re.fullmatch(r"[\s\d,().%-]+", tail)) and len(numbers(tail)) >= 1


SUB_COLUMNS = {"male", "female", "others", "other", "transgender", "total"}


def find_header(rows: list[str], i: int) -> tuple[int, list[int]]:
    """(index, years) of the nearest year header above row i, or (-1, [])."""
    for j in range(i - 1, max(-1, i - 41), -1):
        years = header_years(rows[j])
        if len(years) >= 2:
            return j, years
    return -1, []


def year_groups(years: list[int]) -> list[int]:
    """[2026, 2025, 2026, 2025] -> [0, 0, 1, 1]: a new group starts when a year repeats."""
    group, seen, out = 0, set(), []
    for y in years:
        if y in seen:
            group, seen = group + 1, set()
        seen.add(y)
        out.append(group)
    return out


def block_names(rows: list[str], j: int, n_groups: int) -> list[str]:
    """['standalone', 'consolidated'] when the header (or a row just above it) names the blocks."""
    if n_groups != 2:
        return []
    for k in range(j, max(-1, j - 5), -1):
        t = norm(rows[k])
        found = sorted((t.find(w), w) for w in ("standalone", "consolidated") if w in t)
        if len(found) == 2:
            return [w for _, w in found]
    return []


def sub_columns(rows: list[str], j: int, i: int, width: int) -> list[str]:
    """Per-column sub-headers (male, female, total...) between the year header and the row."""
    for k in range(j + 1, i):
        words = [w for w in re.findall(r"[a-z]+", norm(rows[k])) if w in SUB_COLUMNS]
        if len(words) == width and "total" in words:
            return words
    return []


def check_column(kind, value, shown, fy, report_fy, rows, i, nums, fail, warn, labels) -> None:
    """In a table row, the figure must sit in the right column: the row's fiscal year, the
    consolidated block when standalone and consolidated stand side by side, and the Total
    sub-column when each year is split (male / female / total)."""
    if not is_table_row(rows[i], labels):
        return  # a sentence that happens to be on one line: no columns to check
    j, years = find_header(rows, i)
    if years and len(nums) >= len(years):
        width = len(nums) // len(years)
        cols = nums[-width * len(years):]
        groups = year_groups(years)
        names = block_names(rows, j, max(groups) + 1)
        subs = sub_columns(rows, j, i, len(cols))

        def describe(pos: int) -> str:
            b = pos // width
            parts = [fy_name(years[b])] + ([names[groups[b]]] if names else []) + ([subs[pos]] if subs else [])
            return " ".join(parts)

        if fy_end(fy) not in years:
            fail(f"support: this table has columns for {', '.join(dict.fromkeys(fy_name(y) for y in years))}, "
                 f"not {fy}")
            return
        wanted = [p for p in range(len(cols))
                  if years[p // width] == fy_end(fy)
                  and not (names and kind != "count" and names[groups[p // width]] != "consolidated")
                  and not (subs and subs[p] != "total")]
        hits = [p for p in range(len(cols)) if same(value, cols[p])]
        good = [p for p in hits if p in wanted]
        if good:
            if not names and len({p // width for p in wanted}) > 1:
                warn(f"support: the table has {len({p // width for p in wanted})} {fy} columns "
                     f"(standalone and consolidated?); confirm the block")
            elif width > 1 and not subs and good[0] % width not in (0, width - 1):
                warn("support: each year has several columns here; confirm this is the total")
            return
        target = f"; the {describe(wanted[0])} column is {indian(cols[wanted[0]])}" if wanted else ""
        if hits:
            fail(f"support: {shown} is the {describe(hits[0])} column{target}")
        else:
            fail(f"support: {shown} is not in a year column of the row (a note number?){target}")
        return
    if kind == "count":  # BRSR employee tables: Total (A) first, then male and female columns
        if nums and not same(value, nums[0]):
            if any("total (a)" in norm(r) for r in rows[max(0, i - 8):i]):  # BRSR: "Total (A)" is the first column
                fail(f"support: {shown} is not the Total column; the table puts the total first: {indian(nums[0])}")
            else:
                warn(f"support: this row starts with {indian(nums[0])}; BRSR tables give the total first")
        return
    if kind == "money" and report_fy and len(nums) >= 2:  # no header found: this year first
        current, previous = nums[-2], nums[-1]
        want, other = (current, previous) if fy == report_fy else (previous, current)
        if same(value, want):
            return
        if same(value, other):
            fail(f"support: {shown} is in the other year's column; {fy} is {indian(want)}")
        else:
            warn(f"support: no year header found above the row; {fy} is usually {indian(want)}")


def check_figure(rule, kind, value, shown, fy, report_fy, quote, rows, fail, warn) -> None:
    labelled = [(i, r) for i, r in enumerate(rows) if any(k in norm(r) for k in rule["row"])]
    for i, r in labelled:
        nums = numbers_after(r, rule["row"])
        if any(same(value, x) for x in nums):
            check_column(kind, value, shown, fy, report_fy, rows, i, nums, fail, warn, rule["row"])
            return
    is_table_text = TABLE_RUN.search(norm(quote)) is not None
    if not is_table_text and in_sentence(quote, rule["row"] + rule["words"], value):
        return
    name = rule["row"][0]
    if not is_table_text and labelled:
        fail(f"support: {shown} is not on the '{name}' row (that row reads '{labelled[0][1][:80]}'), "
             f"and no sentence of the quote states it with '{name}'")
    elif not is_table_text:
        fail(f"support: no sentence of the quote states {shown} together with '{name}'")
    elif labelled:
        fail(f"support: {shown} is not on the '{name}' row; that row reads '{labelled[0][1][:100]}'")
    else:
        fail(f"support: no '{name}' row on this page, and the quote is table text that doesn't tie "
             f"{shown} to it")


def check_unit(kind, unit, quote, page_text, fail, warn) -> None:
    u = norm(unit)
    if kind == "money":
        if re.search(r"us\$|usd|\$|eur|gbp", u):
            fail("support: the chart compares rupee figures; take the ₹ figure from the statements")
            return
        scale = unit_scale(u)
        if not scale:
            fail(f"support: unit '{unit}' gives no scale; write e.g. '₹ crore'")
            return
        if scale in ("million", "billion") and not re.search(r"₹|\binr\b|\brs\b|rupee", u):
            fail(f"support: write the unit as '₹ {scale}' so chart.py converts it to crore")
        stated = stated_scale(quote, page_text)
        if not stated:
            warn("support: the page doesn't state its unit; check it by eye")
        elif stated != scale:
            fail(f"support: the page states amounts in ₹ {stated}, the unit column says {scale}")
    elif kind == "percent":
        if not any(s in u for s in ("%", "percent", "per cent")):
            fail(f"support: write the unit as '%' (got '{unit}') so chart.py plots a percentage")
        page = norm(page_text)
        if not any(s in norm(quote) for s in ("%", "percent", "per cent")) and \
                not any(s in page for s in ("(%)", "in %", "(in %)", "%)", "percent")):
            warn("support: neither the quote nor the page marks the figure as a percentage")
    elif unit_scale(u) or re.search(r"₹|\binr\b|\brs\b|rupee", u):
        fail(f"support: write the full count (e.g. 2,23,420) with a plain unit like 'employees', "
             f"not '{unit}'")


def check_scope(page_text: str, page: int, fail, warn) -> None:
    t = norm(page_text)
    consolidated, standalone = "consolidated" in t, "standalone" in t
    if standalone and (not consolidated or
                       ("standalone statement" in t and "consolidated statement" not in t)):
        fail(f"support: page {page} belongs to the standalone statements; the chart uses consolidated figures")
    elif not consolidated:
        warn(f"support: page {page} doesn't say 'consolidated'; confirm by eye")


# ------------------------------------------------------------------ one row
def _check(row: dict, res: dict, reports: Reports, fail, warn) -> None:
    company, metric = row["company"].strip(), row["metric"].strip().lower()
    rule = RULES.get(metric)
    file = Path(row.get("file", "").strip()).name

    # 1 source
    if not file:
        fail("source: no file named", fatal=True)
    if not reports.has(file):
        fail(f"source: {file} is not in {reports.dir}", fatal=True)
    if company not in config.COMPANIES:
        fail(f"source: company '{company}' is not one of {', '.join(config.COMPANIES)}")
    owner = company_for_file(file)
    if owner and owner != company:
        fail(f"source: {file} is {owner}'s report, not {company}'s")
    fy = canonical_fy(row.get("fiscal_year", ""))
    if not fy:
        fail(f"source: fiscal_year '{row.get('fiscal_year', '')}' not understood (write e.g. FY2025-26)",
             fatal=True)
    res["fy"] = fy
    report_fy = reports.report_fy(file)
    if report_fy and fy not in (report_fy, previous_fy(report_fy)):
        fail(f"source: {file} covers {report_fy} (with {previous_fy(report_fy)} comparatives), not {fy}")

    # 2 page
    try:
        page = int(float(row.get("pdf_page", "")))
    except ValueError:
        fail(f"page: pdf_page '{row.get('pdf_page', '')}' is not a number", fatal=True)
    n = reports.page_count(file)
    if not 1 <= page <= n:
        fail(f"page: page {page} is outside 1..{n}", fatal=True)
    plain, rows, label = reports.page(file, page)
    res["page_label"] = label
    quote = row.get("quote", "").strip()
    if len(compact(quote)) < 8:
        fail("page: no quote; copy the line that holds the figure from the page", fatal=True)
    if not reports.has_quote(quote, file, page):
        elsewhere = [p for p in range(1, n + 1) if p != page and reports.has_quote(quote, file, p)][:3]
        fail(f"page: quote not found on page {page}"
             + (f"; it is on page {', '.join(map(str, elsewhere))}" if elsewhere else "; it is not in this report"),
             fatal=True)

    # 3 support
    text, plain_value = row.get("value_text", "").strip(), row.get("value", "").strip()
    shown = text or plain_value
    value = figure_of(shown)
    if value is None:
        fail(f"support: value '{shown}' is not a number", fatal=True)
    if text and plain_value and not same(figure_of(plain_value), value):
        fail(f"support: value {plain_value} and value_text {text} differ")
    res["value"] = value
    if not any(same(value, x) for x in numbers(quote)):
        fail(f"support: {shown} is not in the quote", fatal=True)
    kind = rule["kind"] if rule else kind_from_unit(row.get("unit", ""))
    if rule:
        check_figure(rule, kind, value, shown, fy, report_fy, quote, rows, fail, warn)
    else:
        warn(f"support: no rule for metric '{metric}' in chart.RULES; only quote and page checked")
    if kind == "percent" and not 0 <= value <= 100:
        fail(f"support: {shown} is not a percentage between 0 and 100")
    check_unit(kind, row.get("unit", ""), quote, plain, fail, warn)
    if kind == "money":
        check_scope(plain, page, fail, warn)


def check_row(row: dict, reports: Reports) -> dict:
    res = {"errors": [], "warnings": [], "value": None, "fy": "", "page_label": ""}

    def fail(message: str, fatal: bool = False) -> None:
        res["errors"].append(message)
        if fatal:
            raise Stop

    try:
        _check(row, res, reports, fail, res["warnings"].append)
    except Stop:
        pass
    except Exception as exc:  # an unreadable PDF must not stop the other rows
        res["errors"].append(f"error: {type(exc).__name__}: {exc}")
    return res


# ------------------------------------------------------------------ the whole file
def has_value(row: dict) -> bool:
    return bool(row.get("value_text", "").strip() or row.get("value", "").strip())


def validate(rows: list[dict], reports: Reports) -> tuple[list[dict | None], list[str]]:
    results = [check_row(r, reports) if has_value(r) else None for r in rows]

    # 4 coverage: one row per company, metric and year
    groups = defaultdict(list)
    for i, (row, res) in enumerate(zip(rows, results)):
        if res is not None:
            fy = res["fy"] or row.get("fiscal_year", "").strip()
            groups[(row["company"].strip(), row["metric"].strip().lower(), fy)].append(i)
    for key, idx in groups.items():
        if len(idx) > 1:
            lines = ", ".join(str(i + 2) for i in idx)
            for i in idx:
                results[i]["errors"].append(f"coverage: {key[0]} {key[1]} {key[2]} appears on CSV lines {lines}; keep one")
    notes = []
    metrics = dict.fromkeys(r["metric"].strip().lower() for r, res in zip(rows, results) if res is not None)
    for metric in metrics:
        passing = {r["company"].strip() for r, res in zip(rows, results)
                   if res is not None and not res["errors"] and r["metric"].strip().lower() == metric}
        missing = [c for c in config.COMPANIES if c not in passing]
        if missing:
            notes.append(f"coverage: {metric} has no passing value for {', '.join(missing)}")
    return results, notes


def write_report(path: Path, rows, results, notes, reports: Reports) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checked = [(i, r, res) for i, (r, res) in enumerate(zip(rows, results)) if res is not None]
    failed = sum(bool(res["errors"]) for _, _, res in checked)
    lines = ["# Chart value validation (python chart.py --check)", "",
             f"{len(checked)} filled rows: {len(checked) - failed} pass, {failed} fail. "
             "Failed rows are marked in auto_check; chart.py flags them.", "",
             "Checks: 1 source, 2 page, 3 support (figure on the metric's row or sentence, in the right "
             "year column; unit and consolidated scope match the page), 4 coverage.", "",
             "| CSV line | Company | Metric | FY | Value | Unit | Report | PDF page | Result | Details | Human check |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    cell = lambda s: str(s).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    for i, r, res in checked:
        result = "fail" if res["errors"] else "pass"
        details = "; ".join(res["errors"] + [f"warning: {w}" for w in res["warnings"]])
        title = reports.title(Path(r.get("file", "")).name) if r.get("file") else ""
        lines.append(f"| {i + 2} | {cell(r['company'])} | {cell(r['metric'])} | {cell(res['fy'] or r.get('fiscal_year', ''))} "
                     f"| {cell(r.get('value_text') or r.get('value'))} | {cell(r.get('unit', ''))} | {cell(title)} "
                     f"| {cell(r.get('pdf_page', ''))} | {result} | {cell(details)} | {cell(r.get('human_check', '') or '-')} |")
    if notes:
        lines += ["", "## Coverage", ""] + [f"- {n}" for n in notes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- run the check / fill empty rows
def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = [{k: (v or "") for k, v in r.items() if k} for r in reader]
    return header + [c for c in FIELDS if c not in header], rows


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows({c: row.get(c, "") for c in header} for row in rows)


def fill_verified(path: Path = config.METRICS_CSV, reports_dir: Path = config.REPORTS_DIR) -> int:
    """Put the verified values into rows of metrics.csv that have no value yet. Never overwrites a row
    that has a value, never touches human_check, and skips reports that are not in data/reports."""
    header, rows = _read_csv(path) if path.exists() else (list(FIELDS), [])
    by_key = {(r.get("company", "").strip(), r.get("metric", "").strip().lower()): r for r in rows}
    filled, kept, missing = 0, 0, set()
    for company, metric, value_text, unit, page, printed, quote, note in VERIFIED:
        title, file = VERIFIED_REPORTS[company]
        if not (Path(reports_dir) / file).is_file():
            missing.add(file)
            continue
        row = by_key.get((company, metric))
        if row is None:
            row = {c: "" for c in header} | {"company": company, "metric": metric}
            rows.append(row)
            by_key[(company, metric)] = row
        elif has_value(row):
            kept += 1
            continue
        row.update(value=value_text.replace(",", ""), value_text=value_text, unit=unit, fiscal_year=VERIFIED_FY,
                   report=title, file=file, pdf_page=str(page), page_label=printed, quote=quote, note=note)
        filled += 1
    if not missing:
        for (company, metric), row in by_key.items():
            if metric == "operating_margin" and not has_value(row) and not row.get("note", "").strip():
                row["note"] = OPERATING_MARGIN_NOTE
    if filled or not path.exists():
        _write_csv(path, header, rows)
    parts = [f"{filled} verified value(s) added to {path.name}"] if filled else []
    if kept:
        parts.append(f"{kept} row(s) already had a value and were kept")
    if missing:
        parts.append("not added (report not in data/reports): " + ", ".join(sorted(missing)))
    if parts:
        print("; ".join(parts))
    return filled


def check_metrics(path: Path = config.METRICS_CSV, reports_dir: Path = config.REPORTS_DIR,
                  out: Path = config.OUTPUT_DIR / "metrics_validation.md", write: bool = True):
    """Strict check of every filled row; writes auto_check (and value, page_label, report when empty)."""
    if not path.exists():
        raise SystemExit(f"{path} not found. Run python extract_metrics.py, or put the reports in "
                         f"data/reports and run python chart.py.")
    header, rows = _read_csv(path)
    reports = Reports(reports_dir)
    results, notes = validate(rows, reports)
    for row, res in zip(rows, results):
        if res is None:
            continue
        row["auto_check"] = f"fail: {res['errors'][0]}" if res["errors"] else "pass"
        if res["value"] is not None and not row.get("value", "").strip():
            row["value"] = indian(res["value"]).replace(",", "")
        if res["page_label"] and not row.get("page_label", "").strip():
            row["page_label"] = res["page_label"]
        if row.get("file") and not row.get("report", "").strip() and reports.has(Path(row["file"]).name):
            row["report"] = reports.title(Path(row["file"]).name)
    if write:
        _write_csv(path, header, rows)
    write_report(out, rows, results, notes, reports)

    checked = [(i, r, res) for i, (r, res) in enumerate(zip(rows, results)) if res is not None]
    failed = sum(bool(res["errors"]) for _, _, res in checked)
    print(f"Checked {len(checked)} filled rows: {len(checked) - failed} pass, {failed} fail")
    for i, r, res in checked:
        if res["errors"] or res["warnings"]:
            print(f"  line {i + 2:<3} {r['company']} {r['metric']} {r.get('value_text') or r.get('value')}: "
                  f"{'FAIL' if res['errors'] else 'pass'}")
            for message in res["errors"] + [f"warning: {w}" for w in res["warnings"]]:
                print(f"           {message}")
    for note in notes:
        print(f"  {note}")
    return rows, results, notes


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fill and check metrics.csv only; draw no charts")
    ap.add_argument("--no-fill", action="store_true", help="don't add the verified values to empty rows")
    args = ap.parse_args(argv or [])
    if not args.no_fill:
        fill_verified()
    if config.METRICS_CSV.exists():
        check_metrics()
    if args.check:
        return
    df = load_metrics()
    if df.empty:
        raise SystemExit("metrics.csv has no values yet. Run python extract_metrics.py or fill it by hand.")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for metric in dict.fromkeys(df["metric"]):
        data, warnings = chart_data(df, metric)
        fig = make_figure(data, metric)
        path = config.OUTPUT_DIR / f"r1_{metric}.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"saved {path}")
        for w in warnings:
            print(f"  note: {w}")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    main(sys.argv[1:])