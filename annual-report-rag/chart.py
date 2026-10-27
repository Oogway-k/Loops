"""R1: comparison chart from data/metrics.csv. Every bar carries the PDF page its value came from, and any
mismatch in units, fiscal years or definitions is stated next to the chart.

    python chart.py     saves outputs/r1_<metric>.png for every metric that has values
"""
from __future__ import annotations

import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

FIELDS = config.METRIC_FIELDS
COMPANY_COLORS = {"HCLTech": "#1F5A96", "Infosys": "#2A9D8F", "TCS": "#6A4C93"}
YEAR_COLORS = ["#1F5A96", "#2A9D8F", "#6A4C93", "#E9A23B", "#8D99AE"]
_INR_UNITS = [("crore", 1.0), ("cr", 1.0), ("lakh", 0.01), ("million", 0.1), ("mn", 0.1), ("billion", 100.0), ("bn", 100.0)]


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
        warnings.append("Page check not passed for " + ", ".join(failed) + ". Run python extract_metrics.py --verify.")
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
            checked = row["auto_check"] == "pass"
            ax.bar(x, row["plot_value"], width * 0.9, color=color, alpha=1.0 if checked else 0.35,
                   hatch=None if checked else "//", edgecolor=color, label=year if len(years) > 1 and i == 0 else None)
            shown = f"{row['plot_value']:.1f}%" if row["plot_unit"] == "%" else indian(row["plot_value"])
            status = "" if checked else "\n(page check failed)"
            ax.annotate(f"{shown}\nPDF p. {row['pdf_page']}{status}", (x, row["plot_value"]), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(companies)))
    ax.set_xticklabels(companies, fontsize=10)
    ax.set_ylabel(unit)
    ax.set_title(f"{label}, {years[0]}" if len(years) == 1 else f"{label} by fiscal year", loc="left", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.18)
    if len(years) > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "Source: company annual reports. Page numbers are PDF pages; see metrics.csv for quotes.",
             fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def main() -> None:
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
    main()
