"""Streamlit UI: R1 comparison chart + R2 grounded Q&A.      Run:  streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import config
from chart import chart_data, load_metrics, make_figure
from qa import NOT_FOUND, Answer, QAEngine, cite

st.set_page_config(page_title="Annual reports: compare and ask", page_icon="📑", layout="wide")


@st.cache_resource(show_spinner="Loading the report index...")
def get_engine() -> QAEngine:
    return QAEngine()


def md(text: str) -> str:
    return str(text).replace("$", "\\$")  # stop "US$ ... $" from rendering as maths


def demo_questions() -> list[str]:
    path = Path(__file__).with_name("eval_questions.json")
    if not path.exists():
        return []
    return [q["question"] for q in json.loads(path.read_text(encoding="utf-8")) if q.get("demo")]


def queue_question(q: str) -> None:
    st.session_state["question"] = q
    st.session_state["auto_ask"] = True


def show_answer(ans: Answer) -> None:
    llm_errors = [line.split("  ", 1)[-1] for line in ans.trace if "LLM error" in line]
    if llm_errors:
        st.warning(f"The LLM call failed, so this answer was made in extractive mode. {llm_errors[-1]}")
    if ans.found:
        st.success("Answer. Every sentence passed the four citation checks.")
        for claim in ans.claims:
            sources = "; ".join(dict.fromkeys(cite(e["chunk"]) for e in claim.evidence))
            st.markdown(f"{md(claim.text)}  \n:gray[{md(sources)}]")
        for note in ans.notes:
            st.info(note)
        with st.expander("Evidence quotes"):
            for claim in ans.claims:
                for e in claim.evidence:
                    st.markdown(f"> {md(e['quote'])}\n\n{md(cite(e['chunk']))}")
    else:
        st.error(f"**{NOT_FOUND}**")
        st.caption(ans.reason)
    with st.expander("How this answer was checked"):
        st.code("\n".join(ans.trace) or "(no steps recorded)", language=None)
    st.caption(f"Answered in {ans.seconds:.1f} s by {ans.mode}.")


st.title("HCLTech, Infosys and TCS annual reports")
st.caption("Every figure and answer points to the PDF page it came from. "
           "When the reports don't say something, the app answers “Not found in the reports.”")

if not (config.INDEX_DIR / "chunks.jsonl").exists():
    st.error("There's no report index yet. Put the annual-report PDFs in data/reports/, "
             "run `python ingest.py`, then reload this page.")
    st.stop()

tab_chart, tab_ask, tab_sources = st.tabs(["Compare (R1)", "Ask the reports (R2)", "Sources"])

with tab_chart:
    df = load_metrics()
    if df.empty:
        st.info("The chart needs values first. Run `python extract_metrics.py` (or fill data/metrics.csv by hand), "
                "then `python extract_metrics.py --verify`.")
    else:
        metrics = [m for m in config.METRICS if m in set(df["metric"])] + \
                  [m for m in dict.fromkeys(df["metric"]) if m not in config.METRICS]
        metric = st.selectbox("Metric", metrics, format_func=lambda m: config.METRICS.get(m, {}).get("label", m))
        data, warnings = chart_data(df, metric)
        left, right = st.columns([3, 2])
        with left:
            fig = make_figure(data, metric)
            st.pyplot(fig)
            plt.close(fig)
        with right:
            for w in warnings:
                st.warning(w)
            st.dataframe(data[["company", "value_text", "unit", "fiscal_year", "pdf_page", "page_label",
                               "auto_check", "human_check"]], hide_index=True)
        with st.expander("Quote behind each bar"):
            for row in data.itertuples():
                quote = f"“{md(row.quote)}”" if row.quote else "(no quote recorded)"
                source = ", ".join(x for x in (row.report, f"PDF page {row.pdf_page}") if x)
                st.markdown(f"**{row.company}**, {md(source)}: {quote}")

with tab_ask:
    engine = get_engine()
    questions = demo_questions()
    if questions:
        with st.expander("Demo questions"):
            for i, q in enumerate(questions):
                st.button(q, key=f"demo_{i}", on_click=queue_question, args=(q,))
    question = st.text_input("Question", key="question",
                             placeholder="For example: what attrition rate did HCLTech report?")
    clicked = st.button("Ask", type="primary")
    if (clicked or st.session_state.pop("auto_ask", False)) and question.strip():
        with st.spinner("Searching the reports and checking every citation..."):
            answer = engine.ask(question)
        show_answer(answer)

with tab_sources:
    meta_path = config.INDEX_DIR / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        st.write(f"Index built {meta['created']}: {meta['chunks']} chunks, retrieval with {meta['model']} + BM25.")
        st.dataframe(pd.DataFrame(meta["files"]), hide_index=True)
    if config.REPORTS:
        st.dataframe(pd.DataFrame([{"file": f, **d} for f, d in config.REPORTS.items()]), hide_index=True)
    st.caption("The full source register (title, fiscal year, URL, download date) is in README.md.")
