# Annual-report analyst: HCLTech, Infosys, TCS

HCLTech AI Engineer Hackathon, IIT Gandhinagar, Use Case 2.

Compares three NSE-listed IT companies using only their annual reports. **R1**: a comparison chart where every bar carries the PDF page its value came from. **R2**: a question box whose every sentence is cited to a company, report and page, and which answers "Not found in the reports." when the evidence isn't there.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                       # add one LLM key; with no key the app runs extractively

# 1. Put the three annual-report PDFs in data/reports/ (file names must contain hcl / infosys / tcs)
python ingest.py                     # parse, chunk, embed; a few minutes on CPU for ~1,200 pages

# 2. Chart data (R1)
python extract_metrics.py            # LLM finds each figure + page + quote; each row is auto-checked against the page
#    open every cited page, confirm the value, write "yes" in the human_check column of data/metrics.csv
python extract_metrics.py --verify   # re-check after any hand edit
python chart.py                      # static PNGs in outputs/

# 3. Use it
streamlit run app.py
python ask.py "What attrition rate did HCLTech report?"
python evaluate.py                   # demo set: answerable + unanswerable questions, with latency
```

No PDFs or keys yet? `python tests/test_verifier.py` and `python tests/smoke_test.py` check the whole pipeline on synthetic PDFs in about 10 seconds.

## Architecture

```mermaid
flowchart TD
    PDF["3 annual-report PDFs<br/>HCLTech, Infosys, TCS"] --> P["PyMuPDF parser<br/>page-bounded chunks"]
    P --> IDX[("Vector index<br/>embeddings + BM25")]
    P --> PG[("Page store<br/>raw text per PDF page")]
    PDF -. extract_metrics.py .-> M["metrics.csv<br/>value, unit, FY, page, quote<br/>auto-check + human check"]
    M --> R1["Static chart (R1)"]

    Q[Question] --> S{"Scope guard"}
    S -- "advice / other company" --> NF["Not found in the reports"]
    S --> H["Hybrid retrieval<br/>cosine + BM25, per company"]
    H --> G{"Evidence gate<br/>(before the LLM)"}
    G -- pass --> L["LLM, one call<br/>JSON claims + verbatim quotes"]
    L --> V{"Static verifier<br/>4 checks, no LLM"}
    V -- pass --> A["Answer + page citations"]
    G -- fail --> RT["Single retry<br/>wider pool + reranker + feedback"]
    V -- fail --> RT
    RT --> G2{"Gate"} --> L2["LLM"] --> V2{"Verifier"}
    V2 -- pass --> A
    V2 -- fail --> NF
    PG -. "check 2: quote on that page" .-> V
```

## The two design problems from our sketch

**Verifier position.** It now sits on both sides of the LLM, and neither side uses an LLM:

- *Before generation*: a scope guard (share-price predictions, investment advice, companies outside the corpus) and an evidence gate (do the retrieved pages contain the question's key terms at all?). Both refuse without spending an LLM call.
- *After generation*: the static verifier. The LLM must return claims, each with an excerpt id and a verbatim quote, and only claims that pass all four checks are shown. If nothing passes, there is one retry with a wider, reranked candidate pool and the verifier's reasons as feedback; after that, "Not found in the reports."

**Generation delay.**

- One LLM call on the happy path; verification is code and takes milliseconds.
- Out-of-scope and off-topic questions are refused before any LLM call (well under a second).
- Small prompts (6 to 9 page-bounded excerpts), short JSON output, a small fast model at low effort.
- The retry calls the LLM again only if reranking found new pages or there is verifier feedback to act on.
- The index is built offline, the reranker is loaded at startup, and repeated questions come from a cache.
- The chart never calls an LLM at demo time; its values are read from metrics.csv.

## Verifier checks (verifier.py)

| Check | What must hold for a claim to be shown |
| --- | --- |
| 1. Source | Every cited excerpt exists, and each company named in the claim is backed by that company's own report |
| 2. Page | Every quote is found again in the stored text of the cited PDF page; any page number written in the claim is a cited page |
| 3. Support | Every quote is verbatim in its excerpt, and every figure in the claim appears inside a quote |
| 4. Coverage | Every claim carries evidence; for comparisons, companies with no verified claim are listed as not found |

## Demo (R2: three answerable, two unanswerable)

The app's demo questions come from `eval_questions.json` (entries with `"demo": true`). Run `python evaluate.py` on your PDFs before the demo, keep the answerable questions that your reports actually support, and expect judges to add an unanswerable one of their own.

## Settings worth knowing

`config.py`: `TOP_K` / `TOP_K_PER_COMPANY` (excerpts per question), `RETRY_ON_NOT_FOUND` (recall against refusal speed), `MIN_TERM_COVERAGE` (evidence-gate strictness), `CHUNK_CHARS`, and `REPORTS` (optional nicer titles for citations). `.env`: LLM provider, model and key.

## Source register

| Company | Report title | Fiscal year | URL | Downloaded on | File in data/reports |
| --- | --- | --- | --- | --- | --- |
| HCLTech | | | | | |
| Infosys | | | | | |
| TCS | | | | | |

## Failed attempts

Record every attempt that didn't work, at the end of each iteration.

| Iteration | What we tried | What happened | What we changed |
| --- | --- | --- | --- |
| 1 | | | |
| 2 | | | |
| 3 | | | |

## Team contributions

- **Member 1 (name):** three sentences.
- **Member 2 (name):** three sentences.
- **Member 3 (name):** three sentences.

## Repository layout

| File | Role |
| --- | --- |
| `config.py` | paths, companies, retrieval and LLM settings, chart metrics |
| `ingest.py` | PDF to page-bounded chunks, page store, embeddings |
| `embeddings.py` | sentence-transformers embeddings, TF-IDF fallback |
| `retriever.py` | hybrid cosine + BM25 search, per-company routing, reranker |
| `qa.py` | scope guard, evidence gate, generation, retry, answer assembly |
| `verifier.py` | the four static checks |
| `llm.py` | Anthropic or any OpenAI-compatible provider, or none |
| `extract_metrics.py` | R1 data with page checks |
| `chart.py` | R1 chart |
| `app.py` | Streamlit UI |
| `ask.py`, `evaluate.py` | command line and evaluation |
| `tests/` | verifier unit tests, end-to-end smoke test on synthetic PDFs |

## Known limitations

- The static checks prove a claim is anchored to real text on the cited page; they cannot judge comparative wording such as "highest".
- Citations use PDF page numbers. The printed page number is shown too when the PDF defines page labels.
- Multi-column layouts can interleave text. Quotes are still verified against the same extracted text the LLM saw.
