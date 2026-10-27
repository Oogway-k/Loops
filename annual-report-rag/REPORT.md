# REPORT: Use Case 2, IT Companies in India

**1. Approach.** PyMuPDF parses the HCLTech, Infosys and TCS annual reports ([FY]) into page-bounded chunks, so every chunk carries one PDF page. Questions pass a scope guard, hybrid retrieval (cosine + BM25, per company), an evidence gate, one LLM call returning claims with verbatim quotes, and a four-check deterministic verifier. Failures get one reranked retry, then "Not found in the reports." Chart values live in metrics.csv with page and quote, auto-checked and human-confirmed.

**2. Tools.** Python, PyMuPDF, sentence-transformers (bge-small-en-v1.5, MiniLM cross-encoder), rank-bm25, scikit-learn, rapidfuzz, Streamlit, matplotlib; LLM: [provider/model].

**3. Constraints.** 6.5 hours, CPU only. Static checks prove anchoring, not comparative wording. Text extraction loses some table layout.

**4. Assumptions.** Machine-readable reports, same fiscal year, consolidated Ind AS figures; citations use PDF page numbers.

**5. Top design decisions.**
1. Page-bounded chunks make page citations exact by construction.
2. Deterministic verifier after generation, cheap gate before it: one LLM call per answer, no LLM-judge delay.
3. Per-company routing, enforced by check 1, stops cross-company misattribution.
4. Hybrid retrieval catches exact terms and figures that embeddings miss.
5. Chart values are curated, page-checked data, never generated at demo time.

**6. AI assistance.** Claude (Anthropic) drafted the pipeline scaffold and tests; we [reviewed, changed, verified: describe].

**7. Prior work.** Retrieval-augmented generation (Lewis et al., 2020); reciprocal-rank fusion (Cormack et al., 2009); BM25 (Robertson and Zaragoza, 2009).

**8. Failed attempts.** [Summarise from the README log.]
