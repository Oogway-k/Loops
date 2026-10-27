<!-- # REPORT: Use Case 2, IT Companies in India

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

**8. Failed attempts.** [Summarise from the README log.] -->


# REPORT: Use Case 2, IT Companies in India

**1. Approach.** We compare the FY2025-26 annual reports of HCLTech, Infosys and TCS. PyMuPDF reads each page layout-aware (column order for prose, row order for number-heavy pages), drops running headers and footers, and tags each page with its section heading. Chunks never cross a page, so every chunk carries one PDF page; tables also go to a structured store (tables.jsonl). A question passes a scope guard (share-price, advice, other companies), hybrid retrieval (one FAISS index per company plus BM25, fused by reciprocal rank, with table excerpts added for figure questions), an evidence gate, one local-LLM call returning claims with verbatim quotes, and a four-check deterministic verifier (source, page, support, coverage). Failures get one reranked retry, then "Not found in the reports." R1 charts 12 values (revenue, net profit, headcount, attrition), each re-checked against its PDF page, row and year column.

**2. Tools.** Python, PyMuPDF, faiss-cpu, rank-bm25, sentence-transformers (bge-small-en-v1.5; MiniLM cross-encoder), Hugging Face Transformers, bitsandbytes, rapidfuzz, Streamlit, matplotlib. LLM: Qwen2.5-3B-Instruct, 4-bit, local on an RTX 4050; extractive fallback when no model loads.

**3. Constraints.** 6.5 hours; 8 GB VRAM; no external APIs, so no data leaves the laptop. Small models follow JSON less reliably; the verifier compensates. The checks prove anchoring, not comparative wording.

**4. Assumptions.** Machine-readable PDFs; same fiscal year; consolidated Ind AS figures; PDF page numbers, with printed labels shown. Operating margin is not charted: the three companies define it differently. Attrition definitions differ and are stated on the chart.

**5. Top design decisions.**
1. Page-bounded chunks make citations exact by construction.
2. Deterministic verifier after generation, cheap gate before it: one LLM call, no LLM judge.
3. Per-company FAISS indexes stop cross-company misattribution.
4. Hybrid retrieval catches exact terms and figures embeddings miss.
5. Chart values are curated and page-checked, never generated at demo time.

**6. AI assistance.** Claude (Anthropic) drafted the scaffold, parser, retrieval, verifier and tests; we reviewed each module, ran evaluate.py and checked cited pages by hand.

**7. Prior work.** RAG (Lewis et al., 2020); reciprocal-rank fusion (Cormack et al., 2009); BM25 (Robertson and Zaragoza, 2009); FAISS (Johnson et al., 2019).

**8. Failed attempts.** Raw PyMuPDF order interleaved columns, so we use layout-aware reading. Table detection on every page was too slow, so it runs on number-heavy pages only. Qwen3.5-4B needed development Transformers and a 10 GB download, so we switched to Qwen2.5-3B. At freeze the local model failed to load (missing accelerate), and R2 ran in extractive mode with citations and refusals intact.