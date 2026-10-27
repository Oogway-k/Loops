"""Hybrid retrieval: FAISS cosine search fused with BM25 by reciprocal-rank fusion.

Annual reports mix prose with exact terms and figures ("LTM attrition", "6,07,979"), so keyword search (BM25)
catches what embeddings miss and vice versa. Each company has its own FAISS index, so a TCS question never
gets Infosys pages, and comparison questions get a balanced set from each report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from index_store import VectorStore, load_bm25
from textutil import key_terms, stem_tokens, term_in


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pages(index_dir: Path = config.INDEX_DIR) -> dict[tuple[str, int], dict]:
    return {(p["file"], p["page"]): p for p in read_jsonl(Path(index_dir) / "pages.jsonl")}


@dataclass
class Hit:
    chunk: dict
    score: float  # fused (RRF) score
    dense: float  # cosine similarity (0 if the chunk came only from BM25)
    bm25: float


class Retriever:
    def __init__(self, index_dir: Path = config.INDEX_DIR):
        index_dir = Path(index_dir)
        self.chunks = read_jsonl(index_dir / "chunks.jsonl")
        self.pages = load_pages(index_dir)
        self.meta = json.loads((index_dir / "meta.json").read_text()) if (index_dir / "meta.json").exists() else {}
        self.store = VectorStore(index_dir, self.chunks)
        self.embedder = self.store.embedder
        self.bm25 = load_bm25(index_dir, self.chunks)
        self._by_company = self.store.by_company
        self.companies = [c for c in config.COMPANIES if c in self._by_company]
        self._reranker = None
        self.reranker_status = "not loaded"

    def page_text(self, file: str, page: int) -> str:
        record = self.pages.get((file, int(page)))
        return record["text"] if record else ""

    # ------------------------------------------------------------ search
    def _bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(stem_tokens(query)), dtype=float)

    def _bm25_top(self, scores: np.ndarray, companies: list[str], pool: int) -> list[int]:
        parts = [self._by_company[c] for c in companies if c in self._by_company]
        if not parts:
            return []
        idx = np.concatenate(parts)
        return [int(i) for i in idx[np.argsort(-scores[idx], kind="stable")][:pool]]

    def _fuse(self, dense: list[tuple[int, float]], bm_top: list[int], bm25: np.ndarray, k: int) -> list[Hit]:
        fused: dict[int, float] = {}
        for ranking in ([i for i, _ in dense], bm_top):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank + 1)
        dense_score = dict(dense)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.chunks[i], fused[i], dense_score.get(i, 0.0), float(bm25[i])) for i in best]

    def search(self, query: str, k: int = config.TOP_K, companies: list[str] | None = None,
               pool: int = config.CANDIDATE_POOL) -> list[Hit]:
        companies = [c for c in (companies or self.companies) if c in self._by_company]
        if not companies:
            return []
        pool = max(pool, k)
        q, bm25 = self.store.encode(query), self._bm25_scores(query)
        return self._fuse(self.store.search(q, companies, pool), self._bm25_top(bm25, companies, pool), bm25, k)

    def search_per_company(self, query: str, companies: list[str], k_each: int,
                           pool: int = config.CANDIDATE_POOL) -> list[Hit]:
        q, bm25 = self.store.encode(query), self._bm25_scores(query)  # encode once, reuse per company
        pool = max(pool, k_each)
        hits: list[Hit] = []
        for company in companies:
            if company in self._by_company:
                hits += self._fuse(self.store.search(q, [company], pool),
                                   self._bm25_top(bm25, [company], pool), bm25, k_each)
        return hits

    # ------------------------------------------------------------ rerank (used on the single retry)
    def _get_reranker(self):
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(config.RERANK_MODEL, max_length=512)
                self.reranker_status = f"cross-encoder {config.RERANK_MODEL}"
            except Exception as exc:
                self._reranker = False
                self.reranker_status = f"lexical fallback ({type(exc).__name__})"
        return self._reranker or None

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        model = self._get_reranker()
        if model is not None:
            scores = np.asarray(model.predict([(query, h.chunk["text"]) for h in hits]))
            order = [int(i) for i in np.argsort(-scores)]
        else:  # no cross-encoder: favour chunks that contain more of the question's key terms
            terms = key_terms(query)
            coverage = [sum(term_in(t, h.chunk["text"].lower()) for t in terms) + 10 * h.score for h in hits]
            order = sorted(range(len(hits)), key=lambda i: coverage[i], reverse=True)
        return [hits[i] for i in order[:k]]