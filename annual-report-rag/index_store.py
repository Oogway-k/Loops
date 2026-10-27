"""Retrieval stores: built by ingest.py, loaded by retriever.py.

dense : FAISS, one exact inner-product index per company (IndexIDMap2 over IndexFlatIP). Vectors are
        L2-normalised, so inner product == cosine. FAISS ids are global chunk positions, so a hit maps straight
        back to chunks.jsonl, and per-company indexes give company routing with no post-filtering.
        Falls back to a numpy matrix if faiss isn't installed.
tfidf : sparse TF-IDF matrix (FAISS needs dense vectors), used when the dense model can't be loaded.
bm25  : BM25Okapi over the same text, pickled so the app doesn't rebuild it at startup.
The indexed text is "company | section" + chunk text; the chunk text itself (what the verifier checks) is unchanged.
"""
from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # faiss + torch both ship OpenMP; avoids a Windows/macOS crash

import numpy as np  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

import config  # noqa: E402
from embeddings import Embedder  # noqa: E402
from textutil import stem_tokens  # noqa: E402

try:
    import faiss

    faiss.omp_set_num_threads(config.FAISS_THREADS)
except ImportError:
    faiss = None


def index_text(chunk: dict) -> str:
    section = chunk.get("section") or ""
    return f"{chunk['company']} | {section}\n{chunk['text']}" if section else f"{chunk['company']}\n{chunk['text']}"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name)


def _by_company(chunks: list[dict]) -> dict[str, np.ndarray]:
    out: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        out.setdefault(c["company"], []).append(i)
    return {name: np.array(ids, dtype=np.int64) for name, ids in out.items()}


def build_stores(chunks: list[dict], index_dir: Path, force_tfidf: bool = False) -> dict:
    index_dir = Path(index_dir)
    texts = [index_text(c) for c in chunks]
    embedder, matrix = Embedder.build(texts, index_dir, force_tfidf=force_tfidf)
    for old in list(index_dir.glob("faiss_*.index")) + [index_dir / "embeddings.npy"]:
        old.unlink(missing_ok=True)  # never mix files from an earlier build
    store = "tfidf"
    if embedder.backend == "dense":
        if faiss is not None:
            for name, ids in _by_company(chunks).items():
                index = faiss.IndexIDMap2(faiss.IndexFlatIP(matrix.shape[1]))
                index.add_with_ids(np.ascontiguousarray(matrix[ids], dtype=np.float32), ids)
                faiss.write_index(index, str(index_dir / f"faiss_{_safe(name)}.index"))
            store = "faiss"
        else:
            print("[store] faiss not installed (pip install faiss-cpu); saving a numpy matrix instead")
            np.save(index_dir / "embeddings.npy", matrix)
            store = "numpy"
    bm25 = BM25Okapi([stem_tokens(t) or ["_"] for t in texts])
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    info = {"backend": embedder.backend, "model": embedder.model_name, "vector_store": store}
    (index_dir / "store.json").write_text(json.dumps(info))
    return info


def load_bm25(index_dir: Path, chunks: list[dict]) -> BM25Okapi:
    path = Path(index_dir) / "bm25.pkl"
    if path.exists():
        with open(path, "rb") as f:
            bm25 = pickle.load(f)
        if len(bm25.doc_len) == len(chunks):
            return bm25
    return BM25Okapi([stem_tokens(index_text(c)) or ["_"] for c in chunks])  # old index: rebuild in memory


class VectorStore:
    def __init__(self, index_dir: Path, chunks: list[dict]):
        index_dir = Path(index_dir)
        self.embedder = Embedder.load(index_dir)
        info_path = index_dir / "store.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        self.kind = info.get("vector_store") or ("numpy" if self.embedder.backend == "dense" else "tfidf")
        self.by_company = _by_company(chunks)
        self.indexes, self.matrix = {}, None
        if self.kind == "faiss":
            if faiss is None:
                raise SystemExit("The index was built with FAISS, but faiss isn't installed: pip install faiss-cpu")
            for name in self.by_company:
                path = index_dir / f"faiss_{_safe(name)}.index"
                if path.exists():
                    self.indexes[name] = faiss.read_index(str(path))
        elif self.kind == "numpy":
            self.matrix = np.load(index_dir / "embeddings.npy")
        else:
            from scipy import sparse

            self.matrix = sparse.load_npz(index_dir / "tfidf_matrix.npz").tocsr()

    def encode(self, query: str):
        return self.embedder.encode_query(query)

    def search(self, q, companies: list[str], k: int) -> list[tuple[int, float]]:
        """Top-k (chunk id, cosine) over the given companies' chunks, best first."""
        found: dict[int, float] = {}
        for name in companies:
            ids = self.by_company.get(name)
            if ids is None or len(ids) == 0:
                continue
            if self.kind == "faiss":
                index = self.indexes.get(name)
                if index is None:
                    continue
                scores, hits = index.search(np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32),
                                            min(k, index.ntotal))
                found.update({int(i): float(s) for i, s in zip(hits[0], scores[0]) if i != -1})
            else:
                sub = self.matrix[ids]
                scores = (sub @ q) if self.kind == "numpy" else np.asarray((sub @ q.T).todense()).ravel()
                for j in np.argsort(-scores, kind="stable")[:k]:
                    found[int(ids[j])] = float(scores[j])
        return sorted(found.items(), key=lambda kv: -kv[1])[:k]