"""Embedding backends. ingest.py encodes the chunks once; retriever.py encodes each question.

dense : sentence-transformers (default BAAI/bge-small-en-v1.5, ~130 MB, CPU). L2-normalised; stored in FAISS.
tfidf : scikit-learn TF-IDF, used automatically if the dense model can't be installed or downloaded.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from config import EMBED_MODEL

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, backend: str, model_name: str, model):
        self.backend, self.model_name, self.model = backend, model_name, model

    @classmethod
    def build(cls, texts: list[str], index_dir: Path, force_tfidf: bool = False):
        """Encode all chunk texts. Returns (embedder, matrix); index_store.py decides how to store the matrix."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        if not force_tfidf:
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(EMBED_MODEL)
                matrix = model.encode(texts, batch_size=64, normalize_embeddings=True,
                                      show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
                embedder = cls("dense", EMBED_MODEL, model)
                embedder._save_meta(index_dir)
                return embedder, matrix
            except Exception as exc:  # no package, no internet, no disk...
                print(f"[embed] dense model unavailable ({type(exc).__name__}: {exc}); falling back to TF-IDF")
        from scipy import sparse
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform(texts)  # rows are L2-normalised, so dot product == cosine
        sparse.save_npz(index_dir / "tfidf_matrix.npz", matrix)
        with open(index_dir / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        embedder = cls("tfidf", "tfidf", vectorizer)
        embedder._save_meta(index_dir)
        return embedder, matrix

    def _save_meta(self, index_dir: Path) -> None:
        (Path(index_dir) / "embedder.json").write_text(json.dumps({"backend": self.backend, "model": self.model_name}))

    @classmethod
    def load(cls, index_dir: Path) -> "Embedder":
        meta = json.loads((Path(index_dir) / "embedder.json").read_text())
        if meta["backend"] == "dense":
            from sentence_transformers import SentenceTransformer

            return cls("dense", meta["model"], SentenceTransformer(meta["model"]))
        with open(Path(index_dir) / "tfidf_vectorizer.pkl", "rb") as f:
            return cls("tfidf", "tfidf", pickle.load(f))

    def encode_query(self, query: str):
        """Dense: 1-D float32 unit vector. TF-IDF: 1 x V sparse row."""
        if self.backend == "dense":
            prefix = BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
            q = self.model.encode([prefix + query], normalize_embeddings=True, convert_to_numpy=True)[0]
            return q.astype(np.float32)
        return self.model.transform([query])