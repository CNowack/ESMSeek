"""Cosine k-NN search over candidate embeddings.

Candidate vectors are L2-normalised and indexed; querying with (normalised)
seed vectors via inner product therefore yields cosine similarity. FAISS
(``IndexFlatIP``) is used when available for speed at scale; otherwise an exact
numpy brute-force search produces identical results.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, eps)


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except Exception:
        return False


class KnnIndex:
    """A cosine-similarity nearest-neighbour index over candidate vectors."""

    def __init__(self, vectors: np.ndarray, use_faiss: str = "auto"):
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        self.n, self.dim = vectors.shape
        self._normed = l2_normalize(vectors)

        if use_faiss == "never":
            self.backend = "numpy"
        elif use_faiss == "always":
            self.backend = "faiss"
        else:  # auto
            self.backend = "faiss" if _faiss_available() else "numpy"

        self._index = None
        if self.backend == "faiss":
            if not _faiss_available():
                raise RuntimeError("FAISS requested but faiss is not importable")
            import faiss

            self._index = faiss.IndexFlatIP(self.dim)
            if self.n:
                self._index.add(self._normed)

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, indices)`` of the top-``k`` candidates per query.

        Both arrays have shape ``(n_queries, k_eff)`` where ``k_eff = min(k, n)``.
        Rows are sorted by descending cosine similarity. Indices reference rows
        of the candidate matrix the index was built from.
        """
        if self.n == 0:
            return (
                np.zeros((len(queries), 0), dtype=np.float32),
                np.zeros((len(queries), 0), dtype=np.int64),
            )
        k_eff = min(k, self.n)
        q = l2_normalize(queries)

        if self.backend == "faiss":
            scores, idx = self._index.search(q, k_eff)
            return scores.astype(np.float32), idx.astype(np.int64)

        # Exact numpy fallback.
        sims = q @ self._normed.T  # (n_queries, n)
        idx = np.argsort(-sims, axis=1)[:, :k_eff]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores.astype(np.float32), idx.astype(np.int64)
