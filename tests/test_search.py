from __future__ import annotations

import numpy as np
import pytest

from esmseek.search import KnnIndex, l2_normalize


def test_l2_normalize_unit_rows():
    m = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    n = l2_normalize(m)
    assert np.allclose(np.linalg.norm(n[0]), 1.0)
    # Zero vector stays finite (eps guard), not NaN.
    assert np.all(np.isfinite(n[1]))


@pytest.mark.parametrize("backend", ["numpy", "faiss"])
def test_knn_topk_and_scores(backend):
    rng = np.random.default_rng(0)
    cands = rng.standard_normal((20, 8)).astype(np.float32)
    index = KnnIndex(cands, use_faiss="always" if backend == "faiss" else "never")
    assert index.backend == backend

    # Query with an exact copy of candidate 5 -> it should rank first at ~1.0.
    q = cands[5:6]
    scores, idx = index.search(q, k=3)
    assert idx.shape == (1, 3)
    assert idx[0, 0] == 5
    assert scores[0, 0] == pytest.approx(1.0, abs=1e-5)
    # Scores are sorted descending.
    assert scores[0, 0] >= scores[0, 1] >= scores[0, 2]


def test_knn_faiss_matches_numpy():
    rng = np.random.default_rng(1)
    cands = rng.standard_normal((50, 16)).astype(np.float32)
    seeds = rng.standard_normal((4, 16)).astype(np.float32)
    s_np, i_np = KnnIndex(cands, use_faiss="never").search(seeds, k=5)
    s_fa, i_fa = KnnIndex(cands, use_faiss="always").search(seeds, k=5)
    assert np.array_equal(i_np, i_fa)
    assert np.allclose(s_np, s_fa, atol=1e-5)


def test_knn_k_exceeds_n_is_clamped():
    cands = np.eye(3, dtype=np.float32)
    scores, idx = KnnIndex(cands, use_faiss="never").search(cands[:1], k=10)
    assert idx.shape == (1, 3)


def test_knn_empty_index():
    index = KnnIndex(np.zeros((0, 4), dtype=np.float32), use_faiss="never")
    scores, idx = index.search(np.ones((2, 4), dtype=np.float32), k=5)
    assert scores.shape == (2, 0)
    assert idx.shape == (2, 0)
