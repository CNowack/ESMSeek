from __future__ import annotations

import numpy as np
import pytest

from esmseek import align
from esmseek.embedders import HashingEmbedder


def test_unit_normalize_rows_unit_length():
    mat = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    out = align.unit_normalize(mat)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_similarity_grid_is_cosine_and_anisotropy_subtracts():
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0]], dtype=np.float32)
    grid = align.similarity_grid(a, b)
    assert grid.shape == (2, 1)
    assert np.allclose(grid[:, 0], [1.0, 0.0], atol=1e-6)
    shifted = align.similarity_grid(a, b, anisotropy=0.25)
    assert np.allclose(grid - 0.25, shifted, atol=1e-6)


def test_sw_single_positive_cell():
    grid = np.array([[5.0]], dtype=np.float32)
    assert align.smith_waterman(grid, gap_open=1.0, gap_extend=0.5) == pytest.approx(5.0)


def test_sw_diagonal_accumulates():
    # Two matches on the diagonal should sum; no gaps needed.
    grid = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
    assert align.smith_waterman(grid, gap_open=1.0, gap_extend=0.5) == pytest.approx(2.0)


def test_sw_never_negative():
    grid = np.full((4, 5), -2.0, dtype=np.float32)
    assert align.smith_waterman(grid, gap_open=1.0, gap_extend=0.5) == pytest.approx(0.0)


def test_sw_gap_penalty_reduces_score():
    # Two strong matches separated by a forced gap in the candidate.
    grid = np.array(
        [[2.0, -1.0, -1.0],
         [-1.0, -1.0, 2.0]],
        dtype=np.float32,
    )
    cheap = align.smith_waterman(grid, gap_open=0.5, gap_extend=0.1)
    pricey = align.smith_waterman(grid, gap_open=5.0, gap_extend=5.0)
    # With a cheap gap, bridging both matches (4 - gap) beats a lone match (2).
    assert cheap > 2.0
    # With an expensive gap, the best is a single match.
    assert pricey == pytest.approx(2.0)


def test_numpy_fallback_matches_numba_on_random_grids():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n, m = rng.integers(1, 12, size=2)
        grid = (rng.standard_normal((n, m)) * 0.5).astype(np.float32)
        go, ge = 0.4, 0.1
        ref = align._smith_waterman_numpy(grid, go, ge)
        got = align.smith_waterman(grid, go, ge)
        assert got == pytest.approx(ref, abs=1e-4)


def test_self_alignment_beats_unrelated_with_hash_residues():
    emb = HashingEmbedder(dim=512, k=3)
    seq = "MSKVLTAQEIIDRLNKGEKLSVKDLAEELGVSRQTIYNWLNG"
    unrelated = "WPWPWPCYCYCYGHGHGHKLKLKLMNMNMNDEDEDEFGFGFG"
    seed_mat, same_mat, diff_mat = emb.embed_residues([seq, seq, unrelated])
    self_score = align.align_score(seed_mat, same_mat, gap_open=0.5, gap_extend=0.1)
    diff_score = align.align_score(seed_mat, diff_mat, gap_open=0.5, gap_extend=0.1)
    assert self_score > diff_score


def test_estimate_anisotropy_in_range():
    emb = HashingEmbedder(dim=128, k=3)
    mats = emb.embed_residues(["MSKVLTAQEII", "GHKLMNDEFPQ", "AAAACCCCGGGG"])
    a = align.estimate_anisotropy(mats, sample=2000)
    assert -1.0 <= a <= 1.0
