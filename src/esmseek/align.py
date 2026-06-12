"""Per-residue Smith–Waterman alignment over ESM-C embeddings (pLM-BLAST style).

The pooled cosine score (:mod:`esmseek.search`) collapses each protein to one
vector and so cannot reward a *local* stretch of structural similarity flanked by
divergent regions. This module restores that signal: it aligns the two proteins'
per-residue embedding matrices directly.

Pipeline for a seed/candidate pair:

1. unit-normalise every residue vector, so a dot product *is* the cosine;
2. multiply the two matrices to get the full residue-by-residue similarity grid
   in one matmul;
3. subtract an *anisotropy* offset — ESM-C embeddings are not zero-centred, so
   even unrelated residues share a high baseline cosine; subtracting it
   recentres the grid so gaps and mismatches can actually score negative;
4. run local (Smith–Waterman) alignment with affine gap penalties and return the
   best local score.

The inner Smith–Waterman loop is JIT-compiled with numba when it is importable
(orders of magnitude faster than plain Python); a pure-NumPy/Python fallback with
identical results is used otherwise.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def unit_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise each row of an ``(L, dim)`` residue matrix."""
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, eps)


def similarity_grid(
    seed_mat: np.ndarray,
    cand_mat: np.ndarray,
    anisotropy: float = 0.0,
    normalized: bool = False,
) -> np.ndarray:
    """Residue-by-residue cosine grid ``(L_seed, L_cand)``, minus ``anisotropy``.

    Set ``normalized=True`` to skip re-normalising rows that are already unit
    length (e.g. cached normalised matrices).
    """
    a = seed_mat if normalized else unit_normalize(seed_mat)
    b = cand_mat if normalized else unit_normalize(cand_mat)
    grid = a @ b.T
    if anisotropy:
        grid = grid - anisotropy
    return np.ascontiguousarray(grid, dtype=np.float32)


def estimate_anisotropy(matrices, sample: int = 20000, seed: int = 0) -> float:
    """Estimate the background cosine baseline from a set of residue matrices.

    Concatenates all residues, draws random residue *pairs*, and returns the mean
    cosine — an empirical anisotropy offset to feed back into
    :func:`similarity_grid`. (pLM-BLAST uses a fixed offset; estimating it from
    the candidate pool adapts to the checkpoint.)
    """
    pool = [unit_normalize(m) for m in matrices if len(m)]
    if not pool:
        return 0.0
    allres = np.concatenate(pool, axis=0)
    n = allres.shape[0]
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=sample)
    j = rng.integers(0, n, size=sample)
    keep = i != j
    return float(np.mean(np.sum(allres[i[keep]] * allres[j[keep]], axis=1)))


# --- Smith–Waterman (affine gaps, Gotoh) -----------------------------------
#
# H = best score of a local alignment ending in a match/mismatch at (i, j);
# E = best ending in a gap along the candidate (consume seed residue);
# F = best ending in a gap along the seed.  gap_open/gap_extend are positive
# penalties; gap_open is charged on the first gap residue, gap_extend on each
# subsequent one.  The returned score is the maximum H over the whole grid.


def _smith_waterman_numpy(grid: np.ndarray, gap_open: float, gap_extend: float) -> float:
    """Correct, dependency-free affine-gap Smith–Waterman (row-by-row scan)."""
    n, m = grid.shape
    if n == 0 or m == 0:
        return 0.0
    neg = -1e30
    prevH = np.zeros(m + 1, dtype=np.float64)   # H[i-1, :]
    prevF = np.full(m + 1, neg, dtype=np.float64)  # F[i-1, :]
    best = 0.0
    grid64 = grid.astype(np.float64, copy=False)
    for i in range(1, n + 1):
        curH = np.zeros(m + 1, dtype=np.float64)
        curE = neg
        curF = np.full(m + 1, neg, dtype=np.float64)
        row = grid64[i - 1]
        Hleft = 0.0  # curH[j-1]
        for j in range(1, m + 1):
            curE = max(Hleft - gap_open, curE - gap_extend)
            curF[j] = max(prevH[j] - gap_open, prevF[j] - gap_extend)
            h = prevH[j - 1] + row[j - 1]
            if curE > h:
                h = curE
            if curF[j] > h:
                h = curF[j]
            if h < 0.0:
                h = 0.0
            curH[j] = h
            Hleft = h
            if h > best:
                best = h
        prevH = curH
        prevF = curF
    return float(best)


def _build_numba_sw():
    try:
        import numba
    except Exception:
        return None

    @numba.njit(cache=True, fastmath=True)
    def _sw(grid, gap_open, gap_extend):  # pragma: no cover - compiled
        n, m = grid.shape
        if n == 0 or m == 0:
            return 0.0
        neg = -1e30
        prevH = np.zeros(m + 1)
        prevF = np.full(m + 1, neg)
        curH = np.zeros(m + 1)
        curF = np.full(m + 1, neg)
        best = 0.0
        for i in range(1, n + 1):
            for j in range(m + 1):
                curH[j] = 0.0
                curF[j] = neg
            curE = neg
            Hleft = 0.0
            for j in range(1, m + 1):
                e = Hleft - gap_open
                e2 = curE - gap_extend
                curE = e if e > e2 else e2
                f = prevH[j] - gap_open
                f2 = prevF[j] - gap_extend
                fj = f if f > f2 else f2
                curF[j] = fj
                h = prevH[j - 1] + grid[i - 1, j - 1]
                if curE > h:
                    h = curE
                if fj > h:
                    h = fj
                if h < 0.0:
                    h = 0.0
                curH[j] = h
                Hleft = h
                if h > best:
                    best = h
            for j in range(m + 1):
                prevH[j] = curH[j]
                prevF[j] = curF[j]
        return best

    return _sw


_NUMBA_SW = _build_numba_sw()
HAS_NUMBA = _NUMBA_SW is not None


def smith_waterman(grid: np.ndarray, gap_open: float, gap_extend: float) -> float:
    """Best local alignment score over ``grid`` with affine gap penalties.

    Uses the numba-compiled kernel when available, else a NumPy fallback that
    returns identical scores. ``gap_open``/``gap_extend`` are positive penalties.
    """
    grid = np.ascontiguousarray(grid, dtype=np.float32)
    if _NUMBA_SW is not None:
        return float(_NUMBA_SW(grid.astype(np.float64), float(gap_open), float(gap_extend)))
    return _smith_waterman_numpy(grid, gap_open, gap_extend)


def align_score(
    seed_mat: np.ndarray,
    cand_mat: np.ndarray,
    gap_open: float = 0.5,
    gap_extend: float = 0.1,
    anisotropy: float = 0.0,
    normalized: bool = False,
) -> float:
    """Per-residue local-alignment score between a seed and a candidate.

    ``seed_mat``/``cand_mat`` are ``(L, dim)`` residue matrices. Returns the best
    Smith–Waterman score over the anisotropy-recentred cosine grid.
    """
    grid = similarity_grid(seed_mat, cand_mat, anisotropy=anisotropy, normalized=normalized)
    return smith_waterman(grid, gap_open, gap_extend)
