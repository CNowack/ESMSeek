"""Tier 2 (scaffold): decoy calibration and FDR control.

Tier 1 returns raw cosine similarities, whose meaningful cutoff depends on the
embedding model. Tier 2 turns those scores into calibrated significance values:

1. Generate *decoy* proteins that preserve composition but destroy real
   signal (shuffle / reverse).
2. Embed decoys and score them against the same seeds to build a null score
   distribution.
3. Convert each candidate's score into an empirical p-value against the null,
   then control the false discovery rate with Benjamini–Hochberg.

The pure building blocks below (:func:`make_decoys`, :func:`empirical_pvalues`,
:func:`benjamini_hochberg`) are implemented and tested. The orchestrator
:func:`calibrate_result` wires them into a run; it is marked **experimental**
because the empirical tuning (decoy model, score statistic, calibration set
size) is the remaining Tier-2 work and is not enabled by default.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

import numpy as np

from .records import Hit


def make_decoys(
    sequences: Sequence[str],
    method: str = "shuffle",
    n_per: int = 1,
    seed: int = 0,
) -> List[str]:
    """Generate decoy sequences that preserve residue composition.

    ``method``:
      * ``"shuffle"`` — random permutation of residues (composition-matched null).
      * ``"reverse"`` — reverse the sequence (preserves composition and local
        k-mer multiset direction; a milder null).
    """
    rng = random.Random(seed)
    decoys: List[str] = []
    for s in sequences:
        for _ in range(n_per):
            if method == "shuffle":
                chars = list(s)
                rng.shuffle(chars)
                decoys.append("".join(chars))
            elif method == "reverse":
                decoys.append(s[::-1])
            else:
                raise ValueError(f"Unknown decoy method {method!r}")
    return decoys


def empirical_pvalues(scores: np.ndarray, null_scores: np.ndarray) -> np.ndarray:
    """Right-tailed empirical p-values of ``scores`` against ``null_scores``.

    ``p = (1 + #{null >= score}) / (1 + N_null)`` (add-one smoothing so p is
    never zero). Higher score => smaller p.
    """
    scores = np.asarray(scores, dtype=np.float64)
    null = np.sort(np.asarray(null_scores, dtype=np.float64))
    n = null.size
    if n == 0:
        return np.ones_like(scores)
    # number of null values >= each score, via the sorted null array
    ge = n - np.searchsorted(null, scores, side="left")
    return (1.0 + ge) / (1.0 + n)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values (FDR) for an array of p-values."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(1, m + 1))
    # enforce monotonicity from the largest p-value down
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def calibrate_result(
    hits: Sequence[Hit],
    seed_seqs: Sequence[str],
    candidate_seqs: Sequence[str],
    embedder,
    method: str = "shuffle",
    n_per: int = 1,
    seed: int = 0,
) -> List[Hit]:
    """Experimental: annotate hits with empirical p-values and BH q-values.

    Builds a null by scoring decoys of the candidate pool against the seeds,
    then calibrates each hit's cosine score. Returns the same hits with
    ``pvalue``/``qvalue`` populated. **Experimental** — see module docstring.
    """
    from .search import KnnIndex  # local import to avoid a cycle

    if not hits:
        return list(hits)

    decoys = make_decoys(candidate_seqs, method=method, n_per=n_per, seed=seed)
    decoy_vecs = embedder.embed(decoys)
    seed_vecs = embedder.embed(list(seed_seqs))

    # Null = best cosine of each decoy to any seed (mirrors the Tier-1 statistic).
    index = KnnIndex(decoy_vecs, use_faiss="never")
    null_scores, _ = index.search(seed_vecs, k=len(decoys))
    null = null_scores.max(axis=0) if null_scores.size else np.zeros(0)

    obs = np.array([h.score for h in hits], dtype=np.float64)
    pvals = empirical_pvalues(obs, null)
    qvals = benjamini_hochberg(pvals)

    out: List[Hit] = []
    for h, p, q in zip(hits, pvals, qvals):
        h.pvalue = float(p)
        h.qvalue = float(q)
        out.append(h)
    return out
