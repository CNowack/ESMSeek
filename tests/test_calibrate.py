from __future__ import annotations

import numpy as np

from esmseek.calibrate import benjamini_hochberg, empirical_pvalues, make_decoys


def test_make_decoys_shuffle_preserves_composition():
    seqs = ["MSKVLTAQEII"]
    decoys = make_decoys(seqs, method="shuffle", n_per=3, seed=42)
    assert len(decoys) == 3
    for d in decoys:
        assert sorted(d) == sorted(seqs[0])  # same multiset of residues
    # Deterministic given the seed.
    again = make_decoys(seqs, method="shuffle", n_per=3, seed=42)
    assert decoys == again


def test_make_decoys_reverse():
    assert make_decoys(["ABC"], method="reverse") == ["CBA"]


def test_empirical_pvalues_monotonic_and_bounded():
    null = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    p = empirical_pvalues(np.array([0.6, 0.35, 0.0]), null)
    # Higher score -> smaller p; all within (0, 1].
    assert p[0] < p[1] < p[2]
    assert np.all(p > 0) and np.all(p <= 1)
    # A score above all null values: only the +1 smoothing remains.
    assert p[0] == (1 + 0) / (1 + 5)


def test_empirical_pvalues_empty_null():
    p = empirical_pvalues(np.array([0.9]), np.array([]))
    assert p[0] == 1.0


def test_benjamini_hochberg_known_values():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    q = benjamini_hochberg(p)
    # Monotonic non-decreasing in p order and bounded by 1.
    assert np.all(np.diff(q) >= -1e-12)
    assert np.all(q <= 1.0)
    # Largest p-value's q equals p*m/m = p.
    assert q[-1] == 0.05


def test_benjamini_hochberg_empty():
    assert benjamini_hochberg(np.array([])).size == 0
