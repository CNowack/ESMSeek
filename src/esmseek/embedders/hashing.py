"""Deterministic k-mer feature-hashing embedder (no heavy dependencies).

This backend is **not** a structural model — it is a fast, reproducible
stand-in that turns each protein into an L2-comparable bag-of-k-mers vector.
Sequences that share local composition land near each other under cosine
similarity, which is enough to:

* exercise the full pipeline (and CI) with no model download or GPU, and
* provide a sensible fallback when ESM-C is unavailable.

For real discovery use an ESM-C backend; use ``hash`` for smoke tests,
plumbing, and reproducible fixtures.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from .base import Embedder, stack


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 1024, k: int = 3):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        self._dim = dim
        self.k = k
        self.name = f"hash_k{k}_d{dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_one(self, seq: str) -> np.ndarray:
        seq = seq.upper()
        vec = np.zeros(self._dim, dtype=np.float32)
        if len(seq) < self.k:
            # Too short for a k-mer: fall back to single-residue features.
            kmers = list(seq)
        else:
            kmers = [seq[i : i + self.k] for i in range(len(seq) - self.k + 1)]
        for kmer in kmers:
            digest = hashlib.blake2b(kmer.encode(), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            idx = h % self._dim
            sign = 1.0 if (h >> 63) & 1 else -1.0  # signed hashing reduces collision bias
            vec[idx] += sign
        return vec

    def embed(self, sequences: Sequence[str]) -> np.ndarray:
        return stack([self._embed_one(s) for s in sequences], self._dim)
