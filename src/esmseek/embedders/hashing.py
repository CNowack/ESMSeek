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
from typing import List, Sequence

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

    def _residue_kmer(self, seq: str, i: int) -> str:
        """The k-mer window centred on residue ``i`` (clamped at the ends)."""
        half = self.k // 2
        start = max(0, min(i - half, len(seq) - self.k)) if len(seq) >= self.k else 0
        return seq[start : start + self.k]

    def _residues_one(self, seq: str) -> np.ndarray:
        seq = seq.upper()
        mat = np.zeros((len(seq), self._dim), dtype=np.float32)
        for i in range(len(seq)):
            kmer = self._residue_kmer(seq, i)
            digest = hashlib.blake2b(kmer.encode(), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            mat[i, h % self._dim] = sign
        return mat

    def embed_residues(self, sequences: Sequence[str]) -> List[np.ndarray]:
        """Per-residue features: each residue is the signed hash of its local
        k-mer window. Not a structural signal — but identical local context maps
        to identical residue vectors, so the per-residue aligner finds shared
        substrings, which is enough to exercise it without an ESM-C download.
        """
        return [self._residues_one(s) for s in sequences]
