"""Embedder interface shared by all backends."""

from __future__ import annotations

import abc
from typing import List, Sequence

import numpy as np


class Embedder(abc.ABC):
    """Maps protein sequences to fixed-length pooled embedding vectors.

    Implementations return a ``float32`` array of shape ``(n, dim)`` where row
    ``i`` is the (mean-pooled) embedding of ``sequences[i]``. Vectors are *not*
    required to be L2-normalised here; the search layer normalises before
    computing cosine similarity.

    Backends may *additionally* expose the full per-residue embedding matrix via
    :meth:`embed_residues` — the un-pooled signal the per-residue aligner
    (:mod:`esmseek.align`) consumes. Not every backend supports it, so it is a
    concrete method that raises by default rather than an abstract requirement.
    """

    #: Stable identifier used in cache keys and TSV provenance, e.g. "esmc_300m".
    name: str = "embedder"

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""

    @abc.abstractmethod
    def embed(self, sequences: Sequence[str]) -> np.ndarray:
        """Embed a batch of sequences into an ``(n, dim)`` float32 array."""

    def embed_one(self, sequence: str) -> np.ndarray:
        return self.embed([sequence])[0]

    def embed_residues(self, sequences: Sequence[str]) -> List[np.ndarray]:
        """Return per-residue embeddings: one ``(L_i, dim)`` float32 matrix per
        input, where ``L_i`` is the residue length of ``sequences[i]`` (no BOS/
        EOS boundary tokens). This is the same model pass as :meth:`embed` with
        the mean-pool step skipped, so the rows are exactly what pooling would
        have averaged.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement per-residue embeddings"
        )

    def embed_residues_one(self, sequence: str) -> np.ndarray:
        return self.embed_residues([sequence])[0]


def _empty(dim: int) -> np.ndarray:
    return np.zeros((0, dim), dtype=np.float32)


def stack(vectors: List[np.ndarray], dim: int) -> np.ndarray:
    if not vectors:
        return _empty(dim)
    return np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
