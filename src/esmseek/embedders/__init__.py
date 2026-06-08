"""Embedder backends and the factory used by the CLI/pipeline.

Backends
--------
* ``hash``       — deterministic k-mer feature hashing (no model, for CI/fallback)
* ``esmc-local`` — ESM-C open weights run locally (default for real searches)
* ``esmc-forge`` — ESM-C via the hosted Forge API
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .base import Embedder, stack
from .hashing import HashingEmbedder

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "CachingEmbedder",
    "get_embedder",
    "BACKENDS",
]

BACKENDS = ("hash", "esmc-local", "esmc-forge")


class CachingEmbedder(Embedder):
    """Wrap an embedder with an on-disk per-sequence cache.

    Embeddings are expensive; a discovery pipeline re-runs often. Vectors are
    keyed by ``sha1(embedder_name + sequence)`` so the cache is safe to share
    across runs and never mixes vectors from different models.
    """

    def __init__(self, inner: Embedder, cache_dir: os.PathLike | str):
        self.inner = inner
        self.name = inner.name
        self.cache_dir = Path(cache_dir) / inner.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def dim(self) -> int:
        return self.inner.dim

    def _key_path(self, seq: str) -> Path:
        digest = hashlib.sha1(f"{self.inner.name}\x00{seq}".encode()).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def embed(self, sequences: Sequence[str]) -> np.ndarray:
        results: List[Optional[np.ndarray]] = [None] * len(sequences)
        missing_idx: List[int] = []
        for i, seq in enumerate(sequences):
            path = self._key_path(seq)
            if path.exists():
                results[i] = np.load(path)
            else:
                missing_idx.append(i)

        if missing_idx:
            fresh = self.inner.embed([sequences[i] for i in missing_idx])
            for j, i in enumerate(missing_idx):
                vec = np.asarray(fresh[j], dtype=np.float32)
                np.save(self._key_path(sequences[i]), vec)
                results[i] = vec

        dim = self.inner.dim if not missing_idx else int(results[missing_idx[0]].shape[0])
        return stack([r for r in results if r is not None], dim)


def get_embedder(
    backend: str,
    model: Optional[str] = None,
    device: str = "auto",
    forge_token: Optional[str] = None,
    forge_url: str = "https://forge.evolutionaryscale.ai",
    hash_dim: int = 1024,
    hash_k: int = 3,
    cache_dir: Optional[str] = None,
) -> Embedder:
    """Construct an embedder for ``backend``, optionally wrapped in a disk cache."""
    if backend == "hash":
        emb: Embedder = HashingEmbedder(dim=hash_dim, k=hash_k)
    elif backend == "esmc-local":
        from .esmc import ESMCLocalEmbedder

        emb = ESMCLocalEmbedder(model_name=model or "esmc_300m", device=device)
    elif backend == "esmc-forge":
        from .esmc import ESMCForgeEmbedder

        token = forge_token or os.environ.get("ESM_FORGE_TOKEN")
        emb = ESMCForgeEmbedder(
            model_name=model or "esmc-600m-2024-12", token=token, url=forge_url
        )
    else:
        raise ValueError(f"Unknown backend {backend!r}; choose from {BACKENDS}")

    if cache_dir:
        emb = CachingEmbedder(emb, cache_dir)
    return emb
