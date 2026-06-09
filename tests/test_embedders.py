from __future__ import annotations

import numpy as np

from esmseek.embedders import CachingEmbedder, HashingEmbedder, get_embedder
from esmseek.search import l2_normalize


def test_hashing_shape_and_determinism():
    emb = HashingEmbedder(dim=256, k=3)
    a = emb.embed(["MSKVLTAQEII", "MSKVLTAQEII"])
    assert a.shape == (2, 256)
    assert a.dtype == np.float32
    # Identical inputs -> identical vectors.
    assert np.array_equal(a[0], a[1])
    # Re-running is reproducible.
    b = emb.embed(["MSKVLTAQEII"])
    assert np.array_equal(a[0], b[0])


def test_hashing_similarity_orders_sensibly():
    emb = HashingEmbedder(dim=512, k=3)
    base = "MSKVLTAQEIIDRLNKGEKLSVKDLAEELG"
    near = base[:-1] + "A"          # one substitution
    far = "WWWWWWPPPPPPCCCCCCYYYYYY"   # disjoint composition
    v = l2_normalize(emb.embed([base, near, far]))
    cos_near = float(v[0] @ v[1])
    cos_far = float(v[0] @ v[2])
    assert cos_near > cos_far
    assert cos_far < 0.2


def test_hashing_short_sequence():
    emb = HashingEmbedder(dim=64, k=3)
    v = emb.embed(["MK"])  # shorter than k
    assert v.shape == (1, 64)
    assert np.any(v != 0)


def test_get_embedder_hash_backend():
    emb = get_embedder("hash", hash_dim=128, hash_k=2)
    assert isinstance(emb, HashingEmbedder)
    assert emb.dim == 128


class _CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__(dim=32, k=3)
        self.calls = []

    def embed(self, sequences):
        self.calls.append(list(sequences))
        return super().embed(sequences)


def test_caching_embedder_reuses_disk(tmp_path):
    inner = _CountingEmbedder()
    cached = CachingEmbedder(inner, tmp_path)

    first = cached.embed(["AAAA", "MKVL"])
    assert inner.calls == [["AAAA", "MKVL"]]  # both computed

    # Second call: one cached ("AAAA"), one new ("PQRS").
    second = cached.embed(["AAAA", "PQRS"])
    assert inner.calls[-1] == ["PQRS"]        # only the miss recomputed
    assert np.array_equal(first[0], second[0])  # cached vector identical
    assert cached.dim == inner.dim


def test_caching_via_get_embedder(tmp_path):
    emb = get_embedder("hash", hash_dim=64, cache_dir=str(tmp_path))
    assert isinstance(emb, CachingEmbedder)
    emb.embed(["MSKV"])
    # A cache file should have been written.
    assert any(tmp_path.rglob("*.npy"))
