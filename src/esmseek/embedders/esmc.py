"""ESM-C embedders: local open weights, or the hosted Forge API.

Both paths use the ``esm`` SDK and the same call sequence — ``encode`` a
sequence, then request embeddings via ``logits(..., LogitsConfig(
return_embeddings=True))`` — and mean-pool the per-residue embeddings,
excluding the leading BOS and trailing EOS tokens. ``embed_residues`` shares the
same model pass but skips the pool, returning the full ``(L, dim)`` residue
matrix for the per-residue aligner (:mod:`esmseek.align`).

Heavy dependencies (``torch``, ``esm``) are imported lazily so the rest of the
package imports cleanly without them installed.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .base import Embedder, stack

# Embedding width of the public local checkpoints, used to report `.dim`
# without a forward pass. Unknown models are probed on load.
_KNOWN_DIMS = {"esmc_300m": 960, "esmc_600m": 1152}


class _ESMCBase(Embedder):
    def __init__(self, model_name: str):
        self.name = model_name
        self._client = None
        self._dim: Optional[int] = _KNOWN_DIMS.get(model_name)

    def _ensure_client(self):  # pragma: no cover - requires esm/torch + weights
        raise NotImplementedError

    @property
    def dim(self) -> int:
        if self._dim is None:  # pragma: no cover - only for unknown checkpoints
            self.embed(["M"])
        assert self._dim is not None
        return self._dim

    def _residues(self, embeddings) -> np.ndarray:  # pragma: no cover - needs torch
        # embeddings: torch.Tensor of shape [1, L+2, dim]; drop the leading BOS
        # and trailing EOS, returning the full (L, dim) residue matrix.
        import torch

        with torch.no_grad():
            residues = embeddings[0, 1:-1, :]
        mat = residues.to(torch.float32).cpu().numpy()
        if self._dim is None:
            self._dim = int(mat.shape[1])
        return mat

    def _pool(self, embeddings) -> np.ndarray:  # pragma: no cover - needs torch
        # Mean-pool the per-residue matrix (BOS/EOS already excluded).
        return self._residues(embeddings).mean(axis=0)

    def _forward(self, sequence: str):  # pragma: no cover - needs weights
        from esm.sdk.api import ESMProtein, LogitsConfig

        client = self._ensure_client()
        cfg = LogitsConfig(sequence=True, return_embeddings=True)
        tensor = client.encode(ESMProtein(sequence=sequence))
        return client.logits(tensor, cfg).embeddings

    def embed(self, sequences: Sequence[str]) -> np.ndarray:  # pragma: no cover - needs weights
        vecs: List[np.ndarray] = [self._pool(self._forward(seq)) for seq in sequences]
        return stack(vecs, self._dim or (vecs[0].shape[0] if vecs else 0))

    def embed_residues(self, sequences: Sequence[str]) -> List[np.ndarray]:  # pragma: no cover - needs weights
        return [self._residues(self._forward(seq)) for seq in sequences]


class ESMCLocalEmbedder(_ESMCBase):
    """Run ESM-C locally from open weights (e.g. ``esmc_300m``, ``esmc_600m``)."""

    def __init__(self, model_name: str = "esmc_300m", device: str = "auto"):
        super().__init__(model_name)
        self.device = device

    def _resolve_device(self) -> str:  # pragma: no cover - environment dependent
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _ensure_client(self):  # pragma: no cover - requires esm/torch + weights
        if self._client is None:
            from esm.models.esmc import ESMC

            device = self._resolve_device()
            self._client = ESMC.from_pretrained(self.name).to(device).eval()
        return self._client


class ESMCForgeEmbedder(_ESMCBase):
    """Run ESM-C through the hosted Forge inference API."""

    def __init__(
        self,
        model_name: str = "esmc-600m-2024-12",
        token: Optional[str] = None,
        url: str = "https://forge.evolutionaryscale.ai",
    ):
        super().__init__(model_name)
        self._dim = None  # Forge model dims are not assumed; probed on first call.
        self.token = token
        self.url = url

    def _ensure_client(self):  # pragma: no cover - requires network + token
        if self._client is None:
            if not self.token:
                raise ValueError(
                    "A Forge API token is required (pass --forge-token or set ESM_FORGE_TOKEN)."
                )
            from esm.sdk.forge import ESM3ForgeInferenceClient

            self._client = ESM3ForgeInferenceClient(
                model=self.name, url=self.url, token=self.token
            )
        return self._client
