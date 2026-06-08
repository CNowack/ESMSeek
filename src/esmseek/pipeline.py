"""End-to-end Tier-1 search: FASTA in -> ranked hits.

Stages: resolve seeds -> collect candidate proteins from the query FASTA
(translate DNA records into ORFs; use amino-acid records as-is) -> embed both
-> cosine k-NN (seeds query the candidate index) -> threshold/rank -> hits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embedders import get_embedder
from .records import Candidate, Hit
from .search import KnnIndex
from .seqio import parse_fasta
from .translate import candidates_from_record, seed_to_protein


@dataclass
class SearchConfig:
    # Embedder
    backend: str = "esmc-local"
    model: Optional[str] = None
    device: str = "auto"
    forge_token: Optional[str] = None
    forge_url: str = "https://forge.evolutionaryscale.ai"
    hash_dim: int = 1024
    hash_k: int = 3
    cache_dir: Optional[str] = None
    # Input interpretation
    seq_type: str = "auto"       # subject (DNA) records: auto|dna|protein
    seed_type: str = "auto"      # seed records: auto|dna|protein
    min_aa: int = 100
    require_start: bool = False
    # Search / ranking
    top_k: int = 50              # candidates reported per seed (<=0 => all)
    min_score: float = 0.0       # cosine cutoff (model-dependent; see Tier 2)
    all_pairs: bool = False      # one row per (candidate, seed) vs best-per-candidate
    use_faiss: str = "auto"      # auto|always|never
    # Tier 2 (experimental, off by default)
    calibrate_method: Optional[str] = None  # "shuffle" | "reverse"
    calibrate_n: int = 1                     # decoys generated per candidate


@dataclass
class SearchResult:
    hits: List[Hit]
    n_candidates: int
    n_seeds: int
    meta: dict = field(default_factory=dict)


def _resolve_seeds(seeds_path: str, seed_type: str) -> List[Tuple[str, str]]:
    seeds: List[Tuple[str, str]] = []
    for rec in parse_fasta(seeds_path):
        prot = seed_to_protein(rec, seq_type=seed_type)
        if prot:
            seeds.append((rec.id, prot))
    if not seeds:
        raise ValueError("No usable seed proteins were resolved")
    return seeds


def _collect_candidates(query_path: str, cfg: SearchConfig) -> List[Candidate]:
    candidates: List[Candidate] = []
    for rec in parse_fasta(query_path):
        candidates.extend(
            candidates_from_record(
                rec,
                seq_type=cfg.seq_type,
                min_aa=cfg.min_aa,
                require_start=cfg.require_start,
            )
        )
    return candidates


def run_search(query_path: str, seeds_path: str, cfg: SearchConfig) -> SearchResult:
    """Search the records in ``query_path`` (DNA or amino acid) against seeds."""
    seeds = _resolve_seeds(seeds_path, cfg.seed_type)
    candidates = _collect_candidates(query_path, cfg)

    meta = {
        "backend": cfg.backend,
        "model": cfg.model,
        "min_aa": cfg.min_aa,
        "require_start": cfg.require_start,
    }
    if not candidates:
        return SearchResult(hits=[], n_candidates=0, n_seeds=len(seeds), meta=meta)

    embedder = get_embedder(
        backend=cfg.backend,
        model=cfg.model,
        device=cfg.device,
        forge_token=cfg.forge_token,
        forge_url=cfg.forge_url,
        hash_dim=cfg.hash_dim,
        hash_k=cfg.hash_k,
        cache_dir=cfg.cache_dir,
    )
    meta["embedder"] = embedder.name

    seed_vecs = embedder.embed([s for _, s in seeds])
    cand_vecs = embedder.embed([c.aa_seq for c in candidates])

    index = KnnIndex(cand_vecs, use_faiss=cfg.use_faiss)
    meta["search_backend"] = index.backend

    k = cfg.top_k if cfg.top_k and cfg.top_k > 0 else len(candidates)
    scores, idx = index.search(seed_vecs, k=k)

    hits = _build_hits(seeds, candidates, scores, idx, cfg)

    if cfg.calibrate_method:
        from .calibrate import calibrate_result

        hits = calibrate_result(
            hits,
            seed_seqs=[s for _, s in seeds],
            candidate_seqs=[c.aa_seq for c in candidates],
            embedder=embedder,
            method=cfg.calibrate_method,
            n_per=cfg.calibrate_n,
        )
        meta["calibrated"] = True
        meta["calibrate_method"] = cfg.calibrate_method

    return SearchResult(
        hits=hits, n_candidates=len(candidates), n_seeds=len(seeds), meta=meta
    )


def _build_hits(
    seeds: List[Tuple[str, str]],
    candidates: List[Candidate],
    scores: np.ndarray,
    idx: np.ndarray,
    cfg: SearchConfig,
) -> List[Hit]:
    all_pairs: List[Hit] = []
    for si, (seed_id, _) in enumerate(seeds):
        for rank, (cand_i, score) in enumerate(zip(idx[si], scores[si]), start=1):
            score = float(score)
            if score < cfg.min_score:
                continue
            all_pairs.append(
                Hit(
                    candidate=candidates[int(cand_i)],
                    seed_id=seed_id,
                    score=score,
                    seed_rank=rank,
                )
            )

    if cfg.all_pairs:
        hits = all_pairs
    else:
        # Keep the single best seed match per candidate.
        best: Dict[str, Hit] = {}
        for hit in all_pairs:
            cur = best.get(hit.candidate.id)
            if cur is None or hit.score > cur.score:
                best[hit.candidate.id] = hit
        hits = list(best.values())

    # Deterministic order: score desc, then stable tie-break on ids so output
    # is reproducible regardless of the (FAISS vs numpy) search backend.
    hits.sort(key=lambda h: (-h.score, h.candidate.id, h.seed_id))
    return hits
