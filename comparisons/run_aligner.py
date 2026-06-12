#!/usr/bin/env python3
"""
Score a candidate pool with the per-residue ESM-C aligner (pLM-BLAST style).

This is the third engine for the discrimination test (alongside pooled ESM-C and
Foldseek). For every candidate it returns the best *local* Smith–Waterman score
over the residue-by-residue cosine grid against the seeds — the signal pooled
cosine throws away when it averages each protein to one vector.

USAGE
-----
    python run_aligner.py \
        --seeds split/seeds.faa \
        --pool  candidates.faa \
        --out   esmc_aln.score.tsv \
        --backend esmc-local --model esmc_300m \
        --cache-dir .emb_cache \
        --gap-open 0.5 --gap-extend 0.1 --align-seeds 3 --estimate-anisotropy

OUTPUT
------
Two columns (no header), matching what score_discrimination.py reads:
    candidate_id   best_aligner_score

TRACTABILITY
------------
Smith–Waterman over every seed x candidate pair is a lot of grid-filling
(~700 candidates x 15 seeds x ~500x500 cells). Two standard fixes, both wired
here:
  * prefilter  — rank seeds per candidate by the cheap pooled cosine and align
                 only the top ``--align-seeds`` (default 3); and
  * numba JIT  — the inner SW loop is JIT-compiled when numba is importable
                 (``esmseek.align.HAS_NUMBA``), orders of magnitude faster than
                 plain Python. A correct NumPy fallback runs otherwise.
Use ``--max-len`` to cap residue length (centre-trim) if memory is tight.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Allow running from the comparisons/ directory without installing first.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from esmseek import align  # noqa: E402
from esmseek.embedders import get_embedder  # noqa: E402
from esmseek.records import SeqRecord  # noqa: E402
from esmseek.search import l2_normalize  # noqa: E402
from esmseek.seqio import parse_fasta  # noqa: E402
from esmseek.translate import seed_to_protein  # noqa: E402


def _center_trim(seq: str, max_len: int) -> str:
    if max_len <= 0 or len(seq) <= max_len:
        return seq
    start = (len(seq) - max_len) // 2
    return seq[start : start + max_len]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", required=True, help="FASTA of seed proteins (AA or DNA).")
    ap.add_argument("--pool", required=True, help="FASTA of candidate proteins to score.")
    ap.add_argument("--out", required=True, help="Output TSV: candidate_id<TAB>score.")
    # embedder
    ap.add_argument("--backend", default="esmc-local",
                    help="Embedding backend (default: esmc-local; 'hash' for a "
                         "dependency-free smoke test).")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", default=None,
                    help="Cache pooled + per-residue embeddings across runs.")
    ap.add_argument("--hash-dim", type=int, default=1024)
    ap.add_argument("--hash-k", type=int, default=3)
    # aligner knobs
    ap.add_argument("--gap-open", type=float, default=0.5, help="Gap-open penalty (default 0.5).")
    ap.add_argument("--gap-extend", type=float, default=0.1, help="Gap-extend penalty (default 0.1).")
    ap.add_argument("--anisotropy", type=float, default=0.0,
                    help="Constant background-cosine offset subtracted from every grid.")
    ap.add_argument("--estimate-anisotropy", action="store_true",
                    help="Estimate the offset from the pool instead of using --anisotropy.")
    ap.add_argument("--align-seeds", type=int, default=3,
                    help="Per candidate, align only its top-N seeds by pooled cosine "
                         "(<=0 = all seeds). The two-stage prefilter (default 3).")
    ap.add_argument("--max-len", type=int, default=0,
                    help="Centre-trim residues longer than this (0 = no cap).")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    def log(*a):
        if not args.quiet:
            print(*a, file=sys.stderr)

    seeds = [(r.id, _center_trim(seed_to_protein(r), args.max_len))
             for r in parse_fasta(args.seeds)]
    pool = [(r.id, _center_trim(seed_to_protein(SeqRecord(id=r.id, seq=r.seq), "protein"),
                                args.max_len))
            for r in parse_fasta(args.pool)]
    log(f"[aligner] seeds={len(seeds)} candidates={len(pool)} "
        f"numba={'yes' if align.HAS_NUMBA else 'no (numpy fallback)'}")

    embedder = get_embedder(
        backend=args.backend, model=args.model, device=args.device,
        hash_dim=args.hash_dim, hash_k=args.hash_k, cache_dir=args.cache_dir,
    )

    # Pooled vectors drive the prefilter; residue matrices drive the alignment.
    t0 = time.time()
    seed_pooled = l2_normalize(embedder.embed([s for _, s in seeds]))
    cand_pooled = l2_normalize(embedder.embed([s for _, s in pool]))
    seed_res = [align.unit_normalize(m) for m in embedder.embed_residues([s for _, s in seeds])]
    cand_res = embedder.embed_residues([s for _, s in pool])
    log(f"[aligner] embedded in {time.time()-t0:.1f}s")

    anisotropy = args.anisotropy
    if args.estimate_anisotropy:
        anisotropy = align.estimate_anisotropy(seed_res + cand_res)
        log(f"[aligner] estimated anisotropy offset = {anisotropy:.4f}")

    n_seeds = len(seeds)
    top_n = n_seeds if args.align_seeds <= 0 else min(args.align_seeds, n_seeds)

    t0 = time.time()
    results = []
    for ci, (cid, _) in enumerate(pool):
        cmat = align.unit_normalize(cand_res[ci])
        # Prefilter: pooled-cosine rank of seeds for this candidate.
        cos = cand_pooled[ci] @ seed_pooled.T
        order = np.argsort(-cos)[:top_n]
        best = 0.0
        for si in order:
            grid = align.similarity_grid(seed_res[si], cmat, anisotropy=anisotropy,
                                         normalized=True)
            score = align.smith_waterman(grid, args.gap_open, args.gap_extend)
            if score > best:
                best = score
        results.append((cid, best))
        if not args.quiet and (ci + 1) % 100 == 0:
            log(f"[aligner] aligned {ci+1}/{len(pool)}")
    log(f"[aligner] aligned {len(pool)} candidates in {time.time()-t0:.1f}s")

    with open(args.out, "w") as fh:
        for cid, score in results:
            fh.write(f"{cid}\t{score:.6f}\n")
    log(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
