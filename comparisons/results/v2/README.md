# Per-residue ESM-C aligner — methods & reproduction (v2)

v1 compared **pooled ESM-C** (one mean vector per protein, cosine to best seed)
against **Foldseek/ProstT5** and deferred the custom per-residue aligner, on the
argument that pooled ESM-C only *matched* the off-the-shelf tool. v2 actually
builds that aligner — a pLM-BLAST-style local alignment over the per-residue
embeddings — and adds it as a third engine so the deferral can be tested rather
than assumed. The hypothesis: a *local* score recovers fold-level similarity that
mean-pooling averages away, especially for the divergent LSRs and against the
shared-fold resolvases that both v1 engines leaked.

## What the aligner does (`esmseek.align`)

For each seed/candidate pair:

1. **Per-residue embeddings.** `Embedder.embed_residues` returns the full
   `(L, dim)` ESM-C residue matrix — the same model pass as the pooled path
   (`embedders/esmc.py`), with the BOS/EOS boundary tokens dropped and the
   mean-pool step skipped. So the rows are exactly what pooling would average.
2. **Unit-normalise** every residue vector, so a dot product *is* the cosine
   (`align.unit_normalize`).
3. **Similarity grid** in one matmul: `seed_norm @ cand_norm.T`, shape
   `(L_seed, L_cand)` (`align.similarity_grid`).
4. **Anisotropy offset.** ESM-C embeddings are not zero-centred, so even
   unrelated residues share a high baseline cosine. Subtract an offset so gaps
   and mismatches can score negative — a fixed `--anisotropy`, or
   `--estimate-anisotropy` to measure the background from random residue pairs in
   the pool (`align.estimate_anisotropy`).
5. **Smith–Waterman**, affine gaps (Gotoh), best local score
   (`align.smith_waterman`). The inner loop is numba-JIT'd when available
   (`align.HAS_NUMBA`); a correct NumPy fallback returns identical scores
   (covered by `tests/test_align.py`).

## Keeping it tractable

Full SW over every pair is ~700 candidates x 15 seeds x ~500x500 cells. Both
standard fixes from the plan are wired into `comparisons/run_aligner.py`:

* **Prefilter (two-stage).** Per candidate, rank seeds by the cheap pooled
  cosine and align only the top `--align-seeds` (default 3). Every candidate
  still gets an aligned score (its best over those seeds), so nothing drops out
  of the discrimination test.
* **numba JIT.** Orders of magnitude faster than plain Python for the grid fill.

`--max-len` centre-trims very long residue matrices if memory is tight.

## Reproduce

Rebuild the same labelled pool as v1 (`build_pool.py` / `split_seeds.py`), then
score it with all three engines and compare. Steps 1–2 are unchanged from v1.

```bash
# 1. Per-residue aligner score (this is the new engine).
#    GPU strongly recommended for the ESM-C residue pass over ~700 proteins.
python comparisons/run_aligner.py \
  --seeds split/seeds.faa \
  --pool  candidates.faa \
  --out   comparisons/results/v2/esmc_aln.score.tsv \
  --backend esmc-local --model esmc_300m \
  --cache-dir .emb_cache \
  --gap-open 0.5 --gap-extend 0.1 --align-seeds 3 --estimate-anisotropy

# 2. Pooled ESM-C score (same as v1; reuse v1's file or regenerate):
esmseek search --query candidates.faa --seeds split/seeds.faa \
  --backend esmc-local --model esmc_300m --top-k 0 --all-pairs \
  --no-seq -o /dev/stdout | ...   # reduce to best-cosine-per-candidate -> esmc.score.tsv
#   (v1's comparisons/results/v1/esmc.score.tsv already holds this.)

# 3. Three-way discrimination:
python comparisons/score_discrimination.py \
  --labels labels.tsv \
  --scores pooled=comparisons/results/v1/esmc.score.tsv \
           esmc_aln=comparisons/results/v2/esmc_aln.score.tsv \
           foldseek=comparisons/results/v1/fold.score.tsv \
  --target-recall 0.95
```

`esmc_aln.score.tsv` is two columns, `candidate_id<TAB>score` (higher = more
LSR-like) — the same contract `score_discrimination.py` reads for every engine.

## Results

Run on the v1 694-candidate pool (55 LSR positives, 16 divergent; 639 decoys),
seeds = 15, `esmc_300m`, default aligner knobs (`--gap-open 0.5 --gap-extend 0.1
--align-seeds 3 --estimate-anisotropy`). Threshold set to **95% LSR recall**;
lower decoy pass = better, higher divergent recall = better.

| Engine | AUROC | LSR recall | decoy pass | resolvase | transposase | tyrosine | divergent recall |
|---|---|---|---|---|---|---|---|
| pooled | 0.904 | 0.96 | 0.31 | 0.44 | 0.07 | 0.13 | 0.88 |
| **esmc_aln** | **0.927** | 0.96 | **0.22** | **0.36** | 0.00 | 0.00 | 0.88 |
| foldseek | 0.928 | 0.96 | 0.26 | 0.41 | 0.00 | 0.00 | 0.88 |

**Point estimates flatter the aligner — the bootstrap says it's a tie.** A
3000-resample stratified bootstrap (resampling within the LSR set and each decoy
family; `score_discrimination.py --bootstrap 3000`) puts error bars on every
number. None of the pairwise differences are significant — every paired CI
crosses zero:

| Δ vs esmc_aln | Δdecoy_pass | Δresolvase_pass | Δdivergent_recall | ΔAUROC |
|---|---|---|---|---|
| pooled   | +0.06 [−0.34, +0.17] | +0.05 [−0.35, +0.16] | +0.00 [−0.12, +0.12] | −0.02 [−0.05, 0.00] |
| foldseek | +0.01 [−0.08, +0.06] | +0.03 [−0.09, +0.09] | −0.00 [−0.12, +0.12] | +0.00 [−0.01, +0.01] |

So the v2 reading is: **the per-residue aligner is statistically
indistinguishable from both pooled ESM-C and Foldseek on this 55-positive set.**
The apparent "best decoy-rejecter" edge in the point table is within sampling
noise. v1's "call it a tie and pick the simpler tool" conclusion holds.

Two things the bootstrap makes clear:

* **AUROC is the only stable metric here.** Its CIs are tight (aligner 0.89–0.96,
  Foldseek 0.89–0.96) because it's a global rank statistic. `decoy_pass`-at-95%-
  recall is wildly unstable (aligner CI 0.19–0.81) because the threshold is pinned
  by ~3 LSRs; rank engines by AUROC, not by decoy pass at a fixed recall.
* **One LSR, `Efs2`, drives the instability.** It is the lowest-scoring true LSR
  for *every* engine (pooled 0.918 = its rank-55; Foldseek 98; aligner 14.9 vs its
  other LSRs at 140–344). Sitting right at the 95%-recall threshold, it whips the
  threshold around under resampling. Worth a look biologically — it may be the
  most structurally divergent positive, or mis-annotated.

### Knob sweep (aligner only)

Re-running the aligner with one knob changed at a time (embeddings reused from
`--cache-dir`, so only the alignment recomputes):

| variant | AUROC | decoy_pass | resolvase | decoy_pass 95% CI |
|---|---|---|---|---|
| default (top-3 seeds, aniso on) | 0.927 | 0.22 | 0.36 | 0.19–0.81 |
| `--align-seeds 15` (all seeds)  | 0.929 | 0.23 | 0.37 | 0.17–0.82 |
| anisotropy **off**              | 0.933 | 0.24 | 0.38 | **0.16–0.51** |

* **`--align-seeds 15` buys nothing** — every paired Δ ≈ 0.00 with tight CIs. The
  top-3 pooled-cosine prefilter drops nothing that matters; keep `3` (faster).
* **The anisotropy correction does not earn its place.** Turning it off gives
  marginally higher AUROC and a far *more stable* decoy_pass CI — the offset is
  what collapses `Efs2` to 14.9 and destabilises the threshold. The pLM-BLAST-style
  background subtraction, expected to be the key knob, is at best neutral here.

**Caveat that still holds:** the win that *would* matter — recovering divergent
LSRs — doesn't happen. Divergent recall is 0.88 (≈14/16) for all three engines,
CIs fully overlapping. And the shared resolvase wall persists:
`resolvase__A0ABQ5PNS2` scores LSR-like under every engine (consistent with v1's
read that part of PF00239 is mis-annotated large serine recombinases; the
length / domain-architecture gate is still the fix).

**Bottom line.** Pooled ESM-C, the per-residue aligner, and Foldseek are a
three-way statistical tie on this benchmark. Foldseek wins on *cost* — one CPU
binary, no torch/esm — so it is ESMSeek's default engine; the ESM-C engines stay
selectable via `--engine` for when a larger positive set or further tuning can
show a real difference.

## Tuning knobs that matter

* `--gap-open` / `--gap-extend` — set the locality of the alignment. Higher
  gap-open favours short, contiguous high-similarity stretches.
* `--anisotropy` / `--estimate-anisotropy` — the single most consequential knob:
  too small and every pair scores high (no discrimination); too large and even
  true homologs score near zero. Estimate it from the pool first, then sweep.
* `--align-seeds` — raise toward 15 (all seeds) if the prefilter is dropping the
  true best seed for divergent candidates; lower for speed.
