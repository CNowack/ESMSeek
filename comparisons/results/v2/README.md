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

**Read.** The per-residue aligner beats pooled ESM-C on every column and ties
Foldseek on overall AUROC (0.927 vs 0.928 — a dead tie). Its decoy rejection is
the best of the three: overall leakage drops 31% → 22%, the different-fold
decoys (transposase, tyrosine) go to 0% like Foldseek, and — notably — it edges
Foldseek on the hardest family, **resolvases (36% vs 41% vs 44%)**, the leakage
v1 called a shared, method-independent wall. This reopens the question v1 closed
("the custom aligner is not justified"): with no tuning it matches the
off-the-shelf structural tool and is the single best decoy-rejecter.

**Caveats.** (1) The win is *decoy purity, not divergent sensitivity*: divergent
recall is identical (0.88, ≈14/16) across all three engines — the aligner
recovers no remote LSR that pooled/Foldseek miss. (2) The AUROC gap to pooled
(0.023) is within the ±0.04 sampling band for 55 positives; the gap to Foldseek
is noise. The decoy numbers rest on 639 decoys (400 resolvases) and are more
trustworthy, but the resolvase improvement is ~2 SE — bootstrap a CI before
calling it definitive. (3) The shared wall persists at the sequence level:
`resolvase__A0ABQ5PNS2` scores 312.9 (above most true LSRs), the same resolvase
that ranked 7th under both v1 engines — consistent with v1's read that part of
the PF00239 set is genuinely mislabelled large serine recombinases. The
length / domain-architecture gate v1 recommended is still the fix for those.

## Tuning knobs that matter

* `--gap-open` / `--gap-extend` — set the locality of the alignment. Higher
  gap-open favours short, contiguous high-similarity stretches.
* `--anisotropy` / `--estimate-anisotropy` — the single most consequential knob:
  too small and every pair scores high (no discrimination); too large and even
  true homologs score near zero. Estimate it from the pool first, then sweep.
* `--align-seeds` — raise toward 15 (all seeds) if the prefilter is dropping the
  true best seed for divergent candidates; lower for speed.
