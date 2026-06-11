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

## Status of the numbers

The aligner, the driver, and the three-way harness are implemented and tested.
The **scored comparison table is not filled in here**: reproducing it needs the
v1 694-candidate pool FASTA + `labels.tsv` (regenerated via `build_pool.py`, not
committed to this repo) and a working ESM-C backend (torch + esm weights), which
the development sandbox lacks. Run the commands above on a box with ESM-C to
populate `esmc_aln.score.tsv` and append the resulting table (AUROC, decoy pass
per family, divergent recall) alongside v1's pooled and Foldseek columns.

## Tuning knobs that matter

* `--gap-open` / `--gap-extend` — set the locality of the alignment. Higher
  gap-open favours short, contiguous high-similarity stretches.
* `--anisotropy` / `--estimate-anisotropy` — the single most consequential knob:
  too small and every pair scores high (no discrimination); too large and even
  true homologs score near zero. Estimate it from the pool first, then sweep.
* `--align-seeds` — raise toward 15 (all seeds) if the prefilter is dropping the
  true best seed for divergent candidates; lower for speed.
