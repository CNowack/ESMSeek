# RUN.md — running the three-engine discrimination test

A step-by-step guide for running the per-residue ESM-C aligner and comparing it
against pooled ESM-C and Foldseek. Written for someone comfortable with biology
but new to the command line.

Everything below is typed into the **Terminal** app (macOS: Cmd+Space → "Terminal").
Lines starting with `#` are explanation — don't type those. The `\` at the end of
a line just means "this command continues on the next line"; you can paste a whole
block at once.

---

## Which machine?

**Use a Mac with Apple Silicon (M1/M2/M3).** ESM-C runs on PyTorch, which can use
the Apple **Metal GPU** via `--device mps`. An **AMD** GPU (e.g. Radeon 6750XT)
does *not* accelerate PyTorch on Windows — there the model silently runs on the
CPU. So an M-series Mac is both faster (usable GPU) and simpler (no driver setup)
for a one-off run over ~700 proteins. CPU also works, just slower; pass
`--device cpu` if Metal ever errors.

---

## Part A — Smoke test (2 minutes, no model)

Confirms the code installs and runs. Uses a fake stand-in for ESM-C (the `hash`
backend), so the scores are biologically meaningless — this only proves plumbing.

```bash
# 1. Go to the project folder (adjust the path if you cloned it elsewhere).
cd /path/to/ESMSeek

# 2. Install the package + its test/dev tools.
pip install -e ".[dev]"

# 3. Install numba (speeds up the alignment math).
pip install numba

# 4. Run the aligner on the tiny built-in example.
python comparisons/run_aligner.py \
  --seeds data/seeds/Dn29.faa \
  --pool data/search/lsr_candidates.faa \
  --out esmc_aln.demo.tsv \
  --backend hash \
  --gap-open 0.5 --gap-extend 0.1 --align-seeds 3 --estimate-anisotropy

# 5. Look at the result (two columns: protein id, score).
cat esmc_aln.demo.tsv
```

If step 5 prints a small table, your setup works. Clean up with
`rm esmc_aln.demo.tsv`.

---

## Part B — The real comparison

Scores ~700 proteins with three engines and asks which best separates true LSRs
from look-alike recombinases.

### What you need

- **A Mac (Apple Silicon recommended)** — see "Which machine?" above.
- **Three input files** (you already produced these with `split_seeds.py` /
  `build_pool.py`):
  - `seeds.faa` — the 15 seed LSRs (the queries)
  - `candidates.faa` — the 694-sequence pool to score
  - `labels.tsv` — the hidden answer key (`candidate_id  class  divergent`)
- The v1 saved scores, already in this repo:
  - `comparisons/results/v1/esmc.score.tsv` (pooled ESM-C)
  - `comparisons/results/v1/fold.score.tsv` (Foldseek)

> Tip: keep `seeds.faa`, `candidates.faa`, `labels.tsv` together in one folder.
> The commands below assume they're in the project root; adjust the paths if not.

### B1 — Install the ESM-C model libraries

```bash
cd /path/to/ESMSeek
pip install -e ".[esmc]"          # pulls in torch + esm
```

The ESM-C *weights* (a few GB) download automatically the first time you run an
ESM-C command, not now.

### B2 — Sanity-check your data files

```bash
grep -c ">" candidates.faa        # expect ~694 (number of sequences)
grep -c ">" seeds.faa             # expect 15
head -3 labels.tsv                # header + two rows of the answer key
```

### B3 — Engine 1: the per-residue aligner (the new method)

```bash
python comparisons/run_aligner.py \
  --seeds seeds.faa \
  --pool candidates.faa \
  --out esmc_aln.score.tsv \
  --backend esmc-local --model esmc_300m \
  --device mps \
  --cache-dir .emb_cache \
  --gap-open 0.5 --gap-extend 0.1 --align-seeds 3 --estimate-anisotropy
```

First run downloads the ESM-C weights (one time), then embeds and aligns every
protein. Minutes on the Metal GPU. Progress prints to the screen; it ends with
`[done] wrote esmc_aln.score.tsv`.

What the knobs mean:

| Flag | Plain-English meaning |
|---|---|
| `--device mps` | Use the Apple Metal GPU. Switch to `cpu` if Metal errors. |
| `--cache-dir .emb_cache` | Save embeddings to disk so re-runs are fast. |
| `--gap-open` / `--gap-extend` | Penalty for inserting a gap in the alignment. Higher = prefers short, tight matches. |
| `--estimate-anisotropy` | **Most important knob.** ESM-C thinks every protein pair is somewhat similar (a baseline bias); this measures that baseline from your pool and subtracts it so true similarity stands out. |
| `--align-seeds 3` | For speed, align each candidate only against its 3 most-similar seeds. Raise to `15` to align against all seeds (slower, more thorough). |

### B4 — Engines 2 & 3: reuse the saved v1 scores

Pooled ESM-C and Foldseek were already computed in v1. You don't recompute them —
the next step points straight at the saved files. (If you *want* to regenerate
pooled ESM-C from scratch, ask and I'll give you the `esmseek search` command;
Foldseek needs a separate ProstT5 + Foldseek toolchain, so reuse it unless you
have a reason not to.)

### B5 — Compare all three side by side

```bash
python comparisons/score_discrimination.py \
  --labels labels.tsv \
  --scores pooled=comparisons/results/v1/esmc.score.tsv \
           esmc_aln=esmc_aln.score.tsv \
           foldseek=comparisons/results/v1/fold.score.tsv \
  --target-recall 0.95
```

This prints the verdict table to your screen.

---

## How to read the table

Each engine is held at the threshold where it catches **95% of true LSRs**
(`--target-recall 0.95`). At that fixed sensitivity, the table reports how many
decoys leak through and how many divergent LSRs are still caught.

| Column | What it means | Better = |
|---|---|---|
| `AUROC` | Overall separation. 0.5 = coin flip, 1.0 = perfect. | higher |
| `decoy_pass` | Fraction of *all* decoys that sneak past the threshold. | lower |
| `resolvase_pass` | Resolvase leakage — the hard case (shared catalytic fold). | lower |
| `tyrosine_pass` / `transposase_pass` | Leakage of the easier, different-fold decoys. | lower |
| `divergent_recall` | Fraction of *divergent* LSRs (the remote homologs you care about) still caught. | higher |

**The question:** does `esmc_aln` beat `pooled` — and does it close the gap to (or
beat) `foldseek`, especially on `resolvase_pass` and `divergent_recall`? v1 found
pooled ESM-C only tied Foldseek and recommended *against* building this aligner;
this run tests whether the per-residue version changes that.

**Statistical caveat:** with only ~55 LSR positives (~16 divergent), an AUROC gap
of ~0.01–0.02 is within sampling noise — treat it as a tie. Trust a result when
the gap is large and consistent across several columns.

---

## If results are unconvincing — knobs to sweep

Re-run B3 with one change at a time (the `--cache-dir` makes embedding re-use
free, so only the alignment recomputes):

- **Anisotropy off:** drop `--estimate-anisotropy` (or set `--anisotropy 0`).
  If discrimination collapses, the offset is doing real work.
- **All seeds:** `--align-seeds 15` — in case the 3-seed prefilter is dropping the
  true best seed for a divergent candidate.
- **Gap penalties:** try `--gap-open 1.0 --gap-extend 0.2` (tighter, more local) or
  `--gap-open 0.3 --gap-extend 0.05` (looser, longer alignments).
- **Bigger model:** `--model esmc_600m` (slower, sometimes sharper).

Each run overwrites `esmc_aln.score.tsv`; rename it (e.g. `esmc_aln.aniso_off.tsv`)
if you want to keep a sweep for comparison, and point `--scores` at the renamed
file in B5.

---

See `comparisons/results/v2/README.md` for the method writeup and the full
reproduce recipe (including rebuilding `candidates.faa` / `labels.tsv` from raw
FASTAs if you ever need to).
