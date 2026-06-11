# ESMSeek

**Structural homology search over raw DNA using protein language-model embeddings.**

ESMSeek is a stand-alone CLI that takes a query FASTA — **raw DNA or amino-acid
sequences** — plus a set of seed proteins, and returns the query proteins that
look — *structurally/functionally*, in embedding space — most like the seeds. It
is built to slot into an LSR (large serine recombinase) discovery pipeline:
**FASTA in, TSV out**, no server.

The query type is auto-detected per record: DNA records are translated into ORFs,
amino-acid records are searched as-is, and a mixed FASTA is fine.

The idea is embedding-based homology. Instead of scoring sequence identity
(BLAST/DIAMOND), ESMSeek embeds each protein with [ESM-C](https://github.com/evolutionaryscale/esm),
mean-pools to a single vector, and ranks candidates by **cosine similarity** to
the seeds via **FAISS k-NN**. This recovers remote homologs that share fold and
function at sequence identities where alignment search goes quiet.

```
query DNA   ──▶  6-frame ORFs  ──┐
query AA    ──▶  (used as-is)  ──┼▶  ESM-C embeddings ─┐
                                                       ├─▶  cosine / FAISS k-NN  ──▶  ranked TSV
seeds (AA or DNA) ──────────────────▶  ESM-C embeddings ┘
```

## Two tiers

| Tier | Status | What it adds |
|------|--------|--------------|
| **Tier 1** | ✅ implemented | Pooled ESM-C embeddings + cosine/FAISS k-NN. FASTA in, TSV out. Raw cosine scores rank the hits. |
| **Tier 2** | 🧪 scaffolded | Decoy calibration + FDR. Turns raw cosine into empirical p-/q-values so cutoffs are principled rather than eyeballed. Building blocks (`esmseek.calibrate`) are implemented and tested; the orchestrator is wired behind `--calibrate` and marked experimental pending empirical tuning. |

## Install

```bash
# Core (deterministic `hash` backend works out of the box; light deps):
pip install -e .

# Local ESM-C inference (downloads open weights on first use; GPU recommended):
pip install -e ".[esmc]"

# FAISS for fast/large-scale k-NN (otherwise an exact numpy fallback is used):
pip install -e ".[faiss]"

# Dev / tests:
pip install -e ".[dev]" && pytest
```

`torch`/`esm` are **optional**: the whole pipeline (and the test suite) runs
without them via the `hash` backend, so CI and plumbing never need a model
download or a GPU.

### macOS / Apple Silicon (M-series)

Use a Python **3.11 or 3.12** virtualenv — 3.13/3.14 are ahead of some ML wheels
(notably `faiss-cpu`) and lead to source builds or missing wheels.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[esmc]"     # torch + esm (CPU build runs fine on M-series)
# FAISS is optional; if you want it on macOS, prefer conda-forge to avoid a
# second OpenMP runtime (see below). Otherwise omit it — ESMSeek falls back to
# an exact numpy search automatically.
```

**`OMP: Error #15 ... libomp.dylib already initialized` (abort/crash).** PyTorch's
bundled OpenMP collides with another `libomp` in the process. ESMSeek's CLI
**auto-applies the documented workaround on macOS** (`KMP_DUPLICATE_LIB_OK=TRUE`
+ `OMP_NUM_THREADS=1`, both `setdefault`, so your own settings win), so the
`search`/`embed` commands should no longer crash. If you invoke Torch yourself,
set those two variables before importing it. To fully remove the duplicate,
install FAISS via conda-forge (single libomp) or skip FAISS entirely.

The default device on a Mac is CPU; pass `--device mps` to try the Metal backend
(faster, but some ops may fall back to CPU).

## Quick start

```bash
# DNA query (translated into ORFs):
esmseek search \
  --query examples/contigs.fna \
  --seeds examples/seeds.faa \
  --out   hits.tsv \
  --backend esmc-local --model esmc_300m \
  --min-aa 100 --top-k 50 --min-score 0.5

# Amino-acid query (each record searched as-is, no translation):
esmseek search --query examples/proteins.faa --seeds examples/seeds.faa \
  --backend esmc-local -o hits.tsv
```

`-q` and `--in` are short forms of `--query`; `--dna` is kept as a backward-
compatible alias. Force interpretation with `--seq-type {auto,dna,protein}`.

Smoke-test with no model download (deterministic k-mer backend):

```bash
esmseek search --query examples/contigs.fna --seeds examples/seeds.faa \
  --backend hash --min-aa 60 -o hits.tsv
```

```
candidate_id                         seed_id               cosine   seed_rank  origin  ...  aa_len  aa_seq
contig_metagenome_001|orf1|+1|4-390  LSR_seed_recombinase  0.87...  1          orf     ...  129     MSKV...
```

## Inputs

* `--query` (aliases `-q`, `--in`, `--dna`) — FASTA of sequences to search.
  Each record is auto-detected: **DNA** records are translated in all six frames
  and cut into ORFs; **amino-acid** records are searched directly. A mixed FASTA
  (some DNA, some protein) works. Override detection with `--seq-type dna` or
  `--seq-type protein`. (File extension is irrelevant — detection is by content,
  so a `.fna`/`.faa`/`.fasta` of either type is fine.)
* `--seeds` — FASTA of one or more seed proteins. Seeds may be amino-acid **or**
  DNA (auto-detected; DNA seeds are translated to their longest ORF).

### ORF finding

DNA is translated in all 6 frames; ORFs are **maximal stop-to-stop runs** by
default (sensitive — recovers N-terminally truncated genes common in fragmented
contigs/reads). Use `--require-start` to trim each ORF to its first Met, and
`--min-aa` to set the minimum length (default 100; for full-length LSRs you may
want ~300+).

## Output (TSV)

One row per hit, sorted by descending cosine (deterministic tie-break on IDs).
By default the **best seed per candidate** is reported; `--all-pairs` emits one
row per (candidate, seed).

| column | meaning |
|--------|---------|
| `candidate_id` | `\|`-delimited: `source\|orfN\|<strand><frame>\|start-end` (or the protein ID) |
| `seed_id` | seed this candidate best matched |
| `cosine` | cosine similarity in [-1, 1] (the rank score) |
| `seed_rank` | rank of this candidate within that seed's hit list |
| `origin` | `orf` (translated from DNA) or `protein` |
| `source_id` | source contig / record |
| `strand`, `frame` | `+`/`-`/`.`, reading frame 1–3 (0 for protein input) |
| `nt_start`, `nt_end` | 1-based inclusive forward-strand ORF coordinates |
| `aa_len` | candidate length in residues |
| `aa_seq` | candidate amino-acid sequence (omit with `--no-seq`) |
| `pvalue`, `qvalue` | *only with `--calibrate`* — empirical significance + BH-FDR |

## Backends

| `--backend` | description |
|-------------|-------------|
| `esmc-local` *(default)* | ESM-C open weights run locally (`--model esmc_300m`/`esmc_600m`, `--device auto\|cpu\|cuda`). |
| `esmc-forge` | ESM-C via the hosted Forge API (`--forge-token` or `ESM_FORGE_TOKEN`; `--model esmc-600m-2024-12`). |
| `hash` | Deterministic k-mer feature hashing — no model, no GPU. For CI, plumbing, and reproducible fixtures. **Not** a structural model. |

`--cache-dir DIR` caches embeddings per sequence (keyed by model + sequence
hash) so re-runs over overlapping data are cheap — useful in an iterative
discovery pipeline.

## Search

Candidate vectors are L2-normalised and indexed; seeds query the index, so inner
product = cosine. FAISS (`IndexFlatIP`, exact) is used when available, with an
exact numpy fallback that returns identical results. Control with
`--faiss auto|always|never`, results per seed with `--top-k`, and the reporting
threshold with `--min-score`.

> **On thresholds:** a "good" cosine cutoff is model-dependent (ESM-C pooled
> embeddings are not zero-centred, so even unrelated proteins share a high
> baseline cosine). Tier 1 leaves `--min-score` to you; **Tier 2 calibration**
> exists precisely to replace this guesswork with FDR-controlled significance.

## Tier 2: calibration (experimental)

```bash
esmseek search ... --calibrate shuffle:25
```

Generates 25 composition-matched decoys per candidate, scores them against the
seeds to build a null distribution, and annotates each hit with an empirical
`pvalue` and a Benjamini–Hochberg `qvalue`. The math (`make_decoys`,
`empirical_pvalues`, `benjamini_hochberg`) is implemented and tested; the
remaining Tier-2 work is empirical tuning of the decoy model and calibration set
size, hence the experimental flag.

## Per-residue aligner (experimental)

Pooled cosine collapses each protein to one vector, so it can't reward a *local*
stretch of structural similarity flanked by divergent regions. `esmseek.align`
adds a pLM-BLAST-style alternative: it pulls the full per-residue ESM-C matrix
(`Embedder.embed_residues` — same model pass as pooling, minus the mean), builds
the residue-by-residue cosine grid, subtracts an anisotropy offset, and runs
affine-gap **Smith–Waterman** for the best local score. The inner loop is
numba-JIT'd when available, with a NumPy fallback.

It is wired as a third engine for the discrimination test in
`comparisons/run_aligner.py` (pooled cosine prefilter + numba keep it tractable
over ~700 proteins). See `comparisons/results/v2/README.md` for the method and
the reproduce commands.

```bash
python comparisons/run_aligner.py --seeds seeds.faa --pool candidates.faa \
  --out esmc_aln.score.tsv --backend esmc-local --model esmc_300m \
  --cache-dir .emb_cache --align-seeds 3 --estimate-anisotropy
```

## Utility: `embed`

Precompute/export embeddings (e.g. to build a reusable candidate index):

```bash
esmseek embed --in proteins.faa -o vectors --backend esmc-local
# -> vectors.npy  (N x dim float32) and vectors.ids.txt
```

## Layout

```
src/esmseek/
  cli.py           # argparse CLI: `search`, `embed`
  pipeline.py      # orchestration + SearchConfig/SearchResult
  translate.py     # DNA detection, 6-frame translation, ORF finding
  search.py        # L2-normalise + FAISS/numpy cosine k-NN
  align.py         # per-residue Smith–Waterman aligner (pLM-BLAST style)
  embedders/       # Embedder ABC, hashing, ESM-C (local/forge), disk cache
  calibrate.py     # Tier-2 decoy calibration + FDR
  seqio.py         # FASTA in, TSV out
tests/             # pytest suite (runs on the `hash` backend, no model needed)
examples/          # tiny demo contig + seed
comparisons/       # discrimination test harness (pooled vs aligner vs Foldseek)
```

## License

MIT © Cam Nowack
