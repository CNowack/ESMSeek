# ESMSeek

**Structural homology search over raw DNA using protein language-model embeddings.**

ESMSeek is a stand-alone CLI that takes raw DNA and a set of seed proteins and
returns the proteins encoded in that DNA that look — *structurally/functionally*,
in embedding space — most like the seeds. It is built to slot into an LSR
(large serine recombinase) discovery pipeline: **FASTA in, TSV out**, no server.

The idea is embedding-based homology. Instead of scoring sequence identity
(BLAST/DIAMOND), ESMSeek embeds each protein with [ESM-C](https://github.com/evolutionaryscale/esm),
mean-pools to a single vector, and ranks candidates by **cosine similarity** to
the seeds via **FAISS k-NN**. This recovers remote homologs that share fold and
function at sequence identities where alignment search goes quiet.

```
raw DNA  ──▶  6-frame ORFs  ──▶  ESM-C embeddings ─┐
                                                   ├─▶  cosine / FAISS k-NN  ──▶  ranked TSV
seeds (AA or DNA) ───────────▶  ESM-C embeddings ──┘
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

## Quick start

```bash
esmseek search \
  --dna   examples/contigs.fna \
  --seeds examples/seeds.faa \
  --out   hits.tsv \
  --backend esmc-local --model esmc_300m \
  --min-aa 100 --top-k 50 --min-score 0.5
```

Smoke-test with no model download (deterministic k-mer backend):

```bash
esmseek search --dna examples/contigs.fna --seeds examples/seeds.faa \
  --backend hash --min-aa 60 -o hits.tsv
```

```
candidate_id                         seed_id               cosine   seed_rank  origin  ...  aa_len  aa_seq
contig_metagenome_001|orf1|+1|4-390  LSR_seed_recombinase  0.87...  1          orf     ...  129     MSKV...
```

## Inputs

* `--dna` — FASTA of raw DNA (or, with `--seq-type protein`, proteins). Each
  record is auto-detected; DNA records are translated in all six frames and cut
  into ORFs.
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
  embedders/       # Embedder ABC, hashing, ESM-C (local/forge), disk cache
  calibrate.py     # Tier-2 decoy calibration + FDR
  seqio.py         # FASTA in, TSV out
tests/             # pytest suite (runs on the `hash` backend, no model needed)
examples/          # tiny demo contig + seed
```

## License

MIT © Cam Nowack
