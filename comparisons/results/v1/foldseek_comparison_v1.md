# Foldseek/ProstT5 vs. ESM-C — LSR Discrimination Test (v1)

*Comparison engine selection for the stand-alone LSR filter. Pooled ESM-C embeddings vs. ProstT5-predicted structure searched with Foldseek.*

## Introduction

The stand-alone tool is a **filter**: MGEfinder screens genomes for mobile genetic
elements and reports att sites, and the tool then scores the MGE-resident ORFs
against a set of seed large serine recombinases (LSRs), passing those above a
similarity threshold. Inside an MGE the only hard decision is separating true
LSRs from co-resident recombinase relatives — small serine recombinases
(resolvases), tyrosine recombinases, and transposases. This test asks a single
question: **which comparison engine separates LSRs from those families better —
ESM-C protein embeddings (cosine similarity) or ProstT5-predicted 3Di structure
searched with Foldseek?** att sites are not used here; they are handled upstream
by MGEfinder.

## Methods

**Test set composition.** Seeds and positives were derived from 70 att-validated
LSRs; decoys were drawn from Pfam/UniProt and redundancy-reduced.

| Group | n | Source / definition |
|---|---|---|
| Seeds (queries) | 15 | Representatives of the 15 largest of 48 clusters (MMseqs2, 40% identity) of 70 validated LSRs |
| LSR positives | 55 | Held-out cluster members (33) + near-relatives of seeds (22); **16 divergent** (<30% identity to any seed) |
| Resolvase decoys | 400 | Pfam PF00239 (UniProt), clustered to 50% identity, randomly subsampled |
| Tyrosine recombinase decoys | 119 | Pfam PF00589, clustered to 50% identity |
| Transposase decoys | 120 | UniProt "transposase" (reviewed), clustered to 50% identity |
| **Candidate pool** | **694** | LSR positives + decoys, with a hidden label key |

**Pipeline.** (1) Cluster and split the 70 LSRs into seeds vs. test positives,
flagging divergent ones. (2) Collect and redundancy-reduce the three decoy
families; assemble the labelled pool. (3) Score every candidate against the
seeds with each engine, keeping each candidate's best score to any seed.
(4) Compare both engines at a threshold holding **95% LSR recall**.

**Model conditions.** Both engines run from sequence only, for a fair comparison.

| Engine | Representation | Search / score |
|---|---|---|
| ESM-C | `esmc_300m`, mean-pooled per-protein embedding (CPU) | Cosine similarity to best seed |
| Foldseek/ProstT5 | ProstT5-predicted 3Di structure string (CPU) | `foldseek search -s 9.5 -e 1000`; best bitscore to best seed |

## Results

**Discrimination metrics (threshold set to 95% LSR recall).** Higher AUROC and
divergent recall are better; lower decoy pass-rates are better.

| Engine | AUROC | LSR recall | Decoy pass | Resolvase pass | Tyrosine pass | Transposase pass | Divergent recall |
|---|---|---|---|---|---|---|---|
| ESM-C | 0.904 | 0.96 | 0.31 | 0.44 | 0.13 | 0.07 | 0.88 |
| Foldseek | **0.928** | 0.96 | **0.26** | **0.41** | **0.00** | **0.00** | 0.88 |

**Top-ranked candidates.** LSR ids are unprefixed; decoys carry a family prefix.

| Rank | ESM-C (cosine) | class | Foldseek (bits) | class |
|---|---|---|---|---|
| 1 | Ec03 — 0.9985 | LSR | Ec03 — 3644 | LSR |
| 2 | Cd16 — 0.9946 | LSR | Cd16 — 3391 | LSR |
| 3 | Sa34 — 0.9926 | LSR | Bm99 — 3158 | LSR |
| 4 | Kp03 — 0.9915 | LSR | Sm18 — 2913 | LSR |
| 5 | Sm18 — 0.9911 | LSR | Cd04 — 2749 | LSR |
| 6 | Bm99 — 0.9910 | LSR | Kp03 — 2702 | LSR |
| 7 | A0ABQ5PNS2 — 0.9905 | resolvase | A0ABQ5PNS2 — 2701 | resolvase |
| 8 | Ec04 — 0.9900 | LSR | No67 — 2662 | LSR |
| 9 | A0A6M8FH19 — 0.9891 | resolvase | Sa02 — 2645 | LSR |
| 10 | A0A3D8PMW2 — 0.9883 | resolvase | A0ABW9CL28 — 2596 | resolvase |

**Rank composition (count of each class among the top N).** Tyrosine and
transposase decoys are absent from the top 50 of *both* engines.

| Top N | ESM-C: LSR / resolvase | Foldseek: LSR / resolvase |
|---|---|---|
| 10 | 7 / 3 | 8 / 2 |
| 25 | 15 / 10 | 20 / 5 |
| 50 | 24 / 26 | 30 / 20 |

**Summary.** Overall AUROC favours Foldseek (0.928 vs. 0.904), but with only 55
positives that gap (~0.02) is within the expected sampling noise (~±0.04) and
should be read as a tie. The signal that is *not* noise is the per-family
rejection: Foldseek perfectly rejects tyrosine recombinases and transposases
(both 0.00 pass vs. 0.13 and 0.07 for ESM-C) — structure search cleanly turns
away proteins of a genuinely different fold, while sequence embeddings pick up
spurious cross-fold similarity. Foldseek is also cleaner in the deeper ranks
(20 vs. 15 LSRs in the top 25; 30 vs. 24 in the top 50). Both engines hit the
*same* wall on resolvases: ~41–44% leak through at 95% LSR recall, and the same
resolvase (`A0ABQ5PNS2`) ranks 7th under both. Divergent-LSR recall is identical
(0.88, ~14/16).

## Conclusion

**Adopt Foldseek/ProstT5 as the comparison engine.** It equals or beats pooled
ESM-C on every metric, decisively rejects non-serine folds, matches it on LSR
and divergent recall, and is operationally simpler — a single CPU binary with no
PyTorch/`esm` environment. The custom per-residue ESM-C aligner is therefore not
justified: pooled ESM-C only matches the off-the-shelf tool, which is simpler and
structurally more principled.

**The residual limitation is resolvases, and it is shared and method-independent.**
Because small serine recombinases share the LSR catalytic serine fold, neither a
structural nor an embedding similarity score separates them at the protein level.
Part of the ~41% leakage is likely **decoy contamination**: PF00239 *is* the LSR
catalytic domain, so the resolvase set probably contains genuine large serine
recombinases that legitimately resemble LSRs (pending a length check — small
resolvases are ~150–200 aa, LSRs ~390–650 aa).

**Recommendation.** Use Foldseek with the threshold calibrated here, and add a
**length / domain-architecture gate** (require the large size and C-terminal
DNA-binding region that small resolvases lack) to remove the genuinely small
serine recombinases the similarity score cannot. Pass the remaining
large-serine-recombinase cases to the downstream att-site and validation steps.

**Caveats.** Small positive set (55; 16 divergent) — treat single-metric gaps
cautiously and bootstrap before trusting a winner. Thresholds are specific to the
model/checkpoint used. This test measures discrimination of co-resident families
(the filter's job), not remote discovery versus the HMMER primary.