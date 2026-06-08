"""ESMSeek — structural homology search over raw DNA using protein embeddings.

ESMSeek translates raw DNA into candidate ORFs, embeds them (and a set of
seed proteins) with a protein language model — ESM-C by default — and ranks
candidates by cosine similarity to the seeds via FAISS k-NN. The intent is
embedding-based *structural* homology: hits that sequence search (BLAST) would
miss at low identity but that fold/function like the seeds.

This is "Tier 1" of a two-tier design:

* Tier 1 (implemented): pooled embeddings + cosine/FAISS k-NN, FASTA in, TSV out.
* Tier 2 (scaffolded in :mod:`esmseek.calibrate`): decoy calibration + FDR
  thresholding so scores become calibrated significance values.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
