"""Lightweight data containers passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SeqRecord:
    """A FASTA record (DNA or protein)."""

    id: str
    seq: str
    description: str = ""


@dataclass
class Candidate:
    """A candidate protein to be scored against the seeds.

    For ORFs translated from DNA the coordinate fields locate the ORF on the
    source contig (1-based, inclusive ``nt_start``/``nt_end`` on the forward
    strand regardless of ``strand``). For protein inputs the coordinate fields
    are left at their defaults.
    """

    id: str
    aa_seq: str
    source_id: str                 # contig / source record the candidate came from
    origin: str = "protein"        # "orf" if translated from DNA, else "protein"
    strand: str = "."              # "+", "-", or "." (protein input)
    frame: int = 0                 # reading frame 1..3 (0 for protein input)
    nt_start: int = 0              # 1-based inclusive forward-strand coordinate
    nt_end: int = 0                # 1-based inclusive forward-strand coordinate

    @property
    def aa_len(self) -> int:
        return len(self.aa_seq)


@dataclass
class Hit:
    """A scored (candidate, seed) match, ready to be written to TSV."""

    candidate: Candidate
    seed_id: str
    score: float                   # engine score, higher = better (scale depends on engine)
    seed_rank: int = 0             # rank of this candidate within the seed's hit list (1 = best)
    # Tier-2 calibration fields (populated only when calibration is enabled).
    pvalue: Optional[float] = None
    qvalue: Optional[float] = None
    extra: dict = field(default_factory=dict)
