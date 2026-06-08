"""Shared helpers for tests (importable thanks to pytest's rootdir on sys.path)."""

from __future__ import annotations

from esmseek.translate import _CODON_TABLE

# One representative sense codon per amino acid (inverse of the codon table).
_AA_TO_CODON = {}
for _codon, _aa in _CODON_TABLE.items():
    if _aa != "*" and _aa not in _AA_TO_CODON:
        _AA_TO_CODON[_aa] = _codon

# A fixed ~120-residue protein used as a planted ORF / seed in pipeline tests.
DEMO_PROTEIN = (
    "MSKVLTAQEIIDRLNKGEKLSVKDLAEELGVSRQTIYRWLNGESDLRPSTAKKIADALGVS"
    "VEELFGRDEVKQLLDGMSPEEIANRLGISRQQVYRWVKEGRLPAPDFKIGKRLYVPADAVE"
    "WLLSRQE"
)


def reverse_translate(protein: str) -> str:
    """Turn a protein into a DNA CDS using one fixed codon per residue."""
    return "".join(_AA_TO_CODON[aa] for aa in protein)


def planted_contig(protein: str = DEMO_PROTEIN, pad_codons: int = 3) -> str:
    """A DNA contig with `protein` framed by stop-codon runs on both sides."""
    stops = "TAA" * pad_codons
    return stops + reverse_translate(protein) + stops
