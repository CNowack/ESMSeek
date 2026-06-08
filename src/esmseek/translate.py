"""DNA handling: type detection, reverse complement, translation and ORF finding.

Pure Python / no third-party dependencies. Coordinates follow the common
genomic convention: ``nt_start``/``nt_end`` are 1-based, inclusive, and always
reported on the *forward* strand (``start <= end``); ``strand`` records the
orientation the ORF was found in.
"""

from __future__ import annotations

from typing import List

from .records import Candidate, SeqRecord

# Standard genetic code (NCBI translation table 1).
_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

_COMPLEMENT = str.maketrans("ACGTUNacgtun", "TGCAANtgcaan")
_NUC_CHARS = set("ACGTUNacgtun")


def is_dna(seq: str, threshold: float = 0.9) -> bool:
    """Heuristically decide whether ``seq`` is a nucleotide sequence.

    Returns ``True`` when at least ``threshold`` of the alphabetic characters
    are nucleotide symbols (A/C/G/T/U/N). Empty sequences are treated as protein.
    """
    letters = [c for c in seq if c.isalpha()]
    if not letters:
        return False
    nuc = sum(1 for c in letters if c in _NUC_CHARS)
    return nuc / len(letters) >= threshold


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def translate_frame(seq: str) -> str:
    """Translate a nucleotide string in frame 0 to amino acids.

    Stop codons become ``*`` and any codon containing a non-ACGT base becomes
    ``X``. A trailing partial codon is ignored.
    """
    seq = seq.upper().replace("U", "T")
    n = len(seq) - (len(seq) % 3)
    out = []
    for i in range(0, n, 3):
        out.append(_CODON_TABLE.get(seq[i : i + 3], "X"))
    return "".join(out)


def translate_cds(seq: str) -> str:
    """Translate a coding sequence and strip a single trailing stop codon."""
    aa = translate_frame(seq)
    return aa[:-1] if aa.endswith("*") else aa


def find_orfs(
    record: SeqRecord,
    min_aa: int = 100,
    require_start: bool = False,
) -> List[Candidate]:
    """Find ORFs in all six reading frames of a DNA record.

    ORFs are maximal stop-to-stop runs (the sensitive default — it does not
    require an in-frame Met and so will recover N-terminally truncated genes
    common in fragmented contigs and reads). Set ``require_start`` to trim each
    ORF to its first Met. Only ORFs with at least ``min_aa`` residues are kept.
    """
    seq = record.seq.upper()
    L = len(seq)
    rc = reverse_complement(seq)
    candidates: List[Candidate] = []
    counter = 0

    for strand, work in (("+", seq), ("-", rc)):
        for frame_offset in range(3):
            protein = translate_frame(work[frame_offset:])
            for seg_start, seg in _iter_segments(protein):
                a_start, a_end = seg_start, seg_start + len(seg)
                aa = seg
                if require_start:
                    m = aa.find("M")
                    if m < 0:
                        continue
                    a_start += m
                    aa = aa[m:]
                if len(aa) < min_aa:
                    continue

                # Coordinates of aa interval [a_start, a_end) within `work`.
                s = frame_offset + 3 * a_start
                e = frame_offset + 3 * (a_start + len(aa))
                if strand == "+":
                    nt_start, nt_end = s + 1, e
                else:  # map reverse-complement coords back onto the forward strand
                    nt_start, nt_end = L - e + 1, L - s

                counter += 1
                cid = f"{record.id}|orf{counter}|{strand}{frame_offset + 1}|{nt_start}-{nt_end}"
                candidates.append(
                    Candidate(
                        id=cid,
                        aa_seq=aa,
                        source_id=record.id,
                        origin="orf",
                        strand=strand,
                        frame=frame_offset + 1,
                        nt_start=nt_start,
                        nt_end=nt_end,
                    )
                )
    return candidates


def _iter_segments(protein: str):
    """Yield ``(start_index, segment)`` for each maximal stop-free run."""
    start = None
    for i, ch in enumerate(protein):
        if ch == "*":
            if start is not None:
                yield start, protein[start:i]
                start = None
        elif start is None:
            start = i
    if start is not None:
        yield start, protein[start:]


def candidates_from_record(
    record: SeqRecord,
    seq_type: str = "auto",
    min_aa: int = 100,
    require_start: bool = False,
) -> List[Candidate]:
    """Turn a FASTA record into candidate proteins.

    ``seq_type`` is ``"auto"`` (detect per record), ``"dna"`` (force ORF
    finding) or ``"protein"`` (use the sequence as-is).
    """
    treat_as_dna = seq_type == "dna" or (seq_type == "auto" and is_dna(record.seq))
    if treat_as_dna:
        return find_orfs(record, min_aa=min_aa, require_start=require_start)
    aa = record.seq.upper().replace("*", "").replace("-", "")
    return [Candidate(id=record.id, aa_seq=aa, source_id=record.id, origin="protein")]


def seed_to_protein(record: SeqRecord, seq_type: str = "auto") -> str:
    """Resolve a seed record to a single protein sequence.

    Protein seeds are used directly. DNA seeds are translated by taking the
    longest stop-to-stop ORF across all six frames (robust to unknown frame /
    strand), which for a clean CDS is simply its translation.
    """
    treat_as_dna = seq_type == "dna" or (seq_type == "auto" and is_dna(record.seq))
    if not treat_as_dna:
        return record.seq.upper().replace("*", "").replace("-", "")

    best = ""
    for work in (record.seq.upper(), reverse_complement(record.seq.upper())):
        for frame_offset in range(3):
            for _, seg in _iter_segments(translate_frame(work[frame_offset:])):
                if len(seg) > len(best):
                    best = seg
    if not best:
        raise ValueError(f"Seed {record.id!r} looks like DNA but no ORF could be translated")
    return best
