from __future__ import annotations

import pytest

from esmseek.records import SeqRecord
from esmseek.translate import (
    candidates_from_record,
    find_orfs,
    is_dna,
    reverse_complement,
    seed_to_protein,
    translate_cds,
    translate_frame,
)

from _util import DEMO_PROTEIN, planted_contig, reverse_translate


def test_is_dna():
    assert is_dna("ACGTACGTNNAC")
    assert is_dna("acgtacgt")
    assert not is_dna("MSKVLTAQEII")
    assert not is_dna("")  # empty -> protein


def test_reverse_complement():
    assert reverse_complement("ATGC") == "GCAT"
    assert reverse_complement("AACGTT") == "AACGTT"  # palindrome


def test_translate_frame_and_cds():
    # ATG=M, AAA=K, TAA=stop
    assert translate_frame("ATGAAATAA") == "MK*"
    assert translate_cds("ATGAAATAA") == "MK"
    assert translate_frame("ATGNNNAAA") == "MXK"  # ambiguous codon -> X
    assert translate_frame("ATGA") == "M"          # trailing partial codon ignored


def test_find_orfs_forward_coords():
    cds = reverse_translate("MKV")            # 9 nt
    contig = "TAA" + cds + "TAA"              # stop, ORF, stop  (15 nt)
    rec = SeqRecord(id="c1", seq=contig)
    orfs = find_orfs(rec, min_aa=3)
    fwd = [o for o in orfs if o.strand == "+" and o.aa_seq == "MKV"]
    assert fwd, "planted forward ORF not found"
    o = fwd[0]
    # ORF starts after the first stop codon: nt 4..12 (1-based inclusive).
    assert (o.nt_start, o.nt_end) == (4, 12)
    assert o.frame == 1
    assert contig[o.nt_start - 1 : o.nt_end] == cds


def test_find_orfs_reverse_coords_roundtrip():
    prot = "MKVLT"
    cds = reverse_translate(prot)
    # Build a clean forward ORF then flip the whole contig, so the ORF lives on
    # the reverse strand (delimited by stops when read in reverse-complement).
    contig = reverse_complement("TAA" + cds + "TAA")
    rec = SeqRecord(id="c1", seq=contig)
    orfs = find_orfs(rec, min_aa=len(prot))
    rev = [o for o in orfs if o.strand == "-" and o.aa_seq == prot]
    assert rev, "planted reverse ORF not recovered"
    o = rev[0]
    # Forward-strand coordinates should slice out the reverse-complemented CDS.
    assert reverse_complement(contig[o.nt_start - 1 : o.nt_end]) == cds


def test_find_orfs_min_aa_filter():
    rec = SeqRecord(id="c1", seq=planted_contig())
    short = find_orfs(rec, min_aa=5)
    long = find_orfs(rec, min_aa=1000)
    assert len(long) < len(short)
    assert any(o.aa_seq == DEMO_PROTEIN for o in short)


def test_require_start_trims_to_met():
    # Protein with a leading non-Met stretch before the first M.
    prot = "AAAMKVLT"
    contig = "TAA" + reverse_translate(prot) + "TAA"
    rec = SeqRecord(id="c1", seq=contig)
    with_start = find_orfs(rec, min_aa=1, require_start=True)
    assert any(o.aa_seq == "MKVLT" for o in with_start)
    assert not any(o.aa_seq.startswith("AAA") for o in with_start)


def test_candidates_from_record_protein_passthrough():
    rec = SeqRecord(id="p1", seq="MSKVLT*")
    cands = candidates_from_record(rec, seq_type="auto")
    assert len(cands) == 1
    assert cands[0].origin == "protein"
    assert cands[0].aa_seq == "MSKVLT"  # trailing stop stripped


def test_seed_to_protein_dna_and_aa():
    # AA seed used directly.
    assert seed_to_protein(SeqRecord("s", "MSKVLT")) == "MSKVLT"
    # DNA seed -> longest ORF translation.
    dna = reverse_translate(DEMO_PROTEIN)
    assert seed_to_protein(SeqRecord("s", dna)) == DEMO_PROTEIN


def test_seed_to_protein_untranslatable_dna_raises():
    # Too short to yield any codon in any frame -> no protein resolvable.
    with pytest.raises(ValueError):
        seed_to_protein(SeqRecord("s", "NN"), seq_type="dna")
