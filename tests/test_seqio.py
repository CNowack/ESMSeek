from __future__ import annotations

import pytest

from esmseek.records import Candidate, Hit
from esmseek.seqio import (
    HIT_COLUMNS,
    parse_fasta,
    write_fasta,
    write_tsv,
)
from esmseek.records import SeqRecord


def test_parse_fasta_multiline_and_description(tmp_path):
    p = tmp_path / "in.fasta"
    p.write_text(">c1 a contig\nACGT\nACGT\n>c2\nMSKV\n")
    recs = parse_fasta(p)
    assert [r.id for r in recs] == ["c1", "c2"]
    assert recs[0].seq == "ACGTACGT"
    assert recs[0].description == "a contig"
    assert recs[1].seq == "MSKV"


def test_parse_fasta_empty_raises(tmp_path):
    p = tmp_path / "empty.fasta"
    p.write_text("\n\n")
    with pytest.raises(ValueError):
        parse_fasta(p)


def test_fasta_roundtrip(tmp_path):
    recs = [SeqRecord("a", "MSKVLTAQ", "desc"), SeqRecord("b", "ACGTACGT")]
    p = tmp_path / "out.fasta"
    write_fasta(recs, p, width=4)
    back = parse_fasta(p)
    assert [(r.id, r.seq) for r in back] == [(r.id, r.seq) for r in recs]


def _hit():
    cand = Candidate(
        id="c1|orf1|+1|4-12", aa_seq="MKV", source_id="c1",
        origin="orf", strand="+", frame=1, nt_start=4, nt_end=12,
    )
    return Hit(candidate=cand, seed_id="seedA", score=0.9123, seed_rank=1)


def test_write_tsv_header_and_row(tmp_path):
    p = tmp_path / "hits.tsv"
    write_tsv([_hit()], p)
    lines = p.read_text().strip().splitlines()
    assert lines[0].split("\t") == list(HIT_COLUMNS)
    fields = dict(zip(HIT_COLUMNS, lines[1].split("\t")))
    assert fields["candidate_id"] == "c1|orf1|+1|4-12"
    assert fields["seed_id"] == "seedA"
    assert fields["score"] == "0.912300"
    assert fields["aa_seq"] == "MKV"
    assert fields["strand"] == "+"


def test_write_tsv_no_seq_omits_column(tmp_path):
    p = tmp_path / "hits.tsv"
    write_tsv([_hit()], p, include_seq=False)
    header = p.read_text().splitlines()[0].split("\t")
    assert "aa_seq" not in header


def test_write_tsv_calibration_columns(tmp_path):
    h = _hit()
    h.pvalue, h.qvalue = 0.001, 0.01
    p = tmp_path / "hits.tsv"
    write_tsv([h], p, include_calibration=True)
    header = p.read_text().splitlines()[0].split("\t")
    assert header[-2:] == ["pvalue", "qvalue"]
