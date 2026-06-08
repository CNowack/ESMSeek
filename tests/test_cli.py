from __future__ import annotations

import numpy as np
import pytest

from esmseek.cli import main
from esmseek.seqio import HIT_COLUMNS

from _util import DEMO_PROTEIN, planted_contig


def _make_inputs(tmp_path):
    dna = tmp_path / "contigs.fasta"
    dna.write_text(">contig1\n" + planted_contig() + "\n")
    seeds = tmp_path / "seeds.fasta"
    seeds.write_text(">seedLSR\n" + DEMO_PROTEIN + "\n")
    return dna, seeds


def test_cli_search_writes_tsv(tmp_path):
    dna, seeds = _make_inputs(tmp_path)
    out = tmp_path / "hits.tsv"
    rc = main([
        "search", "--dna", str(dna), "--seeds", str(seeds),
        "-o", str(out), "--backend", "hash", "--hash-dim", "2048",
        "--min-aa", "50", "--quiet",
    ])
    assert rc == 0
    lines = out.read_text().strip().splitlines()
    assert lines[0].split("\t") == list(HIT_COLUMNS)
    rows = [dict(zip(HIT_COLUMNS, ln.split("\t"))) for ln in lines[1:]]
    assert rows, "no hit rows written"
    assert rows[0]["aa_seq"] == DEMO_PROTEIN
    assert float(rows[0]["cosine"]) == pytest.approx(1.0, abs=1e-4)
    assert rows[0]["seed_id"] == "seedLSR"


def test_cli_search_no_seq_flag(tmp_path):
    dna, seeds = _make_inputs(tmp_path)
    out = tmp_path / "hits.tsv"
    main([
        "search", "--dna", str(dna), "--seeds", str(seeds),
        "-o", str(out), "--backend", "hash", "--min-aa", "50",
        "--no-seq", "--quiet",
    ])
    header = out.read_text().splitlines()[0].split("\t")
    assert "aa_seq" not in header


def test_cli_search_calibrate(tmp_path):
    dna, seeds = _make_inputs(tmp_path)
    out = tmp_path / "hits.tsv"
    rc = main([
        "search", "--dna", str(dna), "--seeds", str(seeds),
        "-o", str(out), "--backend", "hash", "--min-aa", "50",
        "--calibrate", "shuffle:3", "--quiet",
    ])
    assert rc == 0
    header = out.read_text().splitlines()[0].split("\t")
    assert header[-2:] == ["pvalue", "qvalue"]


def test_cli_embed(tmp_path):
    fa = tmp_path / "prots.fasta"
    fa.write_text(">a\n" + DEMO_PROTEIN + "\n>b\nMSKVLTAQEII\n")
    prefix = tmp_path / "emb"
    rc = main([
        "embed", "--in", str(fa), "-o", str(prefix),
        "--backend", "hash", "--hash-dim", "256", "--quiet",
    ])
    assert rc == 0
    vecs = np.load(f"{prefix}.npy")
    assert vecs.shape == (2, 256)
    ids = (tmp_path / "emb.ids.txt").read_text().split()
    assert ids == ["a", "b"]


def test_cli_missing_file_returns_error(tmp_path):
    seeds = tmp_path / "seeds.fasta"
    seeds.write_text(">s\n" + DEMO_PROTEIN + "\n")
    rc = main([
        "search", "--dna", str(tmp_path / "nope.fasta"),
        "--seeds", str(seeds), "--backend", "hash", "--quiet",
    ])
    assert rc == 2
