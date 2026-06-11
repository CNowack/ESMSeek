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


def test_cli_search_amino_acid_query(tmp_path):
    # Amino-acid records are accepted as searchable input and used as-is.
    query = tmp_path / "proteins.faa"
    query.write_text(">p1\n" + DEMO_PROTEIN + "\n>p2\nWWWWPPPPCCCCYYYY\n")
    seeds = tmp_path / "seeds.faa"
    seeds.write_text(">seedA\n" + DEMO_PROTEIN + "\n")
    out = tmp_path / "hits.tsv"
    rc = main([
        "search", "--query", str(query), "--seeds", str(seeds),
        "-o", str(out), "--backend", "hash", "--hash-dim", "2048", "--quiet",
    ])
    assert rc == 0
    rows = [dict(zip(HIT_COLUMNS, ln.split("\t")))
            for ln in out.read_text().strip().splitlines()[1:]]
    top = rows[0]
    assert top["candidate_id"] == "p1"          # protein id used directly
    assert top["origin"] == "protein"           # no ORF translation
    assert top["strand"] == "."
    assert float(top["cosine"]) == pytest.approx(1.0, abs=1e-4)


def test_cli_search_dna_alias_still_works(tmp_path):
    dna, seeds = _make_inputs(tmp_path)
    out = tmp_path / "hits.tsv"
    rc = main([  # legacy --dna flag must keep working
        "search", "--dna", str(dna), "--seeds", str(seeds),
        "-o", str(out), "--backend", "hash", "--min-aa", "50", "--quiet",
    ])
    assert rc == 0
    assert out.read_text().strip().splitlines()[1:], "expected hits via --dna alias"


def test_cli_search_mixed_dna_and_protein_query(tmp_path):
    # One DNA contig (translated) and one protein record in the same FASTA.
    query = tmp_path / "mixed.fasta"
    query.write_text(
        ">contig_dna\n" + planted_contig() + "\n"
        ">protein_rec\n" + DEMO_PROTEIN + "\n"
    )
    seeds = tmp_path / "seeds.faa"
    seeds.write_text(">seedA\n" + DEMO_PROTEIN + "\n")
    out = tmp_path / "hits.tsv"
    rc = main([
        "search", "-q", str(query), "--seeds", str(seeds),
        "-o", str(out), "--backend", "hash", "--hash-dim", "2048",
        "--min-aa", "50", "--all-pairs", "--quiet",
    ])
    assert rc == 0
    rows = [dict(zip(HIT_COLUMNS, ln.split("\t")))
            for ln in out.read_text().strip().splitlines()[1:]]
    origins = {r["origin"] for r in rows}
    assert origins == {"orf", "protein"}  # both record types contributed candidates


def test_cli_query_and_dna_mutually_exclusive(tmp_path):
    dna, seeds = _make_inputs(tmp_path)
    with pytest.raises(SystemExit):  # argparse rejects passing both
        main([
            "search", "--query", str(dna), "--dna", str(dna),
            "--seeds", str(seeds), "--backend", "hash", "--quiet",
        ])


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
