"""Engine selection: esmc-align via the hash backend, and the foldseek wrapper.

The foldseek binary isn't available in CI, so we test the pure parsing function
and the dispatch/guard behaviour, not a live foldseek run.
"""

from __future__ import annotations

import numpy as np
import pytest

from esmseek import foldseek
from esmseek.pipeline import SearchConfig, run_search

from _util import DEMO_PROTEIN, planted_contig


def _inputs(tmp_path):
    dna = tmp_path / "contigs.fasta"
    dna.write_text(">contig1\n" + planted_contig() + "\n")
    seeds = tmp_path / "seeds.fasta"
    seeds.write_text(">seedLSR\n" + DEMO_PROTEIN + "\n")
    return str(dna), str(seeds)


def test_esmc_align_engine_recovers_planted_orf(tmp_path):
    dna, seeds = _inputs(tmp_path)
    cfg = SearchConfig(engine="esmc-align", backend="hash", hash_dim=2048,
                       min_aa=50, top_k=50)
    result = run_search(dna, seeds, cfg)
    assert result.meta["engine"] == "esmc-align"
    assert result.meta["score_type"] == "smith_waterman"
    assert result.hits
    # The planted exact copy of the seed should rank first and score > 0.
    assert result.hits[0].candidate.aa_seq == DEMO_PROTEIN
    assert result.hits[0].score > 0.0


def test_unknown_engine_rejected(tmp_path):
    dna, seeds = _inputs(tmp_path)
    cfg = SearchConfig(engine="nope", backend="hash")
    with pytest.raises(ValueError, match="Unknown engine"):
        run_search(dna, seeds, cfg)


def test_calibrate_only_with_pooled(tmp_path):
    dna, seeds = _inputs(tmp_path)
    cfg = SearchConfig(engine="esmc-align", backend="hash", hash_dim=512,
                       min_aa=50, calibrate_method="shuffle", calibrate_n=2)
    with pytest.raises(ValueError, match="esmc-pooled"):
        run_search(dna, seeds, cfg)


def test_foldseek_parse_alignments_keeps_best_bits():
    seed_key = {"q0": 0, "q1": 1}
    cand_key = {"t0": 0, "t1": 1, "t2": 2}
    text = "\n".join([
        "q0\tt0\t100",
        "q0\tt0\t250",   # higher bits for same pair wins
        "q1\tt2\t42",
        "q9\tt0\t999",   # unknown seed key ignored
        "q0\tbadtarget\t5",  # unknown target ignored
        "malformed line",
    ])
    S = foldseek.parse_alignments(text, 2, 3, seed_key, cand_key)
    assert S.shape == (2, 3)
    assert S[0, 0] == pytest.approx(250.0)
    assert S[1, 2] == pytest.approx(42.0)
    assert S[0, 1] == 0.0 and S[1, 0] == 0.0  # no hit -> 0


def test_foldseek_missing_binary_raises(tmp_path, monkeypatch):
    # No such binary on PATH -> a clear FoldseekError (not a crash).
    monkeypatch.setattr(foldseek.shutil, "which", lambda _: None)
    with pytest.raises(foldseek.FoldseekError, match="not found on PATH"):
        foldseek.score_matrix([("s", DEMO_PROTEIN)], [("c", DEMO_PROTEIN)],
                              foldseek_bin="definitely-not-foldseek")


def test_foldseek_missing_prostt5_raises(tmp_path, monkeypatch):
    # Binary "present" but no ProstT5 model configured.
    monkeypatch.setattr(foldseek.shutil, "which", lambda _: "/usr/bin/foldseek")
    monkeypatch.delenv("FOLDSEEK_PROSTT5_MODEL", raising=False)
    with pytest.raises(foldseek.FoldseekError, match="ProstT5"):
        foldseek.score_matrix([("s", DEMO_PROTEIN)], [("c", DEMO_PROTEIN)])
