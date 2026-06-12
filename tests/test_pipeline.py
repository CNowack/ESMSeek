from __future__ import annotations

import pytest

from esmseek.pipeline import SearchConfig, run_search

from _util import DEMO_PROTEIN, planted_contig


def _write_inputs(tmp_path, min_aa_protein=DEMO_PROTEIN):
    dna = tmp_path / "contigs.fasta"
    # Two contigs: one carries the planted ORF, one is unrelated filler.
    dna.write_text(
        ">contig_with_hit\n" + planted_contig(min_aa_protein) + "\n"
        ">contig_filler\n" + ("ATGGCt" * 40) + "\n"
    )
    seeds = tmp_path / "seeds.fasta"
    seeds.write_text(">seedLSR\n" + DEMO_PROTEIN + "\n")
    return str(dna), str(seeds)


def _cfg(**kw):
    base = dict(engine="esmc-pooled", backend="hash", hash_dim=2048, min_aa=50, top_k=50)
    base.update(kw)
    return SearchConfig(**base)


def test_pipeline_recovers_planted_orf(tmp_path):
    dna, seeds = _write_inputs(tmp_path)
    result = run_search(dna, seeds, _cfg())
    assert result.n_seeds == 1
    assert result.n_candidates > 0
    assert result.hits, "expected at least one hit"
    top = result.hits[0]
    assert top.candidate.aa_seq == DEMO_PROTEIN
    assert top.score == pytest.approx(1.0, abs=1e-4)
    assert top.seed_id == "seedLSR"
    assert top.candidate.origin == "orf"
    assert result.meta["search_backend"] in ("faiss", "numpy")


def test_pipeline_min_score_filters(tmp_path):
    dna, seeds = _write_inputs(tmp_path)
    strict = run_search(dna, seeds, _cfg(min_score=0.999))
    # Only the (near) exact match survives a very strict cutoff.
    assert strict.hits
    assert all(h.score >= 0.999 for h in strict.hits)
    assert strict.hits[0].candidate.aa_seq == DEMO_PROTEIN


def test_pipeline_best_per_candidate_vs_all_pairs(tmp_path):
    dna = tmp_path / "c.fasta"
    dna.write_text(">c\n" + planted_contig() + "\n")
    seeds = tmp_path / "s.fasta"
    seeds.write_text(">s1\n" + DEMO_PROTEIN + "\n>s2\n" + DEMO_PROTEIN + "\n")

    best = run_search(str(dna), str(seeds), _cfg())
    pairs = run_search(str(dna), str(seeds), _cfg(all_pairs=True))
    # best-per-candidate has one row for the planted ORF; all-pairs has two.
    best_ids = [h.candidate.id for h in best.hits]
    assert best_ids.count(best.hits[0].candidate.id) == 1
    assert len(pairs.hits) > len(best.hits)


def test_pipeline_protein_input_mode(tmp_path):
    # Subject is already protein; no ORF finding.
    subj = tmp_path / "prot.fasta"
    subj.write_text(">p1\n" + DEMO_PROTEIN + "\n>p2\nWWWWPPPPCCCCYYYY\n")
    seeds = tmp_path / "s.fasta"
    seeds.write_text(">s\n" + DEMO_PROTEIN + "\n")
    result = run_search(str(subj), str(seeds), _cfg(seq_type="protein"))
    assert result.n_candidates == 2
    assert result.hits[0].candidate.id == "p1"
    assert result.hits[0].candidate.origin == "protein"


def test_pipeline_calibration_annotates(tmp_path):
    dna, seeds = _write_inputs(tmp_path)
    result = run_search(dna, seeds, _cfg(calibrate_method="shuffle", calibrate_n=3))
    assert result.meta.get("calibrated") is True
    assert all(h.pvalue is not None and h.qvalue is not None for h in result.hits)
    # The exact match is the strongest and should be the most significant.
    top = result.hits[0]
    assert top.pvalue <= min(h.pvalue for h in result.hits)


def test_pipeline_faiss_and_numpy_agree(tmp_path):
    dna, seeds = _write_inputs(tmp_path)
    a = run_search(dna, seeds, _cfg(use_faiss="always"))
    b = run_search(dna, seeds, _cfg(use_faiss="never"))
    assert [h.candidate.id for h in a.hits[:5]] == [h.candidate.id for h in b.hits[:5]]
