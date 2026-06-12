"""Foldseek/ProstT5 structural-search engine for ESMSeek.

Scores candidates against seeds by predicting a 3Di structural alphabet from
sequence with ProstT5 and searching it with Foldseek — the same engine the v1/v2
discrimination test found statistically indistinguishable from the ESM-C paths,
but with far less runtime overhead (a single CPU binary, no torch/esm), which is
why it is ESMSeek's default engine.

This module shells out to the ``foldseek`` binary. The orchestration
(:func:`score_matrix`) is kept thin and the alignment parsing
(:func:`parse_alignments`) is a pure function so it can be unit-tested without
the binary present.

Requirements at run time (not import time):
  * the ``foldseek`` executable on ``PATH`` (or an explicit path), and
  * a ProstT5 weights directory, so Foldseek can fold sequence → 3Di. Point to
    it with ``--foldseek-prostt5`` or the ``FOLDSEEK_PROSTT5_MODEL`` env var.
    Download once with ``foldseek databases ProstT5 <dir> tmp``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Sequence, Tuple

import numpy as np


class FoldseekError(RuntimeError):
    """Raised when Foldseek is unavailable or a subprocess step fails."""


def _resolve_binary(foldseek_bin: str) -> str:
    path = shutil.which(foldseek_bin)
    if path is None:
        raise FoldseekError(
            f"Foldseek binary {foldseek_bin!r} not found on PATH. Install it "
            "(e.g. `conda install -c bioconda foldseek`) or pass --foldseek-bin, "
            "or switch engine with --engine esmc-align / esmc-pooled."
        )
    return path


def _resolve_prostt5(prostt5_model: str | None) -> str:
    model = prostt5_model or os.environ.get("FOLDSEEK_PROSTT5_MODEL")
    if not model:
        raise FoldseekError(
            "A ProstT5 weights directory is required for the foldseek engine. "
            "Download once with `foldseek databases ProstT5 <dir> tmp`, then pass "
            "--foldseek-prostt5 <dir> or set FOLDSEEK_PROSTT5_MODEL."
        )
    if not os.path.exists(model):
        raise FoldseekError(f"ProstT5 model path does not exist: {model!r}")
    return model


def _write_fasta(path: str, ids: Sequence[str], seqs: Sequence[str]) -> None:
    with open(path, "w") as fh:
        for sid, seq in zip(ids, seqs):
            fh.write(f">{sid}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i : i + 60] + "\n")


def parse_alignments(
    text: str,
    n_seeds: int,
    n_cands: int,
    seed_key: Dict[str, int],
    cand_key: Dict[str, int],
) -> np.ndarray:
    """Turn a Foldseek ``convertalis`` table into a ``(n_seeds, n_cands)`` matrix.

    Each input line is ``query<TAB>target<TAB>bits`` where ``query`` is a seed key
    and ``target`` a candidate key (as written into the temp FASTAs). The best
    (max) bitscore is kept per (seed, candidate); unscored pairs stay 0.
    """
    S = np.zeros((n_seeds, n_cands), dtype=np.float32)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        q, t, bits = parts[0], parts[1], parts[2]
        si, ci = seed_key.get(q), cand_key.get(t)
        if si is None or ci is None:
            continue
        try:
            b = float(bits)
        except ValueError:
            continue
        if b > S[si, ci]:
            S[si, ci] = b
    return S


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise FoldseekError(
            f"Foldseek step failed ({' '.join(cmd[:2])} …, exit {proc.returncode}):\n"
            + (proc.stdout or "")[-2000:]
        )


def score_matrix(
    seeds: Sequence[Tuple[str, str]],
    candidates: Sequence[Tuple[str, str]],
    *,
    foldseek_bin: str = "foldseek",
    prostt5_model: str | None = None,
    sensitivity: float = 9.5,
    evalue: float = 1000.0,
    threads: int = 0,
) -> np.ndarray:
    """Best Foldseek bitscore of every candidate against every seed.

    ``seeds`` / ``candidates`` are ``(id, protein_sequence)`` pairs. Returns a
    ``(n_seeds, n_candidates)`` float32 matrix (higher = more similar; 0 = no hit).

    Internal FASTA ids are positional (``q{i}`` / ``t{j}``) so candidate ids
    containing ``|`` or whitespace can't confuse Foldseek; results are mapped
    back by position.
    """
    binary = _resolve_binary(foldseek_bin)
    model = _resolve_prostt5(prostt5_model)

    seed_ids = [f"q{i}" for i in range(len(seeds))]
    cand_ids = [f"t{j}" for j in range(len(candidates))]
    seed_key = {sid: i for i, sid in enumerate(seed_ids)}
    cand_key = {cid: j for j, cid in enumerate(cand_ids)}

    with tempfile.TemporaryDirectory(prefix="esmseek_foldseek_") as tmp:
        q_fa = os.path.join(tmp, "query.fasta")
        t_fa = os.path.join(tmp, "target.fasta")
        _write_fasta(q_fa, seed_ids, [s for _, s in seeds])
        _write_fasta(t_fa, cand_ids, [s for _, s in candidates])

        qdb = os.path.join(tmp, "qdb")
        tdb = os.path.join(tmp, "tdb")
        aln = os.path.join(tmp, "aln")
        tsv = os.path.join(tmp, "aln.tsv")
        ftmp = os.path.join(tmp, "tmp")

        thread_args = ["--threads", str(threads)] if threads and threads > 0 else []
        # createdb with --prostt5-model folds sequence -> 3Di in one step.
        _run([binary, "createdb", q_fa, qdb, "--prostt5-model", model] + thread_args)
        _run([binary, "createdb", t_fa, tdb, "--prostt5-model", model] + thread_args)
        _run([binary, "search", qdb, tdb, aln, ftmp,
              "-s", str(sensitivity), "-e", str(evalue)] + thread_args)
        _run([binary, "convertalis", qdb, tdb, aln, tsv,
              "--format-output", "query,target,bits"] + thread_args)

        with open(tsv) as fh:
            text = fh.read()

    return parse_alignments(text, len(seeds), len(candidates), seed_key, cand_key)
