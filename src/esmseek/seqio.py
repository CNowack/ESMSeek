"""Minimal, dependency-free FASTA reading and TSV writing.

We deliberately avoid a hard Biopython dependency: a plain-text FASTA parser is
a few lines, keeps the install light, and means the core pipeline runs in any
environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import IO, Iterable, Iterator, List, Sequence, Union

from .records import Hit, SeqRecord

PathLike = Union[str, Path]


def _open_text(path: PathLike) -> IO[str]:
    """Open a path for reading, transparently handling gzip by extension."""
    path = Path(path)
    if path.suffix == ".gz":
        import gzip

        return gzip.open(path, "rt")
    return open(path, "r")


def parse_fasta(path: PathLike) -> List[SeqRecord]:
    """Parse a (optionally gzipped) FASTA file into :class:`SeqRecord` objects."""
    records: List[SeqRecord] = []
    rec_id: str | None = None
    desc = ""
    chunks: List[str] = []

    def flush() -> None:
        if rec_id is not None:
            records.append(SeqRecord(id=rec_id, seq="".join(chunks), description=desc))

    with _open_text(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
                parts = header.split(None, 1)
                rec_id = parts[0] if parts else ""
                desc = parts[1] if len(parts) > 1 else ""
                chunks = []
            else:
                chunks.append(line.strip())
        flush()

    if not records:
        raise ValueError(f"No FASTA records found in {path!r}")
    return records


# Order of columns emitted to the TSV. Keeping this as data makes the output
# contract explicit and easy to extend (Tier 2 appends pvalue/qvalue).
HIT_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "seed_id",
    "score",
    "seed_rank",
    "origin",
    "source_id",
    "strand",
    "frame",
    "nt_start",
    "nt_end",
    "aa_len",
    "aa_seq",
)

CALIBRATION_COLUMNS: tuple[str, ...] = ("pvalue", "qvalue")


def _hit_row(hit: Hit, include_seq: bool, include_calibration: bool) -> List[str]:
    c = hit.candidate
    row = {
        "candidate_id": c.id,
        "seed_id": hit.seed_id,
        "score": f"{hit.score:.6f}",
        "seed_rank": str(hit.seed_rank),
        "origin": c.origin,
        "source_id": c.source_id,
        "strand": c.strand,
        "frame": str(c.frame),
        "nt_start": str(c.nt_start),
        "nt_end": str(c.nt_end),
        "aa_len": str(c.aa_len),
        "aa_seq": c.aa_seq if include_seq else "",
    }
    cols = [col for col in HIT_COLUMNS if include_seq or col != "aa_seq"]
    out = [row[col] for col in cols]
    if include_calibration:
        out.append("" if hit.pvalue is None else f"{hit.pvalue:.3e}")
        out.append("" if hit.qvalue is None else f"{hit.qvalue:.3e}")
    return out


def write_tsv(
    hits: Sequence[Hit],
    out: PathLike | None,
    include_seq: bool = True,
    include_calibration: bool = False,
) -> None:
    """Write hits as TSV. ``out=None`` (or ``"-"``) writes to stdout."""
    cols = [c for c in HIT_COLUMNS if include_seq or c != "aa_seq"]
    header = list(cols)
    if include_calibration:
        header += list(CALIBRATION_COLUMNS)

    fh: IO[str]
    close = False
    if out is None or str(out) == "-":
        fh = sys.stdout
    else:
        fh = open(out, "w")
        close = True
    try:
        fh.write("\t".join(header) + "\n")
        for hit in hits:
            fh.write("\t".join(_hit_row(hit, include_seq, include_calibration)) + "\n")
    finally:
        if close:
            fh.close()


def write_fasta(records: Iterable[SeqRecord], path: PathLike, width: int = 60) -> None:
    """Write records as FASTA (used by tests and the ``embed`` helper)."""
    with open(path, "w") as fh:
        for rec in records:
            header = rec.id if not rec.description else f"{rec.id} {rec.description}"
            fh.write(f">{header}\n")
            seq = rec.seq
            for i in range(0, len(seq), width):
                fh.write(seq[i : i + width] + "\n")


def iter_records(records: Sequence[SeqRecord]) -> Iterator[SeqRecord]:
    yield from records
