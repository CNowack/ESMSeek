"""Command-line interface for ESMSeek.

    esmseek search --query contigs.fasta --seeds seeds.fasta -o hits.tsv   # DNA or AA
    esmseek embed  --in proteins.fasta -o prefix          # utility: cache/export vectors
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

from . import __version__
from .embedders import BACKENDS, get_embedder
from .pipeline import ENGINES, SearchConfig, run_search
from .seqio import parse_fasta, write_tsv
from .translate import seed_to_protein


def _add_embedder_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("embedder")
    g.add_argument(
        "--backend", choices=BACKENDS, default="esmc-local",
        help="Embedding backend (default: esmc-local). Use 'hash' for a "
             "dependency-free deterministic backend (CI/smoke tests).",
    )
    g.add_argument(
        "--model", default=None,
        help="Model name (default: esmc_300m for esmc-local, "
             "esmc-600m-2024-12 for esmc-forge).",
    )
    g.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                   help="Torch device (default: auto = cuda if present else cpu). "
                        "On Apple Silicon, '--device mps' tries the Metal backend.")
    g.add_argument("--forge-token", default=None,
                   help="Forge API token (or set ESM_FORGE_TOKEN).")
    g.add_argument("--forge-url", default="https://forge.evolutionaryscale.ai")
    g.add_argument("--cache-dir", default=None,
                   help="Directory to cache embeddings across runs.")
    g.add_argument("--hash-dim", type=int, default=1024,
                   help="Vector dim for the 'hash' backend (default: 1024).")
    g.add_argument("--hash-k", type=int, default=3,
                   help="k-mer size for the 'hash' backend (default: 3).")


def _parse_calibrate(spec: Optional[str]) -> Tuple[Optional[str], int]:
    if not spec:
        return None, 1
    if ":" in spec:
        method, n = spec.split(":", 1)
        return method, int(n)
    return spec, 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="esmseek",
        description="Structural homology search over raw DNA using protein embeddings.",
    )
    parser.add_argument("--version", action="version", version=f"esmseek {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- search ----------------------------------------------------------
    s = sub.add_parser("search", help="Search DNA or protein records for hits to seeds.")
    qin = s.add_mutually_exclusive_group(required=True)
    qin.add_argument(
        "--query", "-q", "--in", dest="query", metavar="FASTA",
        help="FASTA of sequences to search — raw DNA *or* amino acids "
             "(auto-detected per record; DNA is translated into ORFs, protein "
             "records are searched as-is).",
    )
    qin.add_argument(
        "--dna", dest="query", metavar="FASTA",
        help="Alias for --query, kept for backward compatibility.",
    )
    s.add_argument("--seeds", required=True,
                   help="FASTA of seed proteins (AA or DNA) to compare against.")
    s.add_argument("-o", "--out", default="-",
                   help="Output TSV path ('-' for stdout, the default).")
    s.add_argument(
        "--engine", choices=ENGINES, default="foldseek",
        help="Scoring engine (default: foldseek). 'foldseek' = ProstT5 3Di + "
             "Foldseek (least overhead, no torch/esm); 'esmc-align' = per-residue "
             "Smith–Waterman over ESM-C; 'esmc-pooled' = mean-pooled ESM-C cosine.",
    )
    _add_embedder_args(s)

    g = s.add_argument_group("foldseek engine")
    g.add_argument("--foldseek-bin", default="foldseek",
                   help="Path to the foldseek executable (default: 'foldseek' on PATH).")
    g.add_argument("--foldseek-prostt5", default=None,
                   help="ProstT5 weights dir for sequence→3Di (or set "
                        "FOLDSEEK_PROSTT5_MODEL). Get it via "
                        "`foldseek databases ProstT5 <dir> tmp`.")
    g.add_argument("--foldseek-sensitivity", type=float, default=9.5,
                   help="Foldseek search sensitivity -s (default: 9.5).")
    g.add_argument("--foldseek-evalue", type=float, default=1000.0,
                   help="Foldseek search e-value -e (default: 1000).")

    g = s.add_argument_group("esmc-align engine")
    g.add_argument("--align-gap-open", type=float, default=0.5,
                   help="Smith–Waterman gap-open penalty (default: 0.5).")
    g.add_argument("--align-gap-extend", type=float, default=0.1,
                   help="Smith–Waterman gap-extend penalty (default: 0.1).")
    g.add_argument("--align-anisotropy", type=float, default=0.0,
                   help="Constant cosine-baseline offset subtracted from the grid.")
    g.add_argument("--estimate-anisotropy", action="store_true",
                   help="Estimate the anisotropy offset from the data instead.")

    g = s.add_argument_group("translation")
    g.add_argument("--seq-type", choices=["auto", "dna", "protein"], default="auto",
                   help="How to interpret --query records: auto-detect (default), "
                        "force 'dna' (ORF finding) or force 'protein' (use as-is).")
    g.add_argument("--seed-type", choices=["auto", "dna", "protein"], default="auto",
                   help="How to interpret --seeds records (default: auto-detect).")
    g.add_argument("--min-aa", type=int, default=100,
                   help="Minimum ORF length in residues (default: 100).")
    g.add_argument("--require-start", action="store_true",
                   help="Trim each ORF to its first Met (default: keep stop-to-stop).")

    g = s.add_argument_group("search / ranking")
    g.add_argument("--top-k", type=int, default=50,
                   help="Candidates reported per seed; <=0 for all (default: 50).")
    g.add_argument("--min-score", type=float, default=0.0,
                   help="Minimum engine score to report (default: 0.0). The score "
                        "scale depends on --engine (cosine, SW score, or bits).")
    g.add_argument("--all-pairs", action="store_true",
                   help="Emit one row per (candidate, seed) instead of best-per-candidate.")
    g.add_argument("--no-seq", action="store_true",
                   help="Omit the aa_seq column from the TSV.")
    g.add_argument("--faiss", choices=["auto", "always", "never"], default="auto",
                   help="FAISS usage for k-NN (default: auto).")

    g = s.add_argument_group("calibration (Tier 2, experimental)")
    g.add_argument("--calibrate", default=None, metavar="METHOD[:N]",
                   help="Add empirical p/q-values using decoys, e.g. 'shuffle:5'. "
                        "Experimental; off by default.")

    s.add_argument("--quiet", action="store_true", help="Suppress the stderr summary.")
    s.set_defaults(func=_cmd_search)

    # ---- embed -----------------------------------------------------------
    e = sub.add_parser("embed", help="Embed a FASTA and export vectors (.npy + ids).")
    e.add_argument("--in", dest="infile", required=True,
                   help="FASTA of protein (or DNA) sequences.")
    e.add_argument("-o", "--out", required=True,
                   help="Output prefix; writes <prefix>.npy and <prefix>.ids.txt.")
    e.add_argument("--seq-type", choices=["auto", "dna", "protein"], default="auto")
    _add_embedder_args(e)
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(func=_cmd_embed)

    return parser


def _cmd_search(args: argparse.Namespace) -> int:
    method, n_per = _parse_calibrate(args.calibrate)
    cfg = SearchConfig(
        engine=args.engine,
        backend=args.backend,
        model=args.model,
        device=args.device,
        forge_token=args.forge_token,
        forge_url=args.forge_url,
        hash_dim=args.hash_dim,
        hash_k=args.hash_k,
        cache_dir=args.cache_dir,
        seq_type=args.seq_type,
        seed_type=args.seed_type,
        min_aa=args.min_aa,
        require_start=args.require_start,
        top_k=args.top_k,
        min_score=args.min_score,
        all_pairs=args.all_pairs,
        use_faiss=args.faiss,
        align_gap_open=args.align_gap_open,
        align_gap_extend=args.align_gap_extend,
        align_anisotropy=args.align_anisotropy,
        align_estimate_anisotropy=args.estimate_anisotropy,
        foldseek_bin=args.foldseek_bin,
        foldseek_prostt5=args.foldseek_prostt5,
        foldseek_sensitivity=args.foldseek_sensitivity,
        foldseek_evalue=args.foldseek_evalue,
        calibrate_method=method,
        calibrate_n=n_per,
    )
    result = run_search(args.query, args.seeds, cfg)
    write_tsv(
        result.hits,
        None if args.out == "-" else args.out,
        include_seq=not args.no_seq,
        include_calibration=bool(result.meta.get("calibrated")),
    )
    if not args.quiet:
        m = result.meta
        extra = ""
        if m.get("engine") == "foldseek":
            extra = f"score={m.get('score_type')}"
        else:
            extra = (f"backend={m.get('backend')} embedder={m.get('embedder')} "
                     f"score={m.get('score_type')}")
        print(
            f"[esmseek] engine={m.get('engine')} candidates={result.n_candidates} "
            f"seeds={result.n_seeds} hits={len(result.hits)} {extra}".rstrip(),
            file=sys.stderr,
        )
    return 0


def _cmd_embed(args: argparse.Namespace) -> int:
    import numpy as np

    records = parse_fasta(args.infile)
    proteins = [seed_to_protein(r, seq_type=args.seq_type) for r in records]
    embedder = get_embedder(
        backend=args.backend,
        model=args.model,
        device=args.device,
        forge_token=args.forge_token,
        forge_url=args.forge_url,
        hash_dim=args.hash_dim,
        hash_k=args.hash_k,
        cache_dir=args.cache_dir,
    )
    vecs = embedder.embed(proteins)
    np.save(f"{args.out}.npy", vecs)
    with open(f"{args.out}.ids.txt", "w") as fh:
        for r in records:
            fh.write(f"{r.id}\n")
    if not args.quiet:
        print(
            f"[esmseek] embedded {len(records)} sequences -> {args.out}.npy "
            f"(dim={vecs.shape[1] if vecs.size else embedder.dim})",
            file=sys.stderr,
        )
    return 0


def _macos_openmp_guard() -> None:
    """Avoid the duplicate-libomp abort that PyTorch hits on macOS.

    On macOS, PyTorch's bundled ``libomp`` frequently collides with another
    OpenMP runtime already loaded in the process (Homebrew's libomp, FAISS,
    etc.), aborting with "OMP: Error #15 ... libomp.dylib already initialized".
    We set the documented workaround *before* any Torch import, and pin a single
    OpenMP thread so the workaround stays numerically safe. Both use
    ``setdefault`` so an explicit user setting always wins. CLI-only: importing
    esmseek as a library never mutates the environment.
    """
    if sys.platform == "darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")


def main(argv: Optional[List[str]] = None) -> int:
    _macos_openmp_guard()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"esmseek: error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # engine failures (e.g. Foldseek) -> clean exit
        from .foldseek import FoldseekError

        if isinstance(exc, FoldseekError):
            print(f"esmseek: error: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
