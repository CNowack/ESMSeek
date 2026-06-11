#!/usr/bin/env python3
"""
Build the candidate pool and its hidden answer key for the discrimination test.

Combines your test-positive LSRs with the decoy families into one FASTA, and
writes a labels.tsv that score_discrimination.py reads. LSR rows carry their
divergent flag (looked up from split_manifest.tsv); decoys are tagged by family.

USAGE
-----
    python build_pool.py \
        --lsr split/test_positives.faa \
        --manifest split/split_manifest.tsv \
        --decoy resolvase=res_rep_rep_seq.fasta \
        --decoy tyrosine=tyr_rep_rep_seq.fasta \
        --decoy transposase=tnp_rep_rep_seq.fasta \
        --out-fasta candidates.faa \
        --out-labels labels.tsv

Decoy sequence ids are prefixed with the family name (e.g. resolvase__P12345)
so every id in the pool is unique and traceable. Only dependency: none.
"""

from __future__ import annotations
import argparse, sys


def read_fasta(path):
    recs, rid, desc, seq = [], None, "", []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(">"):
                if rid is not None:
                    recs.append((rid, desc, "".join(seq)))
                head = line[1:].split(None, 1)
                rid, desc, seq = head[0], (head[1] if len(head) > 1 else ""), []
            else:
                seq.append(line.strip())
    if rid is not None:
        recs.append((rid, desc, "".join(seq)))
    return recs


def read_manifest(path):
    """id -> divergent string ('True'/'False'/'NA')."""
    div = {}
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        ci, di = header.index("lsr_id"), header.index("divergent")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            div[f[ci]] = f[di]
    return div


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lsr", required=True, help="FASTA of LSR test positives")
    ap.add_argument("--manifest", required=True, help="split_manifest.tsv")
    ap.add_argument("--decoy", action="append", default=[],
                    help="family=FASTA, repeatable (e.g. resolvase=res_rep_rep_seq.fasta)")
    ap.add_argument("--out-fasta", required=True)
    ap.add_argument("--out-labels", required=True)
    args = ap.parse_args()

    div = read_manifest(args.manifest)

    pool = []     # (id, seq)
    labels = []   # (id, class, divergent)
    counts = {}

    # LSR positives
    n_div = 0
    for rid, _desc, seq in read_fasta(args.lsr):
        d = div.get(rid, "NA")
        labels.append((rid, "lsr", d))
        pool.append((rid, seq))
        n_div += int(d.lower() == "true")
    counts["lsr"] = len(pool)

    # decoys
    seen = {rid for rid, _ in pool}
    for spec in args.decoy:
        if "=" not in spec:
            sys.exit(f"--decoy expects family=FASTA, got {spec!r}")
        fam, path = spec.split("=", 1)
        n = 0
        for rid, _desc, seq in read_fasta(path):
            new_id = f"{fam}__{rid}"
            if new_id in seen:
                continue
            seen.add(new_id)
            labels.append((new_id, fam, "NA"))
            pool.append((new_id, seq))
            n += 1
        counts[fam] = n

    # write pool fasta
    with open(args.out_fasta, "w") as fh:
        for rid, seq in pool:
            fh.write(f">{rid}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i+60] + "\n")

    # write labels
    with open(args.out_labels, "w") as fh:
        fh.write("candidate_id\tclass\tdivergent\n")
        for rid, cls, d in labels:
            fh.write(f"{rid}\t{cls}\t{d}\n")

    print(f"[done] pool = {len(pool)} sequences -> {args.out_fasta}")
    print(f"        labels -> {args.out_labels}")
    summary = "  ".join(f"{k}={v}" for k, v in counts.items())
    print(f"        {summary}  (lsr divergent={n_div})")
    if counts["lsr"] == 0:
        print("[warn] no LSR positives found — check --lsr path")
    if sum(v for k, v in counts.items() if k != "lsr") == 0:
        print("[warn] no decoys added — check --decoy paths")


if __name__ == "__main__":
    main()