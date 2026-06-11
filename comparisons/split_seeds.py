#!/usr/bin/env python3
"""
Step 2 — cluster the validated LSRs and split them into seeds vs test positives,
flagging the experimentally meaningful "divergent" subset.

WHAT IT DOES
------------
1. Clusters all_lsrs.faa by sequence identity (MMseqs2).
2. Picks SEED clusters and uses their representatives as the query set (seeds.faa).
   Everything in the *other* clusters becomes test positives (test_positives.faa).
   Splitting by whole clusters is the point: a test sequence that lives in a
   non-seed cluster is, by construction, below the clustering identity to every
   seed -> a real remote-homology test, not a near-duplicate the search would
   trivially recover.
3. Measures each test positive's best identity to ANY seed and flags it
   `divergent` when that best identity is below DIVERGENT_MAX_ID. Those are the
   needles the whole experiment is built to find.

LEAKAGE WARNING (read this)
---------------------------
If seeds were just "one representative per cluster" for ALL clusters, every test
sequence would sit in a seed's own cluster and therefore be >=CLUSTER_ID similar
to a seed -> no divergent test set, experiment broken. This script avoids that by
holding out whole clusters as test-only.

REQUIREMENTS
------------
- MMseqs2 on PATH  (conda install -c bioconda mmseqs2)
- Python 3.9+, no other deps.

USAGE
-----
    python split_seeds.py all_lsrs.faa --outdir split/
    # or pin canonical seeds by id (one lsr_id per line), all else becomes test:
    python split_seeds.py all_lsrs.faa --outdir split/ --seed-ids canonical_seeds.txt
"""

from __future__ import annotations
import argparse, os, subprocess, sys
from collections import defaultdict

# ---- knobs ----------------------------------------------------------------
CLUSTER_ID       = 0.40   # identity for clustering (whole-cluster holdout boundary)
COVERAGE         = 0.70   # bidirectional coverage for clustering
TARGET_SEEDS     = 15     # how many seed clusters to designate (ignored if --seed-ids)
DIVERGENT_MAX_ID = 30.0   # a test positive is "divergent" if best %id to any seed < this
# ---------------------------------------------------------------------------


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


def write_fasta(recs, path):
    with open(path, "w") as fh:
        for rid, desc, seq in recs:
            fh.write(f">{rid}{(' ' + desc) if desc else ''}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i+60] + "\n")


def run(cmd):
    print("[run]", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cluster(infile, outdir):
    prefix = os.path.join(outdir, "clust")
    tmp = os.path.join(outdir, "tmp_clust")
    run(["mmseqs", "easy-cluster", infile, prefix, tmp,
         "--min-seq-id", str(CLUSTER_ID), "-c", str(COVERAGE)])
    # clust_cluster.tsv: representative_id <TAB> member_id
    clusters = defaultdict(list)
    with open(prefix + "_cluster.tsv") as fh:
        for line in fh:
            rep, mem = line.rstrip("\n").split("\t")
            clusters[rep].append(mem)
    return clusters  # {rep_id: [member_ids...]}


def best_identity_to_seeds(test_path, seeds_path, outdir):
    """Return {test_id: best_percent_identity_to_any_seed} (0 if no hit)."""
    out = os.path.join(outdir, "test_vs_seeds.tsv")
    tmp = os.path.join(outdir, "tmp_search")
    run(["mmseqs", "easy-search", test_path, seeds_path, out, tmp,
         "--format-output", "query,target,pident", "-s", "7.5"])
    best = {}
    raw = []
    with open(out) as fh:
        for line in fh:
            q, _, pid = line.rstrip("\n").split("\t")[:3]
            raw.append((q, float(pid)))
    # MMseqs reports pident as a fraction (0-1) in some versions, percent in others.
    scale = 100.0 if (raw and max(p for _, p in raw) <= 1.0) else 1.0
    for q, pid in raw:
        pid *= scale
        if pid > best.get(q, -1):
            best[q] = pid
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta", help="all_lsrs.faa")
    ap.add_argument("--outdir", default="split")
    ap.add_argument("--seed-ids", default=None,
                    help="optional file of lsr_ids (one per line) to force as seeds")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    recs = read_fasta(args.fasta)
    by_id = {rid: (rid, desc, seq) for rid, desc, seq in recs}
    print(f"[info] {len(recs)} input sequences", file=sys.stderr)

    # ---- choose seed ids ---------------------------------------------------
    if args.seed_ids:
        seed_ids = {ln.strip() for ln in open(args.seed_ids) if ln.strip()}
        missing = seed_ids - set(by_id)
        if missing:
            sys.exit(f"[error] seed ids not in fasta: {sorted(missing)}")
        source = "explicit --seed-ids"
        cluster_of = {rid: "n/a" for rid in by_id}
    else:
        clusters = cluster(args.fasta, args.outdir)
        cluster_of = {m: rep for rep, members in clusters.items() for m in members}
        n_clusters = len(clusters)
        print(f"[info] {n_clusters} clusters at {int(CLUSTER_ID*100)}% identity",
              file=sys.stderr)
        if n_clusters <= TARGET_SEEDS:
            print(f"[warn] only {n_clusters} clusters; set is too redundant to leave a "
                  f"divergent test set. Lower CLUSTER_ID or add more positives.",
                  file=sys.stderr)
        # Designate seed clusters = the largest TARGET_SEEDS clusters (well-populated
        # 'core' subfamilies). Their reps are the seeds; all OTHER clusters are
        # held out entirely as (divergent) test positives. Tune as needed; or use
        # --seed-ids to pin canonical integrases instead.
        ordered = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
        seed_clusters = dict(ordered[:TARGET_SEEDS])
        seed_ids = set(seed_clusters.keys())  # representatives
        source = f"largest {len(seed_ids)} of {n_clusters} clusters"

    # ---- assign roles ------------------------------------------------------
    # seed                : the query sequences
    # positive_test       : held-out cluster members (the remote-homology needles)
    # positive_easy       : non-rep members of a seed's own cluster (near-relatives;
    #                       kept but labeled so they never contaminate the divergent metric)
    roles = {}
    for rid in by_id:
        if rid in seed_ids:
            roles[rid] = "seed"
        elif cluster_of.get(rid, "n/a") in seed_ids:
            roles[rid] = "positive_easy"
        else:
            roles[rid] = "positive_test"

    seeds   = [by_id[r] for r in by_id if roles[r] == "seed"]
    tests   = [by_id[r] for r in by_id if roles[r] in ("positive_test", "positive_easy")]
    write_fasta(seeds, os.path.join(args.outdir, "seeds.faa"))
    write_fasta(tests, os.path.join(args.outdir, "test_positives.faa"))

    # ---- divergence of each test positive vs seeds -------------------------
    best = best_identity_to_seeds(
        os.path.join(args.outdir, "test_positives.faa"),
        os.path.join(args.outdir, "seeds.faa"),
        args.outdir,
    )

    # ---- manifest ----------------------------------------------------------
    man = os.path.join(args.outdir, "split_manifest.tsv")
    n_div = 0
    with open(man, "w") as fh:
        fh.write("lsr_id\trole\tsource_cluster\tbest_pident_to_seed\tdivergent\n")
        for rid in by_id:
            role = roles[rid]
            if role == "seed":
                fh.write(f"{rid}\tseed\t{cluster_of.get(rid,'n/a')}\tNA\tNA\n")
                continue
            pid = best.get(rid, 0.0)               # no hit => effectively very divergent
            div = (role == "positive_test") and (pid < DIVERGENT_MAX_ID)
            n_div += int(div)
            fh.write(f"{rid}\t{role}\t{cluster_of.get(rid,'n/a')}\t{pid:.1f}\t{div}\n")

    n_seed = sum(1 for r in roles.values() if r == "seed")
    n_test = sum(1 for r in roles.values() if r == "positive_test")
    n_easy = sum(1 for r in roles.values() if r == "positive_easy")
    print(f"\n[done] seed source: {source}", file=sys.stderr)
    print(f"  seeds={n_seed}  test_positive={n_test} (divergent<{int(DIVERGENT_MAX_ID)}%={n_div})  "
          f"positive_easy={n_easy}", file=sys.stderr)
    print(f"  wrote seeds.faa, test_positives.faa, split_manifest.tsv -> {args.outdir}/",
          file=sys.stderr)
    if n_div < 10:
        print(f"  [warn] only {n_div} divergent test positives -> low statistical power; "
              f"consider growing the positive set or lowering CLUSTER_ID.", file=sys.stderr)


if __name__ == "__main__":
    main()