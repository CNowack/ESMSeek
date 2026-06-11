#!/usr/bin/env python3
"""
Step — score how well each engine separates true LSRs from co-resident decoys.

Answers the only question that matters for the MGE filter: at a score threshold
that still catches most true LSRs, how many of the look-alike recombinase
families (especially resolvases, which share the catalytic serine domain) leak
through? Lower leakage at equal LSR recall = the better engine.

INPUTS
------
--labels labels.tsv      three columns, header required:
                            candidate_id   class   divergent
                         class is 'lsr' for true positives, anything else is a
                         decoy family name (e.g. resolvase / tyrosine / transposase).
                         divergent is True/False/NA (from split_manifest.tsv; only
                         meaningful for lsr rows).
--scores NAME=FILE ...    one or more engines. Each FILE is two columns:
                            candidate_id   score        (higher = more LSR-like)
                         Candidates absent from a score file are treated as the
                         worst possible score for that engine (no hit).
--target-recall R         LSR recall to hold fixed when comparing leakage (default 0.95).

OUTPUT
------
A per-engine table: AUROC, the threshold at the requested LSR recall, the decoy
pass-rate at that threshold (overall and per family), and the divergent-LSR
recall at that threshold.

Only dependency: numpy.
"""

from __future__ import annotations
import argparse, sys
import numpy as np


def read_labels(path):
    rows = {}
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        ci, cl, dv = header.index("candidate_id"), header.index("class"), header.index("divergent")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            rows[f[ci]] = (f[cl].strip().lower(), f[dv].strip())
    return rows  # id -> (class, divergent_str)


def read_scores(path):
    s = {}
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 2:
                continue
            try:
                s[f[0]] = float(f[1])
            except ValueError:
                continue  # skip header if present
    return s


def auroc(pos, neg):
    """P(random positive scores above random negative); ties = 0.5. Mann-Whitney."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(allv.size, float)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ranks within ties
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    R1 = ranks[:pos.size].sum()
    U1 = R1 - pos.size * (pos.size + 1) / 2.0
    return U1 / (pos.size * neg.size)


def evaluate(name, scores, labels, target_recall):
    worst = min(scores.values()) - 1.0 if scores else 0.0
    ids = list(labels.keys())
    sc = np.array([scores.get(i, worst) for i in ids])
    cls = np.array([labels[i][0] for i in ids])
    is_lsr = cls == "lsr"
    lsr_scores = sc[is_lsr]
    dec_scores = sc[~is_lsr]

    A = auroc(lsr_scores, dec_scores)

    # threshold that retains `target_recall` of LSRs:
    # keep score >= threshold; choose the (1-recall) lower quantile of LSR scores.
    thr = np.quantile(lsr_scores, 1.0 - target_recall, method="lower")
    realized_recall = float((lsr_scores >= thr).mean())

    passed = sc >= thr
    overall_decoy_pass = float(passed[~is_lsr].mean()) if (~is_lsr).any() else float("nan")

    # per decoy family
    fam_pass = {}
    for fam in sorted(set(cls[~is_lsr])):
        m = cls == fam
        fam_pass[fam] = float(passed[m].mean())

    # divergent-LSR recall at this threshold
    divergent = np.array([labels[i][1].lower() == "true" for i in ids])
    div_mask = is_lsr & divergent
    div_recall = float(passed[div_mask].mean()) if div_mask.any() else float("nan")
    n_div = int(div_mask.sum())

    return {
        "name": name, "auroc": A, "thr": float(thr),
        "lsr_recall": realized_recall, "decoy_pass": overall_decoy_pass,
        "fam_pass": fam_pass, "div_recall": div_recall, "n_div": n_div,
        "n_lsr": int(is_lsr.sum()), "n_decoy": int((~is_lsr).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--scores", nargs="+", required=True,
                    help="NAME=FILE pairs, e.g. esmc=esmc.score.tsv foldseek=fold.score.tsv")
    ap.add_argument("--target-recall", type=float, default=0.95)
    args = ap.parse_args()

    labels = read_labels(args.labels)
    engines = []
    for spec in args.scores:
        if "=" not in spec:
            sys.exit(f"--scores expects NAME=FILE, got {spec!r}")
        name, path = spec.split("=", 1)
        engines.append(evaluate(name, read_scores(path), labels, args.target_recall))

    n_lsr = engines[0]["n_lsr"]; n_dec = engines[0]["n_decoy"]
    print(f"\npool: {n_lsr} LSR positives ({engines[0]['n_div']} divergent), "
          f"{n_dec} decoys | LSR recall held at {args.target_recall:.0%}\n")
    fams = sorted({f for e in engines for f in e["fam_pass"]})
    header = ["engine", "AUROC", "thr", "LSR_recall", "decoy_pass"] + [f"{f}_pass" for f in fams] + ["divergent_recall"]
    print("\t".join(header))
    for e in engines:
        row = [e["name"], f"{e['auroc']:.3f}", f"{e['thr']:.3g}",
               f"{e['lsr_recall']:.2f}", f"{e['decoy_pass']:.2f}"]
        row += [f"{e['fam_pass'].get(f, float('nan')):.2f}" for f in fams]
        row += [f"{e['div_recall']:.2f}"]
        print("\t".join(row))

    print("\nread: higher AUROC is better; at matched LSR recall, LOWER decoy_pass "
          "is better (especially resolvase). Watch divergent_recall — an engine that\n"
          "drops the divergent LSRs is failing the case you built this for.")
    if n_lsr < 30 or engines[0]["n_div"] < 10:
        print(f"\n[warn] small sample (LSR={n_lsr}, divergent={engines[0]['n_div']}): "
              "treat a small AUROC gap as a tie; bootstrap CIs before trusting a winner.")


if __name__ == "__main__":
    main()