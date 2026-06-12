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


def _metrics(sc, cls, is_lsr, divergent, target_recall):
    """Compute the discrimination metrics for one engine on aligned arrays.

    ``sc``/``cls``/``is_lsr``/``divergent`` are parallel arrays over the same set
    of candidates (in the same order). Returns a dict of scalar metrics plus the
    per-family pass-rate dict. Pulled out of :func:`evaluate` so the bootstrap can
    call it on resampled index sets.
    """
    dec_mask = ~is_lsr
    lsr_scores = sc[is_lsr]
    A = auroc(lsr_scores, sc[dec_mask])

    # threshold that retains `target_recall` of LSRs:
    # keep score >= threshold; choose the (1-recall) lower quantile of LSR scores.
    thr = np.quantile(lsr_scores, 1.0 - target_recall, method="lower")
    realized_recall = float((lsr_scores >= thr).mean())

    passed = sc >= thr
    overall_decoy_pass = float(passed[dec_mask].mean()) if dec_mask.any() else float("nan")

    fam_pass = {}
    for fam in sorted(set(cls[dec_mask])):
        m = cls == fam
        fam_pass[fam] = float(passed[m].mean())

    div_mask = is_lsr & divergent
    div_recall = float(passed[div_mask].mean()) if div_mask.any() else float("nan")

    return {
        "auroc": A, "thr": float(thr), "lsr_recall": realized_recall,
        "decoy_pass": overall_decoy_pass, "fam_pass": fam_pass, "div_recall": div_recall,
    }


def _engine_arrays(scores, labels):
    """Build the per-candidate arrays for one engine, aligned to ``labels`` order.

    Candidates missing from a score file get that engine's worst score (no hit).
    """
    worst = min(scores.values()) - 1.0 if scores else 0.0
    ids = list(labels.keys())
    sc = np.array([scores.get(i, worst) for i in ids], dtype=float)
    cls = np.array([labels[i][0] for i in ids])
    is_lsr = cls == "lsr"
    divergent = np.array([labels[i][1].lower() == "true" for i in ids])
    return sc, cls, is_lsr, divergent


def evaluate(name, scores, labels, target_recall):
    sc, cls, is_lsr, divergent = _engine_arrays(scores, labels)
    m = _metrics(sc, cls, is_lsr, divergent, target_recall)
    m.update({
        "name": name,
        "n_div": int((is_lsr & divergent).sum()),
        "n_lsr": int(is_lsr.sum()), "n_decoy": int((~is_lsr).sum()),
    })
    return m


def _stratified_resamples(cls, n_boot, rng):
    """Yield ``n_boot`` index arrays, each resampling *within* every class with
    replacement so the LSR / per-decoy-family counts are preserved (the test's
    design is fixed; only sampling noise is bootstrapped)."""
    groups = [np.where(cls == c)[0] for c in sorted(set(cls.tolist()))]
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])


def _ci(values, lo=2.5, hi=97.5):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def bootstrap(engine_data, fams, target_recall, n_boot, seed):
    """Shared-resample bootstrap across all engines.

    The SAME stratified resample is applied to every engine each iteration, so
    per-engine CIs and paired engine-vs-engine differences are both valid (the
    pairing controls for which candidates happen to be drawn).

    Returns ``(per_engine_ci, paired_ci)``:
      * ``per_engine_ci[name][metric]   -> (lo, hi)``
      * ``paired_ci[name][metric]       -> (delta, lo, hi)``  (name minus baseline)
    """
    names = [d["name"] for d in engine_data]
    # any engine shares the same cls/is_lsr/divergent ordering
    cls = engine_data[0]["cls"]
    metric_keys = ["auroc", "decoy_pass", "div_recall"] + [f"{f}_pass" for f in fams]
    dist = {name: {k: [] for k in metric_keys} for name in names}

    rng = np.random.default_rng(seed)
    for idx in _stratified_resamples(cls, n_boot, rng):
        for d in engine_data:
            m = _metrics(d["sc"][idx], cls[idx], d["is_lsr"][idx], d["divergent"][idx],
                         target_recall)
            dist[d["name"]]["auroc"].append(m["auroc"])
            dist[d["name"]]["decoy_pass"].append(m["decoy_pass"])
            dist[d["name"]]["div_recall"].append(m["div_recall"])
            for f in fams:
                dist[d["name"]][f"{f}_pass"].append(m["fam_pass"].get(f, float("nan")))

    per_engine_ci = {name: {k: _ci(dist[name][k]) for k in metric_keys} for name in names}

    baseline = names[0]
    paired_ci = {}
    for name in names[1:]:
        paired_ci[name] = {}
        for k in ["decoy_pass", "div_recall", "auroc"] + [f"{f}_pass" for f in fams]:
            diff = np.asarray(dist[name][k], float) - np.asarray(dist[baseline][k], float)
            diff = diff[~np.isnan(diff)]
            lo, hi = _ci(diff)
            paired_ci[name][k] = (float(np.mean(diff)) if diff.size else float("nan"), lo, hi)
    return per_engine_ci, paired_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--scores", nargs="+", required=True,
                    help="NAME=FILE pairs, e.g. esmc=esmc.score.tsv foldseek=fold.score.tsv")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N",
                    help="Bootstrap resamples for 95%% CIs (e.g. 2000). 0 = off.")
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    args = ap.parse_args()

    labels = read_labels(args.labels)
    engines = []
    engine_data = []  # raw arrays kept for the bootstrap
    for spec in args.scores:
        if "=" not in spec:
            sys.exit(f"--scores expects NAME=FILE, got {spec!r}")
        name, path = spec.split("=", 1)
        scores = read_scores(path)
        engines.append(evaluate(name, scores, labels, args.target_recall))
        sc, cls, is_lsr, divergent = _engine_arrays(scores, labels)
        engine_data.append({"name": name, "sc": sc, "cls": cls,
                            "is_lsr": is_lsr, "divergent": divergent})

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

    if args.bootstrap > 0:
        per_engine_ci, paired_ci = bootstrap(
            engine_data, fams, args.target_recall, args.bootstrap, args.bootstrap_seed)
        ci_cols = ["AUROC", "decoy_pass"] + [f"{f}_pass" for f in fams] + ["divergent_recall"]
        key_of = {"AUROC": "auroc", "decoy_pass": "decoy_pass",
                  "divergent_recall": "div_recall",
                  **{f"{f}_pass": f"{f}_pass" for f in fams}}
        print(f"\n95% CI ({args.bootstrap} stratified resamples), shown as lo–hi:")
        print("\t".join(["engine"] + ci_cols))
        for e in engines:
            row = [e["name"]]
            for c in ci_cols:
                lo, hi = per_engine_ci[e["name"]][key_of[c]]
                row.append(f"{lo:.2f}–{hi:.2f}")
            print("\t".join(row))

        baseline = engines[0]["name"]
        print(f"\npaired difference vs '{baseline}' (Δ = engine − {baseline}; "
              f"95% CI; * = CI excludes 0):")
        print("\t".join(["engine", "Δdecoy_pass"] + [f"Δ{f}_pass" for f in fams]
                        + ["Δdivergent_recall", "ΔAUROC"]))
        for e in engines[1:]:
            cells = [e["name"]]
            for k in ["decoy_pass"] + [f"{f}_pass" for f in fams] + ["div_recall", "auroc"]:
                delta, lo, hi = paired_ci[e["name"]][k]
                star = "*" if (lo > 0 or hi < 0) else ""
                cells.append(f"{delta:+.2f} [{lo:+.2f},{hi:+.2f}]{star}")
            print("\t".join(cells))
        print("\nread: for Δdecoy_pass / Δresolvase_pass, NEGATIVE means the engine "
              "leaks fewer decoys than the baseline; for Δdivergent_recall, POSITIVE means\n"
              "it catches more of the divergent LSRs. A '*' (CI excludes 0) means the "
              "difference is unlikely to be sampling noise. List the baseline engine FIRST.")
    elif n_lsr < 30 or engines[0]["n_div"] < 10:
        print(f"\n[warn] small sample (LSR={n_lsr}, divergent={engines[0]['n_div']}): "
              "treat a small AUROC gap as a tie; add --bootstrap 2000 for CIs.")


if __name__ == "__main__":
    main()