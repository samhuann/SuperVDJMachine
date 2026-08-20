"""Export the data-driven V confusion groups and the calibration curve that lets a
candidate set be requested by *measured coverage* rather than nominal posterior mass.

    PYTHONPATH=. python scripts/export_confusion_groups.py [--out DIR]

Writes, per chain:
  confusion_groups_{TRA,TRB}.tsv   gene -> group id (leakage dendrogram cut at the
                                   number of IMGT families -- the same grouping the
                                   manuscript's confusion figures use), with the
                                   gene's mean posterior self-mass.
  calibration_{TRA,TRB}.tsv        nominal posterior mass -> empirical coverage and
                                   set size, measured on the canonical set.

Calibration is computed exactly, in one pass. Sort each sequence's pre-selection V
posterior descending and let c be the cumulative mass strictly *before* the annotated
gene. A set built to reach nominal mass p contains the annotated gene exactly when
c < p, so coverage(p) is the ECDF of c and the nominal mass needed for a target
coverage is a quantile of c. Set size at p is the number of genes until the
cumulative mass reaches p.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import supervdj.aggregate as A

NOMINAL = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 0.995, 0.999]
COVERAGE = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]


def per_sequence(posteriors, truths):
    """(mass before the true gene, sorted mass vector) per sequence."""
    before, sorted_masses = [], []
    for post, true in zip(posteriors, truths):
        items = sorted(post.items(), key=lambda kv: -kv[1])
        genes = [g for g, _ in items]
        masses = np.array([m for _, m in items])
        cum = np.cumsum(masses)
        if true in genes:
            i = genes.index(true)
            before.append(float(cum[i] - masses[i]))
        else:
            before.append(np.inf)          # annotated gene not a model candidate
        sorted_masses.append(cum)
    return np.array(before), sorted_masses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/confusion_groups"))
    ap.add_argument("--min-count", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import json
    df = A.load()
    for chain in ("TRA", "TRB"):
        genes, M = A.build_leakage(df, chain, args.min_count)
        labels = A._cluster_labels(genes, M)
        diag = dict(zip(genes, np.diag(M)))
        order = sorted(set(labels.values()),
                       key=lambda c: (-sum(v == c for v in labels.values()),
                                      min(g for g in genes if labels[g] == c)))
        remap = {c: i + 1 for i, c in enumerate(order)}
        gtab = pd.DataFrame([{"chain": chain, "gene": g, "group": remap[labels[g]],
                              "self_mass": round(float(diag[g]), 6)} for g in genes])
        gtab = gtab.sort_values(["group", "gene"])
        gtab.to_csv(args.out / f"confusion_groups_{chain}.tsv", sep="\t", index=False)
        sizes = gtab.group.value_counts()
        print(f"[{chain}] {len(genes)} genes (>= {args.min_count} seqs) -> {gtab.group.nunique()} "
              f"groups; largest {sizes.iloc[0]}, {(sizes == 1).sum()} singletons")

        s = A._slice(df, chain=chain, axis="V", model="pre", resolution="gene")
        before, cums = per_sequence((json.loads(p) for p in s["posterior_json"]),
                                    s["true_gene"].astype(str))
        rows = []
        for p in NOMINAL:
            size = np.array([int(np.searchsorted(c, p) + 1) for c in cums])
            rows.append({"chain": chain, "nominal_mass": p,
                         "empirical_coverage": round(float((before < p).mean()), 4),
                         "mean_set_size": round(float(size.mean()), 2),
                         "median_set_size": int(np.median(size))})
        cov_rows = []
        finite = before[np.isfinite(before)]
        for c in COVERAGE:
            # nominal mass whose measured coverage is c (quantile of the ECDF)
            q = float(np.quantile(finite, c * len(before) / len(finite))) if c * len(before) / len(finite) <= 1 else np.nan
            cov_rows.append({"chain": chain, "target_coverage": c,
                             "nominal_mass_needed": round(q, 4) if np.isfinite(q) else "unreachable"})
        cal = pd.DataFrame(rows)
        cal.to_csv(args.out / f"calibration_{chain}.tsv", sep="\t", index=False)
        pd.DataFrame(cov_rows).to_csv(args.out / f"coverage_targets_{chain}.tsv",
                                      sep="\t", index=False)
        print(cal.to_string(index=False))
        print(pd.DataFrame(cov_rows).to_string(index=False))
        miss = int(np.isinf(before).sum())
        print(f"  annotated gene absent from the model's candidate set: {miss:,} / {len(before):,} "
              f"({miss / len(before):.2%}) -- these can never be covered\n")

    print("wrote", args.out)


if __name__ == "__main__":
    main()
