"""R2 major 5, combined reading: does the DATA-CHOSEN grouping reproduce across cohorts?

    PYTHONPATH=. python3.9 scripts/groups_across_cohorts.py

validate_groups.py drops the fixed cluster count but stays in the analysis set;
validate_cohort.py crosses cohorts but compares the fixed-count grouping on both sides.
The reviewer asks for both at once. This reads the leakage matrices the cohort runs now
save, re-chooses K per cohort by silhouette, and compares cohort to analysis set with both
groupings chosen freely.

Writes results/validate_groups/across_cohorts.json.
"""
import json
import os
import pathlib
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.validate_groups import group_free_k  # noqa: E402
from supervdj import aggregate as A  # noqa: E402
from supervdj.resolution import family_of  # noqa: E402

OUT = pathlib.Path("results/validate_groups")
SRC = [("NSCLC", "TRA"), ("NSCLC", "TRB"), ("HNSCC", "TRB"), ("EMERSON", "TRB")]
DIRS = [pathlib.Path(d) for d in os.environ.get(
    "SUPERVDJ_LEAK_DIRS", "results/validation_cohort:results/tier5_emerson/tuple").split(":")]


def dist(M):
    S = (M + M.T) / 2.0
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return (D + D.T) / 2.0


def load(cohort, chain):
    for d in DIRS:
        f = d / f"leakage_{cohort}_{chain}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            return [str(g) for g in z["genes"]], z["M"]
    return None, None


def main():
    base = json.loads((OUT / "groups.json").read_text())   # analysis-set groupings
    rep = {}
    for cohort, chain in SRC:
        genes, M = load(cohort, chain)
        if genes is None:
            print(f"  {cohort} {chain}: no leakage matrix yet, skipped")
            continue
        D = dist(M)
        K_free, sil, lab_free = group_free_k(D, len(genes))
        K_fam = len(set(family_of(g) for g in genes))
        Z = linkage(squareform(D, checks=False), method="average")
        lab_fix = fcluster(Z, t=K_fam, criterion="maxclust")

        free = dict(zip(genes, lab_free))
        fix = dict(zip(genes, lab_fix))
        ms_free = base[chain]["unconstrained_grouping"]["labels"]
        ms_fix = base[chain]["manuscript_grouping"]["labels"]
        fam = {g: family_of(g) for g in genes}

        def ari(a, b):
            sh = [g for g in genes if g in a and g in b]
            return (round(float(A._adjusted_rand([a[g] for g in sh], [b[g] for g in sh])), 4),
                    len(sh))

        r = {
            "n_genes": len(genes), "K_data_chosen": int(K_free), "silhouette": round(sil, 4),
            "K_family_matched": int(K_fam),
            "free_vs_analysis_free": ari(free, ms_free),       # the combined test
            "fixed_vs_analysis_fixed": ari(fix, ms_fix),       # what the paper already reports
            "free_vs_family": ari(free, fam),
            "fixed_vs_family": ari(fix, fam),
        }
        rep[f"{cohort}_{chain}"] = r
        print(f"  {cohort:8s} {chain}  {len(genes):3d} genes  K: data {K_free:3d} "
              f"(sil {sil:.3f}) vs matched {K_fam:3d}")
        print(f"      data-chosen grouping vs analysis set's data-chosen: "
              f"ARI {r['free_vs_analysis_free'][0]:.3f} on {r['free_vs_analysis_free'][1]} shared genes")
        print(f"      fixed-count, for reference:                        "
              f"ARI {r['fixed_vs_analysis_fixed'][0]:.3f}")
        print(f"      vs IMGT families: data-chosen {r['free_vs_family'][0]:.3f}, "
              f"fixed {r['fixed_vs_family'][0]:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "across_cohorts.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(f"\nwrote {OUT / 'across_cohorts.json'}")


if __name__ == "__main__":
    main()
