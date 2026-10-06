"""R2 major concern 5: validate the V confusion groups without the imposed cluster count.

    PYTHONPATH=. python scripts/validate_groups.py [--replicates 200]

Two tests, both asked for in the comment:

  1. Stability WITHOUT the constraint. The manuscript cuts the leakage dendrogram at
     K = number of /DV-preserving V families, and reports bootstrap stability at that
     fixed K (mean ARI 0.944 for alpha). This repeats the same bootstrap with K chosen
     from the data in every replicate, so nothing is held fixed.

  2. Across independent repertoires. Handled by scripts/validate_cohort.py, which
     re-derives the grouping in each held-out tumour cohort and scores it against the
     analysis-set grouping; this script just reports those numbers alongside.

Writes results/validate_groups/groups.json.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supervdj.resolution import family_of
from supervdj.aggregate import _adjusted_rand

PATH = Path("results/posteriors.tsv")
OUT = Path("results/validate_groups")
MIN_COUNT = 20


def load_chain(chain):
    """One streaming pass -> (genes, true_gene index per sequence, posterior matrix).
    Same slice as supervdj.aggregate.build_leakage: V axis, pre-selection, gene
    resolution, status ok."""
    rows = []
    with PATH.open(newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if (r["chain"] != chain or r["axis"] != "V" or r["model"] != "pre"
                    or r["resolution"] != "gene" or r["status"] != "ok"
                    or (r["mode"] or "") != ""):
                continue
            rows.append((r["true_gene"], r["posterior_json"]))
    from collections import Counter
    cnt = Counter(g for g, _ in rows)
    genes = sorted(g for g, c in cnt.items() if c >= MIN_COUNT)
    idx = {g: i for i, g in enumerate(genes)}
    sel = [(idx[g], pj) for g, pj in rows if g in idx]
    true = np.fromiter((i for i, _ in sel), dtype=np.int32, count=len(sel))
    P = np.zeros((len(sel), len(genes)), dtype=np.float32)
    for k, (_, pj) in enumerate(sel):
        for g, m in json.loads(pj).items():
            j = idx.get(g)
            if j is not None:
                P[k, j] = m
    return genes, true, P


def leakage_from(true, P, n_genes):
    M = np.zeros((n_genes, n_genes))
    cnt = np.bincount(true, minlength=n_genes).astype(float)
    np.add.at(M, true, P)
    return M / np.maximum(cnt, 1)[:, None]


def distance(M):
    S = (M + M.T) / 2.0
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return (D + D.T) / 2.0


def silhouette(D, lab):
    """Mean silhouette from a precomputed distance matrix; singletons score 0."""
    lab = np.asarray(lab)
    out = np.zeros(len(lab))
    others = set(lab)
    for i in range(len(lab)):
        same = lab == lab[i]
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, lab == c].mean() for c in others - {lab[i]})
        out[i] = (b - a) / max(a, b)
    return float(out.mean())


def _selfcheck():
    D = np.array([[0, .1, .9, 1.], [.1, 0, 1., .9], [.9, 1., 0, .1], [1., .9, .1, 0]])
    assert silhouette(D, [0, 0, 1, 1]) > 0.8
    assert silhouette(D, [0, 1, 0, 1]) < silhouette(D, [0, 0, 1, 1])
    M = np.array([[.8, .2], [.3, .7]])
    assert abs(leakage_from(np.array([0, 1]), M, 2) - M).max() < 1e-9


def group_free_k(D, n_genes):
    """Cut the dendrogram at the K the data prefers (best mean silhouette)."""
    Z = linkage(squareform(D, checks=False), method="average")
    best = None
    for K in range(2, n_genes):
        lab = fcluster(Z, t=K, criterion="maxclust")
        if len(set(lab)) < 2:
            continue
        s = silhouette(D, lab)
        if best is None or s > best[1]:
            best = (K, s, lab)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    _selfcheck()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    report = {}

    for chain in ("TRA", "TRB"):
        genes, true, P = load_chain(chain)
        n = len(genes)
        fam = [family_of(g) for g in genes]
        K_fam = len(set(fam))
        D_full = distance(leakage_from(true, P, n))
        Z_full = linkage(squareform(D_full, checks=False), method="average")
        fixed_lab = fcluster(Z_full, t=K_fam, criterion="maxclust")      # manuscript grouping
        K_free, sil_free, free_lab = group_free_k(D_full, n)

        # test 1: bootstrap with K re-chosen inside every replicate
        aris = []
        for _ in range(args.replicates):
            take = rng.integers(0, len(true), len(true))
            Db = distance(leakage_from(true[take], P[take], n))
            _, _, lab_b = group_free_k(Db, n)
            aris.append(_adjusted_rand(list(free_lab), list(lab_b)))
        aris = np.array(aris)

        report[chain] = {
            "n_genes": n, "n_sequences": int(len(true)), "K_families": K_fam,
            "manuscript_grouping": {
                "K": K_fam, "silhouette": silhouette(D_full, fixed_lab),
                "ari_vs_family": float(_adjusted_rand(fam, list(fixed_lab))),
                "labels": {g: int(l) for g, l in zip(genes, fixed_lab)}},
            "unconstrained_grouping": {
                "K": int(K_free), "silhouette": float(sil_free),
                "ari_vs_family": float(_adjusted_rand(fam, list(free_lab))),
                "ari_vs_manuscript_grouping": float(_adjusted_rand(list(fixed_lab), list(free_lab))),
                "labels": {g: int(l) for g, l in zip(genes, free_lab)}},
            "bootstrap_stability_K_free": {
                "replicates": int(args.replicates),
                "mean_ari": float(aris.mean()),
                "ci95": [float(np.quantile(aris, 0.025)), float(np.quantile(aris, 0.975))]},
        }
        r = report[chain]
        print(f"[{chain}] {n} V genes, {len(true):,} sequences, {K_fam} IMGT families")
        print(f"  manuscript grouping   K={K_fam:3d}  ARI vs family {r['manuscript_grouping']['ari_vs_family']:.4f}")
        print(f"  unconstrained         K={K_free:3d}  ARI vs family {r['unconstrained_grouping']['ari_vs_family']:.4f}"
              f"   ARI vs manuscript grouping {r['unconstrained_grouping']['ari_vs_manuscript_grouping']:.4f}")
        b = r["bootstrap_stability_K_free"]
        print(f"  bootstrap stability with K free: mean ARI {b['mean_ari']:.3f} "
              f"[{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}] over {args.replicates} replicates")

    (OUT / "groups.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT/'groups.json'}")


if __name__ == "__main__":
    main()
