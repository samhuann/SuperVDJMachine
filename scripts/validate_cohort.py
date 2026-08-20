"""Replicate the recoverability measurements on held-out tumour cohorts.

    PYTHONPATH=. python scripts/validate_cohort.py [--n 10000] [--workers 14]

Two cohorts that contributed nothing to the analysis set are put through the same
pipeline and the headline quantities recomputed:

  NSCLC   Liu et al., Cell 2025 (GEO GSE243013): 434,458 single cells, 231 patients,
          paired alpha/beta with V and J annotations.
  HNSCC   Cha et al., Cell Rep Med 2025 (GEO GSE286827): 14,401 beta clones, 15 patients.

Each cohort is normalized with the same gene reconciliation and CDR3 boundary rules
as the analysis set, deduplicated on the (cdr3_aa, v_gene, j_gene, chain) tuple, and
then stripped of every tuple that appears in the canonical set, so what remains is
genuinely held out. A fixed-seed subsample keeps the OLGA pass tractable.

Writes results/validation_cohort/{cohort_chain}_metrics.tsv and summary.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

import supervdj.aggregate as A
from supervdj.resolution import family_of

# Where the two GEO cohorts live. Defaults to data/cohorts/ inside the repo; set
# SUPERVDJ_COHORT_DIR to point elsewhere. Expected layout:
#   <dir>/data-nsclc/tcr.csv.gz        GSE243013, paired alpha/beta single-cell TCRs
#   <dir>/data-hnscc2/hnscc2_clones.csv GSE286827, beta clones
BASE = Path(os.environ.get("SUPERVDJ_COHORT_DIR", "data/cohorts"))
OUT = Path("results/validation_cohort")
MIN_COUNT = 20
_M = {}


def load_cohorts():
    """-> {(cohort, chain): DataFrame[cdr3_raw, v_raw, j_raw, chain]}"""
    nsclc = pd.read_csv(BASE / "data-nsclc/tcr.csv.gz")
    hn = pd.read_csv(BASE / "data-hnscc2/hnscc2_clones.csv")
    out = {}
    for chain, pre in (("TRA", "TRA"), ("TRB", "TRB")):
        d = nsclc[[f"{pre}_cdr3", f"{pre}_v_gene", f"{pre}_j_gene"]].dropna()
        d.columns = ["cdr3_raw", "v_raw", "j_raw"]
        d = d.assign(chain=chain)
        out[("NSCLC", chain)] = d
    hnb = hn[["cdr3", "v", "j"]].dropna()
    hnb.columns = ["cdr3_raw", "v_raw", "j_raw"]
    out[("HNSCC", "TRB")] = hnb.assign(chain="TRB")
    return out


def normalize(df, source, chain):
    """Same normalization the analysis set went through."""
    from ingest.gene_names import GeneReconciler
    from ingest.imgt_boundaries import ImgtBoundaries
    from ingest.normalize import normalize_rows

    rec = GeneReconciler.from_olga()
    bnd = ImgtBoundaries.from_dir(Path("data/imgt"))
    kept, rejected = normalize_rows(df.to_dict("records"), source, rec, bnd)
    kept = kept.drop_duplicates(subset=["cdr3_aa", "v_gene", "j_gene", "chain"])
    return kept, len(rejected)


def canonical_tuples(chain):
    if chain == "TRA":
        d = pd.read_csv("results/ingest/canonical_alpha_pooled.tsv", sep="\t")
    else:
        d = pd.read_csv("results/ingest/canonical_unique.tsv", sep="\t")
        d = d[d.chain == "TRB"]
    return set(map(tuple, d[["cdr3_aa", "v_gene", "j_gene"]].values))


def _init(chain):
    from supervdj.cache import ValueCache
    from supervdj.models import load_chain_models
    _M["models"] = load_chain_models(chain, use_sonia=False)
    _M["cache"] = ValueCache(None)


def _posteriors(cdr3):
    """(V posterior, J posterior) for one CDR3, or None if OLGA calls it impossible."""
    from supervdj.posterior import preselection_posterior
    v = preselection_posterior(_M["models"], _M["cache"], cdr3, "V")
    if not v or sum(v.values()) == 0:
        return None
    j = preselection_posterior(_M["models"], _M["cache"], cdr3, "J")
    return v, j


def entropy(p):
    return -sum(m * np.log(m) for m in p.values() if m > 0)


def rank_of(p, gene):
    ranked = sorted(p.items(), key=lambda kv: -kv[1])
    for i, (g, _) in enumerate(ranked, 1):
        if g == gene:
            return i
    return None


def leakage(genes_true, posts, min_count=MIN_COUNT):
    counts = pd.Series(genes_true).value_counts()
    keep = sorted(counts[counts >= min_count].index)
    idx = {g: i for i, g in enumerate(keep)}
    M = np.zeros((len(keep), len(keep)))
    n = np.zeros(len(keep))
    for tg, p in zip(genes_true, posts):
        i = idx.get(tg)
        if i is None:
            continue
        n[i] += 1
        for g, m in p.items():
            j = idx.get(g)
            if j is not None:
                M[i, j] += m
    return keep, M / np.maximum(n, 1)[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="subsample per cohort-chain")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    paper_groups = {c: pd.read_csv(f"results/confusion_groups/confusion_groups_{c}.tsv", sep="\t")
                    for c in ("TRA", "TRB")}
    rows, summary = [], {}
    for (cohort, chain), raw in sorted(load_cohorts().items()):
        kept, n_rejected = normalize(raw, cohort.lower(), chain)
        canon = canonical_tuples(chain)
        held = kept[~kept.apply(lambda r: (r.cdr3_aa, r.v_gene, r.j_gene) in canon, axis=1)]
        sub = held.sample(n=min(args.n, len(held)), random_state=args.seed).reset_index(drop=True)
        print(f"[{cohort} {chain}] {len(raw):,} rows -> {len(kept):,} unique -> {len(held):,} held out "
              f"-> {len(sub):,} sampled ({n_rejected:,} rejected in normalization)")

        with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(chain,)) as pool:
            res = pool.map(_posteriors, list(sub.cdr3_aa), chunksize=16)
        ok = [(r, v, j) for r, (v, j) in ((row, x) for row, x in zip(sub.itertuples(), res) if x)]
        n_zero = len(sub) - len(ok)

        vH = [entropy(v) for _, v, _ in ok]
        jH = [entropy(j) for _, _, j in ok]
        vrank = [rank_of(v, r.v_gene) for r, v, _ in ok]
        vtop1 = [max(v.values()) for _, v, _ in ok]
        keep, M = leakage([r.v_gene for r, _, _ in ok], [v for _, v, _ in ok])

        # grouping agreement with the paper's confusion groups, on shared genes
        ari = float("nan")
        if len(keep) >= 3:
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform
            S = (M + M.T) / 2.0
            D = 1.0 - S
            np.fill_diagonal(D, 0.0)
            D = (D + D.T) / 2.0
            Z = linkage(squareform(D, checks=False), method="average")
            K = len(set(family_of(g) for g in keep))
            lab = dict(zip(keep, fcluster(Z, t=K, criterion="maxclust")))
            pg = paper_groups[chain].set_index("gene")["group"].to_dict()
            shared = [g for g in keep if g in pg]
            if len(shared) >= 3:
                ari = A._adjusted_rand([pg[g] for g in shared], [lab[g] for g in shared])

        m = {
            "cohort": cohort, "chain": chain, "n_scored": len(ok), "n_zero_pgen": n_zero,
            "v_gene_entropy": round(float(np.mean(vH)), 4),
            "j_gene_entropy": round(float(np.mean(jH)), 4),
            "top1": round(float(np.mean([r == 1 for r in vrank if r])), 4),
            "top10": round(float(np.mean([r <= 10 for r in vrank if r])), 4),
            "top20": round(float(np.mean([r <= 20 for r in vrank if r])), 4),
            "conf_0.5": round(float(np.mean([m0 >= 0.5 for m0 in vtop1])), 4),
            "conf_0.99": round(float(np.mean([m0 >= 0.99 for m0 in vtop1])), 4),
            "n_genes_ge20": len(keep),
            "ari_vs_paper_groups": None if np.isnan(ari) else round(float(ari), 4),
        }
        rows.append(m)
        summary[f"{cohort}_{chain}"] = m
        print("   ", {k: v for k, v in m.items() if k not in ("cohort", "chain")})

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "cohort_metrics.tsv", sep="\t", index=False)
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print("\n", tab.to_string(index=False), sep="")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
