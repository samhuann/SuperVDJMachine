"""Test whether narrower V usage explains the tumour cohorts' lower V entropy.

    PYTHONPATH=. python scripts/cohort_usage_reweight.py

The held-out cohorts sit below the analysis set in beta V-gene conditional entropy
(2.4365 and 2.4565 nats against 2.7055). The manuscript attributes that to clonal
expansion narrowing V usage. This tests the attribution instead of asserting it:
reweight the analysis set so its annotated-V marginal matches the cohort's, leaving
every per-sequence posterior untouched, and see how much of the gap closes.

Only the marginal changes. If usage is the whole story the reweighted analysis-set
entropy lands on the cohort's; whatever is left over is something else.

Writes results/validation_cohort/usage_reweight.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import supervdj.aggregate as A
from validate_cohort import canonical_tuples, load_cohorts, normalize

OUT = Path("results/validation_cohort/usage_reweight.json")
SEED, NSUB, NBOOT = 0, 10000, 1000


def cohort_v_usage(cohort, chain):
    """The annotated-V marginal of exactly the subsample validate_cohort.py scored."""
    raw = load_cohorts()[(cohort, chain)]
    kept, _ = normalize(raw, cohort.lower(), chain)
    canon = canonical_tuples(chain)
    held = kept[~kept.apply(lambda r: (r.cdr3_aa, r.v_gene, r.j_gene) in canon, axis=1)]
    sub = held.sample(n=min(NSUB, len(held)), random_state=SEED)
    return sub.v_gene.value_counts(normalize=True), len(sub)


def reweight(H, genes, target):
    """Weighted mean of H after matching genes' marginal to target. Genes the target
    never uses get weight 0; genes the source never uses cannot be matched at all, so
    their target mass is reported as unmatched rather than silently ignored."""
    src = genes.value_counts(normalize=True)
    unmatched = float(target[~target.index.isin(src.index)].sum())
    w = genes.map(target).fillna(0.0).to_numpy() / genes.map(src).to_numpy()
    return float(np.average(H, weights=w)), unmatched, w


def boot(H, w, n=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(H), size=(n, len(H)))
    draws = np.array([np.average(H[i], weights=w[i]) for i in idx])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    df = A.load()
    obs = pd.read_csv("results/validation_cohort/cohort_metrics.tsv", sep="\t")
    out = {}
    for cohort, chain in (("NSCLC", "TRB"), ("HNSCC", "TRB"), ("NSCLC", "TRA")):
        s = A._slice(df, chain=chain, axis="V", model="pre", resolution="gene")
        H, genes = s.entropy_nats.to_numpy(), s.true_gene.reset_index(drop=True)
        base = float(H.mean())

        same, _, w_same = reweight(H, genes, genes.value_counts(normalize=True))
        assert abs(same - base) < 1e-9, f"self-reweight moved the mean: {same} vs {base}"

        target, n_sub = cohort_v_usage(cohort, chain)
        rw, unmatched, w = reweight(H, genes, target)
        lo, hi = boot(H, w)
        cohort_H = float(obs[(obs.cohort == cohort) & (obs.chain == chain)].v_gene_entropy.iloc[0])
        gap = base - cohort_H
        key = f"{cohort}_{chain}"
        out[key] = {"analysis_H": round(base, 4), "cohort_H": round(cohort_H, 4),
                    "reweighted_H": round(rw, 4), "ci": [round(lo, 4), round(hi, 4)],
                    "gap": round(gap, 4), "closed": round(base - rw, 4),
                    "frac_closed": round((base - rw) / gap, 3) if gap else None,
                    "unmatched_target_mass": round(unmatched, 5), "n_cohort": n_sub}
        print(f"[{key}] analysis {base:.4f} -> reweighted {rw:.4f} [{lo:.4f}, {hi:.4f}] "
              f"vs cohort {cohort_H:.4f}; closes {out[key]['closed']:.4f} of {gap:.4f} "
              f"({out[key]['frac_closed']})")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
