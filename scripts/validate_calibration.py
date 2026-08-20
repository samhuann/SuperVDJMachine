"""Held-out check that the shipped calibration curve generalizes.

    PYTHONPATH=. python scripts/validate_calibration.py

Splits the canonical set in half by a hash of the CDR3, builds the nominal-mass
lookup on the first half, then measures on the second half how often the annotated
V gene actually falls inside a candidate set requested at a given coverage. If the
curve only memorized its own data, held-out coverage would drift from the request.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import supervdj.aggregate as A

TARGETS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def mass_before_true(posteriors, truths):
    """Cumulative posterior mass strictly before the annotated gene, per sequence."""
    out = []
    for post, true in zip(posteriors, truths):
        items = sorted(post.items(), key=lambda kv: -kv[1])
        cum = 0.0
        for gene, mass in items:
            if gene == true:
                break
            cum += mass
        else:
            cum = np.inf                       # annotated gene is not a candidate
        out.append(cum)
    return np.array(out)


def half(seqs):
    """Stable 50/50 split on the CDR3 string."""
    return np.array([int(hashlib.md5(s.encode()).hexdigest(), 16) % 2 for s in seqs])


def main():
    df = A.load()
    rows = []
    for chain in ("TRA", "TRB"):
        s = A._slice(df, chain=chain, axis="V", model="pre", resolution="gene")
        before = mass_before_true((json.loads(p) for p in s["posterior_json"]),
                                  s["true_gene"].astype(str))
        fold = half(list(s["cdr3"]))
        fit, test = before[fold == 0], before[fold == 1]
        for t in TARGETS:
            nominal = float(np.quantile(fit[np.isfinite(fit)], t))   # fit on half A
            achieved = float((test < nominal).mean())                # measure on half B
            rows.append({"chain": chain, "target_coverage": t,
                         "nominal_mass_from_fit_half": round(nominal, 4),
                         "held_out_coverage": round(achieved, 4),
                         "error": round(achieved - t, 4),
                         "n_fit": int((fold == 0).sum()), "n_test": int((fold == 1).sum())})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    worst = out.error.abs().max()
    Path("results/validation_cohort").mkdir(parents=True, exist_ok=True)
    Path("results/validation_cohort/held_out_calibration.json").write_text(
        json.dumps({"rows": rows, "worst_abs_error": float(worst)}, indent=2))
    print(f"\nlargest deviation between requested and held-out coverage: {worst:.4f}")
    return 0 if worst < 0.02 else 1


if __name__ == "__main__":
    raise SystemExit(main())
