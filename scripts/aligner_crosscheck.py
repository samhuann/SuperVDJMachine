"""R1 comment 1, fallback: cross-check the confusion blocks against an independent aligner.

    PYTHONPATH=. python3.9 scripts/aligner_crosscheck.py [--donors 40]

The reviewer asks for the confusion-block boundaries to be cross-verified with an
independent annotation strategy. The immunoSEQ pipeline is one: it aligns the actual reads
against its own germline reference and, where it cannot separate two or more V genes,
reports `v_gene` as "unresolved" and lists the candidates in `v_gene_ties`. Those tie sets
are an independent tool's own statement of which V genes are mutually indistinguishable,
obtained without OLGA and from sequence data rather than from a CDR3 string.

Two tests:

  1. Do tied gene pairs carry more posterior leakage in our confusion matrix than untied
     pairs?
  2. For the tie sets that cross IMGT family boundaries -- where the family partition
     would not predict indistinguishability -- do the genes fall together in our
     data-driven grouping? This is the discriminating comparison, since within-family ties
     are already explained by the nomenclature.

Writes results/tier3/aligner_crosscheck.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from itertools import combinations
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.gene_names import GeneReconciler  # noqa: E402
from supervdj.resolution import family_of  # noqa: E402

RAW = Path(os.environ.get("SUPERVDJ_EMERSON_DIR", "data/emerson-raw"))
META = Path(os.environ.get("SUPERVDJ_EMERSON_META", "data/emerson-metadata.csv"))
POSTERIORS = Path("results/posteriors.tsv")
OUT = Path("results/tier3")
MIN_COUNT = 20
CHAIN = "TRB"

AWK = r"""
NR==1 { for (i = 1; i <= NF; i++) h[$i] = i; next }
$h["frame_type"] == "In" && $h["amino_acid"] != "" && $h["v_gene_ties"] != "" {
    print $h["v_gene_ties"]
}
"""


def tie_sets(n_donors, seed):
    m = pd.read_csv(META)
    m = m[(m.cohort == "P_discovery") & (m.counting_method == "v2")].copy()
    m["age_num"] = pd.to_numeric(m.age, errors="coerce")
    have = {p.stem for p in RAW.glob("*.tsv")}
    elig = sorted(m[(m.age_num >= 18) & (m.subject.isin(have))].subject)
    import random
    pick = random.Random(seed).sample(elig, min(n_donors, len(elig)))
    c = Counter()
    for sid in sorted(pick):
        out = subprocess.run(["awk", "-F\t", AWK, str(RAW / f"{sid}.tsv")],
                             capture_output=True, text=True, check=True).stdout
        c.update(out.splitlines())
    return c, len(pick)


def leakage():
    """The manuscript's V-by-V leakage matrix for TRB, in one streaming pass."""
    acc, cnt = {}, Counter()
    with POSTERIORS.open(newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if (r["chain"] != CHAIN or r["axis"] != "V" or r["model"] != "pre"
                    or r["resolution"] != "gene" or r["status"] != "ok"
                    or (r["mode"] or "") != ""):
                continue
            tg = r["true_gene"]
            cnt[tg] += 1
            d = acc.setdefault(tg, Counter())
            d.update(json.loads(r["posterior_json"]))
    genes = sorted(g for g, n in cnt.items() if n >= MIN_COUNT)
    idx = {g: i for i, g in enumerate(genes)}
    M = np.zeros((len(genes), len(genes)))
    for tg, row in acc.items():
        i = idx.get(tg)
        if i is None:
            continue
        for g, m in row.items():
            j = idx.get(g)
            if j is not None:
                M[i, j] = m / cnt[tg]
    return genes, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ties, n_donors = tie_sets(args.donors, args.seed)
    print(f"{n_donors} donors, {sum(ties.values()):,} unresolved clones, "
          f"{len(ties)} distinct tie sets")

    rec = GeneReconciler.from_olga()
    genes, M = leakage()
    gi = {g: i for i, g in enumerate(genes)}
    S = (M + M.T) / 2.0                                   # symmetric leakage
    # the manuscript's grouping: dendrogram cut at the number of /DV-preserving families
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    Z = linkage(squareform(D, checks=False), method="average")
    fam = [family_of(g) for g in genes]
    grp = dict(zip(genes, fcluster(Z, t=len(set(fam)), criterion="maxclust")))
    famof = dict(zip(genes, fam))

    tied_pairs, unmapped, clones_used = {}, Counter(), 0
    for s, n in ties.items():
        mapped, bad = [], False
        for raw in s.split(","):
            g, _ = rec.map_v(raw.strip(), CHAIN)
            if g is None or g not in gi:
                unmapped[raw.strip()] += n
                bad = True
            else:
                mapped.append(g)
        if bad or len(mapped) < 2:
            continue
        clones_used += n
        for a, b in combinations(sorted(set(mapped)), 2):
            tied_pairs[(a, b)] = tied_pairs.get((a, b), 0) + n
    print(f"  usable tie sets -> {len(tied_pairs)} distinct gene pairs, "
          f"{clones_used:,} clones; {len(unmapped)} gene names not in the OLGA set "
          f"({sum(unmapped.values()):,} clones dropped)")

    all_pairs = list(combinations(genes, 2))
    tied = np.array([S[gi[a], gi[b]] for a, b in all_pairs if (a, b) in tied_pairs])
    untied = np.array([S[gi[a], gi[b]] for a, b in all_pairs if (a, b) not in tied_pairs])
    u = mannwhitneyu(tied, untied, alternative="greater")
    auc = u.statistic / (len(tied) * len(untied))

    # the discriminating subset: ties that cross IMGT family boundaries
    cross = [(a, b) for (a, b) in tied_pairs if famof[a] != famof[b]]
    within = [(a, b) for (a, b) in tied_pairs if famof[a] == famof[b]]
    cross_same_group = sum(grp[a] == grp[b] for a, b in cross)
    within_same_group = sum(grp[a] == grp[b] for a, b in within)
    # baseline: how often does any cross-family pair share one of our groups?
    cross_all = [(a, b) for a, b in all_pairs if famof[a] != famof[b]]
    base = sum(grp[a] == grp[b] for a, b in cross_all) / len(cross_all)

    rep = {
        "donors": n_donors, "unresolved_clones": int(sum(ties.values())),
        "distinct_tie_sets": len(ties), "usable_gene_pairs": len(tied_pairs),
        "clones_in_usable_sets": clones_used,
        "unmapped_gene_names": {k: int(v) for k, v in unmapped.items()},
        "leakage": {
            "n_tied_pairs": int(len(tied)), "n_untied_pairs": int(len(untied)),
            "median_tied": float(np.median(tied)), "median_untied": float(np.median(untied)),
            "auc": float(auc), "mannwhitney_p_greater": float(u.pvalue)},
        "grouping_agreement": {
            "cross_family_tied_pairs": len(cross),
            "cross_family_tied_in_same_group": int(cross_same_group),
            "cross_family_tied_fraction": (cross_same_group / len(cross)) if cross else None,
            "baseline_any_cross_family_pair_same_group": float(base),
            "within_family_tied_pairs": len(within),
            "within_family_tied_in_same_group": int(within_same_group)},
        "cross_family_tie_detail": [
            {"genes": [a, b], "families": [famof[a], famof[b]],
             "same_confusion_group": bool(grp[a] == grp[b]),
             "leakage": float(S[gi[a], gi[b]]), "clones": int(tied_pairs[(a, b)])}
            for a, b in sorted(cross, key=lambda p: -tied_pairs[p])],
    }
    L = rep["leakage"]
    print(f"\nleakage, tied vs untied gene pairs:")
    print(f"  tied   n={L['n_tied_pairs']:4d}  median {L['median_tied']:.5f}")
    print(f"  untied n={L['n_untied_pairs']:4d}  median {L['median_untied']:.5f}")
    print(f"  AUC {L['auc']:.3f}, Mann-Whitney p = {L['mannwhitney_p_greater']:.3g} (one-sided)")
    g = rep["grouping_agreement"]
    print(f"\ncross-family ties, the discriminating subset:")
    print(f"  {g['cross_family_tied_pairs']} pairs the aligner cannot separate despite "
          f"different IMGT families")
    if cross:
        print(f"  of those, {g['cross_family_tied_in_same_group']} fall in the same "
              f"confusion group ({100 * g['cross_family_tied_fraction']:.0f}%), against a "
              f"{100 * g['baseline_any_cross_family_pair_same_group']:.1f}% baseline for "
              f"cross-family pairs generally")
        for d in rep["cross_family_tie_detail"]:
            print(f"    {d['genes'][0]:10s} / {d['genes'][1]:10s}  "
                  f"same group: {str(d['same_confusion_group']):5s}  "
                  f"leakage {d['leakage']:.4f}  {d['clones']:,} clones")
    print(f"\nwithin-family ties: {g['within_family_tied_in_same_group']} of "
          f"{g['within_family_tied_pairs']} in the same confusion group")
    (OUT / "aligner_crosscheck.json").write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT / 'aligner_crosscheck.json'}")


if __name__ == "__main__":
    main()
