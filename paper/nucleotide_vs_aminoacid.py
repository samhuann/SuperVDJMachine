"""R2 minor concern 8: nucleotide versus amino-acid recovery, on the healthy bulk cohort.

    PYTHONPATH=. python3.9 scripts/nucleotide_vs_aminoacid.py [--n 10000] [--workers 60]

The three analysis-set databases record only amino-acid CDR3s, so this comparison is not
possible there. The Emerson immunoSEQ exports carry the full nucleotide `rearrangement`
alongside the amino-acid CDR3, with `v_index` and `cdr3_length` locating the junction:
rearrangement[v_index : v_index + cdr3_length] translates to the reported amino_acid for
20,000/20,000 in-frame clones checked, so the nucleotide CDR3 is recoverable exactly.

For each sequence the same posterior recipe is run twice -- once over the amino-acid CDR3,
once over the nucleotide CDR3 -- marginalizing the opposite gene in both cases. The
amino-acid side calls the manuscript's own preselection_posterior; the nucleotide side is
the same recipe with OLGA's nucleotide Pgen. The comparison is paired: only sequences
scored under both representations are reported, so the difference is translation loss and
not a difference in which sequences were scored.

Writes results/tier3/3e_nucleotide.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import random
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.imgt_boundaries import ImgtBoundaries, _translate  # noqa: E402
from ingest.gene_names import GeneReconciler  # noqa: E402

RAW = Path(os.environ.get("SUPERVDJ_EMERSON_DIR", "data/emerson-raw"))
META = Path(os.environ.get("SUPERVDJ_EMERSON_META", "data/emerson-metadata.csv"))
OUT = Path("results/tier3")
CHAIN = "TRB"

AWK = r"""
NR==1 { for (i = 1; i <= NF; i++) h[$i] = i; next }
$h["frame_type"] == "In" && $h["amino_acid"] != "" && $h["rearrangement"] != "" \
  && $h["v_gene"] != "" && $h["j_gene"] != "" \
  && $h["v_gene_ties"] == "" && $h["j_gene_ties"] == "" {
    print $h["amino_acid"] "\t" $h["v_gene"] "\t" $h["j_gene"] "\t" \
          substr($h["rearrangement"], $h["v_index"] + 1, $h["cdr3_length"])
}
"""

_M = {}


def _init():
    from supervdj.cache import ValueCache
    from supervdj.models import load_chain_models
    _M["m"] = load_chain_models(CHAIN, use_sonia=False)
    _M["c"] = ValueCache(None)


def _nt_posterior(cdr3_nt, axis):
    """The manuscript's pre-selection recipe with OLGA's nucleotide Pgen: score each
    candidate gene with the opposite gene left unconstrained, then normalize."""
    m = _M["m"]
    genes = m.candidates(axis)
    w = {}
    for g in genes:
        v, j = (g, None) if axis == "V" else (None, g)
        p = m.olga.compute_nt_CDR3_pgen(cdr3_nt, v, j, print_warnings=False)
        w[g] = float(p) if p and p > 0 else 0.0
    tot = sum(w.values())
    return {g: x / tot for g, x in w.items()} if tot > 0 else {}


def _both(task):
    """(V, J) posteriors from the amino-acid CDR3 and from the nucleotide CDR3."""
    from supervdj.posterior import preselection_posterior
    aa, nt = task
    out = {}
    for axis in ("V", "J"):
        a = preselection_posterior(_M["m"], _M["c"], aa, axis)
        n = _nt_posterior(nt, axis)
        if not a or not n:
            return None
        out[axis] = (a, n)
    return out


def entropy(p):
    return float(-sum(m * np.log(m) for m in p.values() if m > 0))


def rank(p, gene):
    return 1 + sum(1 for g, m in p.items() if m > p.get(gene, 0.0))


def donors(n_donors, seed):
    m = pd.read_csv(META)
    m = m[(m.cohort == "P_discovery") & (m.counting_method == "v2")].copy()
    m["age_num"] = pd.to_numeric(m.age, errors="coerce")
    have = {p.stem for p in RAW.glob("*.tsv")}
    elig = m[(m.age_num >= 18) & (m.subject.isin(have))]
    return sorted(random.Random(seed).sample(list(elig.subject), min(n_donors, len(elig))))


def canonical_cdr3s():
    d = pd.read_csv("results/ingest/canonical_unique.tsv", sep="\t")
    return set(d[d.chain == "TRB"].cdr3_aa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--donors", type=int, default=60)
    ap.add_argument("--per-donor", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rec = GeneReconciler.from_olga()
    bnd = ImgtBoundaries.from_dir(Path("data/imgt"))
    canon = canonical_cdr3s()
    rng = random.Random(args.seed)

    rows, seen = [], set()
    drop = {"gene": 0, "boundary": 0, "translate": 0, "overlap": 0}
    for sid in donors(args.donors, args.seed):
        out = subprocess.run(["awk", "-F\t", AWK, str(RAW / f"{sid}.tsv")],
                             capture_output=True, text=True, check=True).stdout.splitlines()
        for ln in (out if len(out) <= args.per_donor else rng.sample(out, args.per_donor)):
            aa_raw, v_raw, j_raw, nt = ln.split("\t")
            v, _ = rec.map_v(v_raw, CHAIN)
            j, _ = rec.map_j(j_raw, CHAIN)
            if v is None or j is None:
                drop["gene"] += 1
                continue
            aa, _ = bnd.normalize_cdr3(aa_raw, j)
            if aa is None:
                drop["boundary"] += 1
                continue
            # normalize_cdr3 validates without trimming, so the nucleotide maps 1:1
            if len(nt) != 3 * len(aa) or _translate(nt, 1) != aa:
                drop["translate"] += 1
                continue
            if aa in canon:
                drop["overlap"] += 1
                continue
            key = (aa, v, j)
            if key in seen:
                continue
            seen.add(key)
            rows.append((aa, nt, v, j))
    print(f"{len(rows):,} unique held-out rearrangements with a verified nucleotide CDR3")
    print(f"  dropped: {drop}")

    take = rows if len(rows) <= args.n else rng.sample(rows, args.n)
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        res = pool.map(_both, [(a, n) for a, n, _, _ in take], chunksize=8)

    acc = {ax: {"aa_H": [], "nt_H": [], "aa_r": [], "nt_r": []} for ax in ("V", "J")}
    n_scored = 0
    for (aa, nt, v, j), r in zip(take, res):
        if r is None:
            continue
        n_scored += 1
        for ax, true in (("V", v), ("J", j)):
            pa, pn = r[ax]
            acc[ax]["aa_H"].append(entropy(pa))
            acc[ax]["nt_H"].append(entropy(pn))
            acc[ax]["aa_r"].append(rank(pa, true))
            acc[ax]["nt_r"].append(rank(pn, true))

    report = {"chain": CHAIN, "n_candidates_scored": len(take), "n_paired": n_scored,
              "donors": args.donors, "seed": args.seed, "drops": drop}
    for ax in ("V", "J"):
        a = acc[ax]
        aH, nH = np.array(a["aa_H"]), np.array(a["nt_H"])
        ar, nr = np.array(a["aa_r"]), np.array(a["nt_r"])
        report[ax] = {
            "amino_acid": {"mean_entropy_nats": float(aH.mean()),
                           "top1": float((ar == 1).mean()), "top10": float((ar <= 10).mean())},
            "nucleotide": {"mean_entropy_nats": float(nH.mean()),
                           "top1": float((nr == 1).mean()), "top10": float((nr <= 10).mean())},
            "translation_loss": {
                "entropy_nats": float(aH.mean() - nH.mean()),
                "top1": float((nr == 1).mean() - (ar == 1).mean()),
                "top10": float((nr <= 10).mean() - (ar <= 10).mean()),
                "fraction_of_amino_acid_entropy": float((aH.mean() - nH.mean()) / aH.mean()),
                "rank_improved": float((nr < ar).mean()),
                "rank_unchanged": float((nr == ar).mean()),
                "rank_worsened": float((nr > ar).mean())},
        }
        r = report[ax]
        print(f"\n[{CHAIN} {ax}]  n = {n_scored:,} paired")
        print(f"  amino acid  entropy {r['amino_acid']['mean_entropy_nats']:.4f}  "
              f"top-1 {r['amino_acid']['top1']:.4f}  top-10 {r['amino_acid']['top10']:.4f}")
        print(f"  nucleotide  entropy {r['nucleotide']['mean_entropy_nats']:.4f}  "
              f"top-1 {r['nucleotide']['top1']:.4f}  top-10 {r['nucleotide']['top10']:.4f}")
        t = r["translation_loss"]
        print(f"  lost in translation: {t['entropy_nats']:.4f} nats "
              f"({100 * t['fraction_of_amino_acid_entropy']:.1f}% of the amino-acid entropy), "
              f"top-1 {t['top1']:+.4f}")
        print(f"  per-sequence rank: improved {100*t['rank_improved']:.1f}%, "
              f"unchanged {100*t['rank_unchanged']:.1f}%, worsened {100*t['rank_worsened']:.1f}%")
    (OUT / "3e_nucleotide.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT / '3e_nucleotide.json'}")


if __name__ == "__main__":
    main()
