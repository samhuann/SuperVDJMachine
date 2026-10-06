"""Build a healthy bulk TRB sample from the Emerson 2017 immunoSEQ cohort.

    PYTHONPATH=. python3.9 scripts/prepare_emerson.py [--donors 150] [--per-donor 3000]

R1 comment 2 and R2 major concern 3 ask for a healthy, unenriched repertoire: neither
antigen-sorted nor tumour-infiltrating. Emerson et al. 2017 (Nat Genet) is 786 healthy
bone-marrow-registry volunteers sequenced by immunoSEQ, bulk unsorted peripheral blood,
TRB only.

Sampling is spread across donors rather than taken from a few deep repertoires, so no
single individual's V usage dominates. Within a donor, clones are drawn uniformly over
*distinct rearrangements*, not weighted by template count, so clonal expansion does not
bias the sample toward expanded clones -- the manuscript's analysis set is likewise one
row per distinct rearrangement.

Writes results/emerson/emerson_trb_clones.tsv.gz with the columns validate_cohort.py
expects (cdr3_raw, v_raw, j_raw, chain), plus provenance columns it ignores.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import subprocess
import os
from pathlib import Path

import pandas as pd

RAW = Path(os.environ.get("SUPERVDJ_EMERSON_DIR", "data/emerson-raw"))
META = Path(os.environ.get("SUPERVDJ_EMERSON_META", "data/emerson-metadata.csv"))
OUT = Path("results/emerson")

# one awk pass per donor file: in-frame, productive, unambiguous V and J gene calls.
# Columns are looked up by header name because kit versions reorder them (they happen
# to agree across this cohort, but the lookup costs nothing and cannot silently break).
AWK = r"""
NR==1 { for (i = 1; i <= NF; i++) h[$i] = i; next }
$h["frame_type"] == "In" && $h["amino_acid"] != "" \
  && $h["v_gene"] != "" && $h["j_gene"] != "" \
  && $h["v_gene_ties"] == "" && $h["j_gene_ties"] == "" {
    print $h["amino_acid"] "\t" $h["v_gene"] "\t" $h["j_gene"] "\t" $h["templates"]
}
"""


def donors(n_donors, seed, min_age, method="v2"):
    """Healthy adult donors from one immunoSEQ chemistry.

    counting_method v1 and v2 are different quantification chemistries with different
    primer sets; v1 files report reads and leave `templates` empty. Mixing them would
    put two amplification biases in one sample, so one is chosen -- v2, which is 590 of
    the 666 discovery donors.
    """
    m = pd.read_csv(META)
    m = m[m.cohort == "P_discovery"].copy()          # the healthy discovery cohort
    m["age_num"] = pd.to_numeric(m.age, errors="coerce")
    adults = m[(m.age_num >= min_age) & (m.counting_method == method)]
    have = {p.stem for p in RAW.glob("*.tsv")}
    adults = adults[adults.subject.isin(have)]
    rng = random.Random(seed)
    pick = sorted(rng.sample(list(adults.subject), min(n_donors, len(adults))))
    return pick, adults.set_index("subject")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", type=int, default=150)
    ap.add_argument("--per-donor", type=int, default=3000, help="distinct clones per donor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-age", type=int, default=18)
    ap.add_argument("--method", default="v2", choices=("v1", "v2"),
                    help="immunoSEQ counting method; v1 and v2 are not pooled")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pick, meta = donors(args.donors, args.seed, args.min_age, args.method)
    print(f"{len(pick)} adult P_discovery {args.method} donors selected "
          f"(seed {args.seed}, age >= {args.min_age})")
    rng = random.Random(args.seed)
    rows, stats = [], []
    for k, sid in enumerate(pick, 1):
        out = subprocess.run(["awk", "-F\t", AWK, str(RAW / f"{sid}.tsv")],
                             capture_output=True, text=True, check=True).stdout
        clones = [ln.split("\t") for ln in out.splitlines()]
        take = clones if len(clones) <= args.per_donor else rng.sample(clones, args.per_donor)
        for aa, v, j, t in take:
            rows.append((aa, v, j, "TRB", sid, int(t) if t else -1))
        stats.append({"subject": sid, "clones_in_file": len(clones), "sampled": len(take),
                      "cmv": meta.loc[sid, "cmv"], "age": meta.loc[sid, "age"],
                      "sex": meta.loc[sid, "sex"]})
        if k % 25 == 0 or k == len(pick):
            print(f"  {k}/{len(pick)} donors, {len(rows):,} clones so far")

    df = pd.DataFrame(rows, columns=["cdr3_raw", "v_raw", "j_raw", "chain", "subject", "templates"])
    n_raw = len(df)
    df = df.drop_duplicates(subset=["cdr3_raw", "v_raw", "j_raw"])
    path = OUT / "emerson_trb_clones.tsv.gz"
    with gzip.open(path, "wt") as fh:
        df.to_csv(fh, sep="\t", index=False)
    s = pd.DataFrame(stats)
    s.to_csv(OUT / "emerson_donors.tsv", sep="\t", index=False)
    summary = {
        "donors": len(pick), "per_donor_cap": args.per_donor, "seed": args.seed,
        "min_age": args.min_age, "counting_method": args.method,
        "clones_sampled": int(n_raw), "clones_unique_tuple": int(len(df)),
        "median_clones_per_donor_in_file": float(s.clones_in_file.median()),
        "singleton_fraction": float((df.templates == 1).mean()),
        "cmv": s.cmv.value_counts().to_dict(),
        "age_median": float(pd.to_numeric(s.age, errors="coerce").median()),
    }
    (OUT / "emerson_sample.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{n_raw:,} clones sampled -> {len(df):,} unique (cdr3, V, J)")
    print(f"median clones per donor file: {summary['median_clones_per_donor_in_file']:,.0f}")
    print(f"singletons (templates == 1): {100 * summary['singleton_fraction']:.1f}%")
    print(f"CMV: {summary['cmv']}, median age {summary['age_median']:.0f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
