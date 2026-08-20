"""CLI: normalize and deduplicate CDR3 tables, with an overlap report.

Per input file: parse -> normalize (gene reconciliation + CDR3 boundary) ->
intra-file dedup. Across files: inter-file dedup with a retained ``sources``
provenance record, pairwise overlap, and a conflict check. Each file is its own
source, named after the filename, so the overlap table says how much two
datasets share.

    python -m ingest.run_ingest --input cohort_a.tsv cohort_b.csv

This is the reporting front end. :mod:`supervdj.canonical` runs the same
normalization and then applies the OLGA Pgen filter.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from ingest.adapters import generic
from ingest.dedup import find_conflicts, intra_dedup, inter_dedup, pairwise_overlap
from ingest.gene_names import GeneReconciler
from ingest.imgt_boundaries import DEFAULT_IMGT_DIR, ImgtBoundaries
from ingest.normalize import normalize_rows
from ingest.schema import CANONICAL_COLUMNS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, nargs="+", required=True, metavar="TABLE",
                   help="CDR3 tables (TSV/CSV/XLSX) with chain, CDR3, V and J columns")
    p.add_argument("--out-dir", type=Path, default=Path("results/ingest"))
    p.add_argument("--imgt-dir", type=Path, default=DEFAULT_IMGT_DIR)
    p.add_argument("--head", type=int, default=8, help="Rows to preview.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    print("Loading OLGA gene sets + IMGT boundary reference ...")
    reconciler = GeneReconciler.from_olga()
    boundaries = ImgtBoundaries.from_dir(args.imgt_dir)
    print(f"  IMGT boundary ref: {len(boundaries.v_end_residue)} V genes, "
          f"{len(boundaries.j_end_residue)} J genes "
          f"(convention: CDR3 = conserved C ... conserved F/W, the OLGA junction)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    kept_frames: List[pd.DataFrame] = []
    rejected_frames: List[pd.DataFrame] = []

    print("\n=== Per-source ===")
    for path in args.input:
        source = path.stem
        raw = list(generic.read_raw(path))
        kept, rejected = normalize_rows(raw, source, reconciler, boundaries)
        deduped, collapsed = intra_dedup(kept)
        kept_frames.append(deduped)
        rejected_frames.append(rejected)
        n_raw = len(raw)
        print(f"\n[{source}]  raw={n_raw}")
        print(f"  kept (pre-dedup) = {len(kept)}   rejected = {len(rejected)}")
        print(f"  intra-dedup collapsed = {collapsed}  ->  unique rows = {len(deduped)}")
        if len(rejected):
            top = rejected["reason"].str.split(":").str[0].value_counts().head(6)
            print(f"  rejection reasons (top): {dict(top)}")
        if len(deduped):
            print(f"  first {args.head} normalized rows:")
            print(deduped.head(args.head).to_string(index=False))

    rejected_all = (pd.concat(rejected_frames, ignore_index=True)
                    if rejected_frames else pd.DataFrame())
    canonical_all = (pd.concat(kept_frames, ignore_index=True)
                     if kept_frames else pd.DataFrame(columns=CANONICAL_COLUMNS))
    canonical_all.to_csv(args.out_dir / "canonical_all.tsv", sep="\t", index=False)
    rejected_all.to_csv(args.out_dir / "rejected.tsv", sep="\t", index=False)

    # Inter-dataset dedup + overlap (meaningful once >1 source is present).
    unique = inter_dedup(canonical_all)
    overlap = pairwise_overlap(unique)
    conflicts = find_conflicts(canonical_all)
    unique.to_csv(args.out_dir / "canonical_unique.tsv", sep="\t", index=False)
    overlap.to_csv(args.out_dir / "overlap_pairwise.tsv", sep="\t", index=False)
    conflicts.to_csv(args.out_dir / "conflicts.tsv", sep="\t", index=False)

    print("\n=== Overlap summary ===")
    print(f"  unique rearrangements (cdr3_aa,v_gene,j_gene,chain): {len(unique)}")
    if not overlap.empty:
        print(overlap.to_string(index=False))
    else:
        print("  (single input -> no cross-file overlap to report)")
    print(f"  field conflicts among kept columns: {len(conflicts)} "
          f"(canonical schema keeps only key+source, so none are possible)")
    print(f"\nWrote outputs to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
