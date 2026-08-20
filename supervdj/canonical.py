"""Build a model-consistent analysis set from one or more CDR3 tables.

One command, tables in -> canonical set out::

    python -m supervdj.canonical --input my_tcrs.tsv --out results/canonical

Three stages:

1. **Ingest** -- read each table, normalize gene names to the OLGA gene set and
   CDR3s to the IMGT junction convention, then deduplicate on the full
   ``(cdr3_aa, v_gene, j_gene, chain)`` tuple within and across files. This yields
   the unique rearrangements, each carrying the files it came from.
2. **Pgen filter** -- compute OLGA ``Pgen(CDR3)`` for every unique rearrangement and
   drop the sequences the model calls impossible (``Pgen == 0``); their pre- and
   post-selection posteriors are empty by construction, so they cannot be analyzed.
3. **Write** -- one TSV per chain plus a JSON summary of the counts.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from ingest.imgt_boundaries import DEFAULT_IMGT_DIR

_MODELS = {}


def _pgen_init(chain: str) -> None:
    """Load the chain's OLGA model once per worker process."""
    from supervdj.models import load_chain_models

    _MODELS[chain] = load_chain_models(chain, use_sonia=False).olga


def _pgen(args):
    chain, cdr3 = args
    return _MODELS[chain].compute_aa_CDR3_pgen(cdr3)


def ingest(inputs: Sequence[Path], out_dir: Path, imgt_dir: Path) -> pd.DataFrame:
    """Stage 1: read -> normalize -> intra- and inter-file dedup.

    Each input file is its own source, named after the filename, so the retained
    provenance says which file a rearrangement came from.
    """
    from ingest.adapters import generic
    from ingest.dedup import inter_dedup, intra_dedup
    from ingest.gene_names import GeneReconciler
    from ingest.imgt_boundaries import ImgtBoundaries
    from ingest.normalize import normalize_rows

    reconciler = GeneReconciler.from_olga()
    boundaries = ImgtBoundaries.from_dir(imgt_dir)

    kept, rejected = [], []
    for path in inputs:
        source = path.stem
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
        raw = list(generic.read_raw(path))
        ok, bad = normalize_rows(raw, source, reconciler, boundaries)
        deduped, collapsed = intra_dedup(ok)
        kept.append(deduped)
        rejected.append(bad)
        print(f"  [{source}] raw={len(raw):,} kept={len(ok):,} rejected={len(bad):,} "
              f"collapsed={collapsed:,} -> unique={len(deduped):,}")

    allrows = pd.concat(kept, ignore_index=True)
    unique = inter_dedup(allrows)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(rejected, ignore_index=True).to_csv(out_dir / "rejected.tsv", sep="\t", index=False)
    print(f"  inter-source dedup -> {len(unique):,} unique rearrangements "
          f"({len(pd.concat(rejected, ignore_index=True)):,} raw records rejected)")
    return unique


def pgen_filter(unique: pd.DataFrame, chain: str, workers: int) -> pd.DataFrame:
    """Stage 2: drop the sequences OLGA calls impossible (Pgen == 0)."""
    sub = unique[unique.chain == chain].reset_index(drop=True)
    jobs = [(chain, s) for s in sub.cdr3_aa]
    if workers > 1:
        with mp.get_context("spawn").Pool(workers, initializer=_pgen_init, initargs=(chain,)) as pool:
            pgen = pool.map(_pgen, jobs, chunksize=64)
    else:
        _pgen_init(chain)
        pgen = [_pgen(j) for j in jobs]
    sub["pgen"] = pgen
    canonical = sub[sub.pgen > 0].reset_index(drop=True)
    print(f"  [{chain}] {len(sub):,} unique -> {len(canonical):,} canonical "
          f"({len(sub) - len(canonical):,} zero-Pgen dropped)")
    return canonical


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, nargs="+", required=True, metavar="TABLE",
                    help="one or more CDR3 tables (TSV/CSV/XLSX) with chain, CDR3, V and J "
                         "columns")
    ap.add_argument("--imgt-dir", type=Path, default=DEFAULT_IMGT_DIR,
                    help="IMGT germline FASTAs used to place the junction boundaries")
    ap.add_argument("--out", type=Path, default=Path("results/canonical"),
                    help="output directory (default: results/canonical)")
    ap.add_argument("--chains", nargs="+", default=["TRA", "TRB"], choices=["TRA", "TRB"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="processes for the Pgen pass (default: cpu_count-1)")
    args = ap.parse_args(argv)

    print("stage 1: ingest")
    unique = ingest(args.input, args.out, args.imgt_dir)

    print(f"stage 2: OLGA Pgen filter ({args.workers} workers)")
    summary = {"unique": {}, "canonical": {}, "zero_pgen": {}}
    for chain in args.chains:
        canonical = pgen_filter(unique, chain, args.workers)
        canonical.to_csv(args.out / f"canonical_{chain}.tsv", sep="\t", index=False)
        n_unique = int((unique.chain == chain).sum())
        summary["unique"][chain] = n_unique
        summary["canonical"][chain] = len(canonical)
        summary["zero_pgen"][chain] = n_unique - len(canonical)

    with open(args.out / "canonical_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(f"\nwrote {args.out}/canonical_{{{','.join(args.chains)}}}.tsv "
          f"and canonical_summary.json")
    for chain in args.chains:
        print(f"  N({chain}) = {summary['canonical'][chain]:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
