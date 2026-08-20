"""CLI: compute the per-sequence V/J posterior table (the source of truth).

Example::

    python -m supervdj.run \
        --input data/cdr3_annotated.tsv \
        --out results/posteriors.tsv \
        --post-modes grid fixed \
        --resolutions gene family

No figures are produced; this only writes the canonical posterior table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm

from supervdj.analyze import analyze_sequence
from supervdj.cache import ValueCache
from supervdj.io import load_dataset
from supervdj.models import load_chain_models
from supervdj.resolution import load_group_mapping


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True,
                   help="Annotated CDR3 table (chain/cdr3/V/J columns).")
    p.add_argument("--out", type=Path, default=Path("results/posteriors.tsv"))
    p.add_argument("--chains", nargs="+", default=None,
                   help="Restrict to TRA/TRB (default: whatever is in the data).")
    p.add_argument("--cache-dir", type=Path, default=Path("results/cache"))
    p.add_argument("--no-sonia", action="store_true",
                   help="Skip the SONNIA post-selection arm (OLGA pre-selection only).")
    p.add_argument("--post-modes", nargs="+", default=["grid", "fixed"],
                   choices=["grid", "fixed"])
    p.add_argument("--resolutions", nargs="+", default=["gene"],
                   help="Any of: gene, family, group.")
    p.add_argument("--group-map", type=Path, default=None,
                   help="gene->group TSV/CSV, required when 'group' resolution is used.")
    p.add_argument("--unmapped", choices=["self", "drop"], default="self",
                   help="For group resolution: genes absent from the map.")
    p.add_argument("--limit", type=int, default=None, help="Cap rows (debug).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    use_sonia = not args.no_sonia
    if args.no_sonia:
        post_modes: List[str] = []
    else:
        post_modes = list(args.post_modes)

    mapping = None
    if "group" in args.resolutions:
        if args.group_map is None:
            raise SystemExit("--resolutions group requires --group-map")
        mapping = load_group_mapping(args.group_map)

    rows = load_dataset(args.input)
    if args.chains:
        wanted = {c.upper() for c in args.chains}
        rows = [r for r in rows if r.chain in wanted]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No usable rows in input.")

    chains = sorted({r.chain for r in rows})
    print(f"Loaded {len(rows)} rows across chains {chains}; "
          f"sonia={'on' if use_sonia else 'off'}, post_modes={post_modes}, "
          f"resolutions={args.resolutions}")

    all_records: List[dict] = []
    for chain in chains:
        chain_rows = [r for r in rows if r.chain == chain]
        models = load_chain_models(chain, use_sonia=use_sonia)
        print(f"[{chain}] candidates: {len(models.v_genes)} V / {len(models.j_genes)} J")
        cache = ValueCache(args.cache_dir / f"cache_{chain}.pkl")
        try:
            for row in tqdm(chain_rows, desc=f"{chain}", unit="seq"):
                all_records.extend(analyze_sequence(
                    models, cache, row,
                    run_pre=True,
                    post_modes=post_modes,
                    resolutions=args.resolutions,
                    mapping=mapping,
                    unmapped=args.unmapped,
                ))
        finally:
            cache.save()
        print(f"[{chain}] cache: {len(cache)} entries "
              f"(hits={cache.hits}, misses={cache.misses})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_records)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
