"""``supervdj`` command-line interface.

Subcommands:

* ``predict`` — rank V/J candidates for a single CDR3.
* ``usage``   — print empirical V/J usage frequencies from a VDJdb TSV.
* ``eval``    — benchmark prediction against VDJdb labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from supervdj import __version__
from supervdj.models import PredictionConfig
from supervdj.output import render
from supervdj.reference import load_reference, split_v_j
from supervdj.scoring import predict


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="supervdj",
        description=(
            "Rank plausible TCR V/J gene candidates for a CDR3 amino-acid "
            "sequence. Output is a calibrated ranking, not a definitive call."
        ),
    )
    parser.add_argument("--version", action="version", version=f"supervdj {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- predict ---------------------------------------------------------
    p = sub.add_parser("predict", help="Predict ranked V/J candidates for a CDR3")
    p.add_argument("--chain", required=True, choices=["TRA", "TRB"])
    p.add_argument("--cdr3", required=True, help="CDR3 amino-acid sequence")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--format",
        dest="fmt",
        choices=["pretty", "csv", "tsv", "json"],
        default="pretty",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--v-ref", type=Path, default=None, help="Custom V reference TSV")
    p.add_argument("--j-ref", type=Path, default=None, help="Custom J reference TSV")
    p.add_argument(
        "--include-nonfunctional",
        action="store_true",
        help="Include IMGT pseudogenes and ORFs in the reference",
    )
    p.add_argument("--no-olga", action="store_true")
    p.add_argument("--no-sonia", action="store_true")
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the V×J scoring progress bar",
    )
    p.add_argument("--motif-weight", type=float, default=1.0)
    p.add_argument("--usage-weight", type=float, default=1.0)
    p.add_argument(
        "--v-cnn-dir",
        type=Path,
        default=None,
        help="Load a pre-trained VGeneCNN from this directory (see train_v_cnn.py)",
    )
    p.add_argument("--v-model-weight", type=float, default=1.0)
    p.add_argument(
        "--allow-noncanonical-ends",
        action="store_true",
        help="Do not require CDR3 to start with C and end with F/W",
    )

    # --- usage -----------------------------------------------------------
    u = sub.add_parser(
        "usage",
        help="Print empirical V/J usage frequencies from a VDJdb-style TSV",
    )
    u.add_argument("--vdjdb", type=Path, required=True)
    u.add_argument("--chain", required=True, choices=["TRA", "TRB"])
    u.add_argument(
        "--top", type=int, default=20, help="Show this many top V and J genes"
    )

    # --- eval ------------------------------------------------------------
    e = sub.add_parser(
        "eval",
        help="Evaluate prediction top-k accuracy against VDJdb labels",
    )
    e.add_argument("--vdjdb", type=Path, required=True)
    e.add_argument("--chain", required=True, choices=["TRA", "TRB"])
    e.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Limit the number of labelled rows scored (default 200)",
    )
    e.add_argument("--top-k", type=int, default=50)
    e.add_argument("--no-olga", action="store_true")
    e.add_argument("--no-sonia", action="store_true")
    e.add_argument(
        "--v-cnn-dir",
        type=Path,
        default=None,
        help="Load a pre-trained VGeneCNN from this directory (see train_v_cnn.py)",
    )
    e.add_argument("--v-model-weight", type=float, default=1.0)
    e.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the per-row evaluation progress bar",
    )
    e.add_argument(
        "--out-rows",
        type=Path,
        default=None,
        help="Optional CSV path to write per-row evaluation details",
    )

    return parser


def _load_v_model(args: argparse.Namespace):
    # The supervised V model is a pre-trained CNN artifact (see train_v_cnn.py).
    cnn_dir = getattr(args, "v_cnn_dir", None)
    if cnn_dir is None:
        return None
    from supervdj.v_model import VGeneCNN

    return VGeneCNN.load(cnn_dir)


def _make_predict_config(args: argparse.Namespace) -> PredictionConfig:
    return PredictionConfig(
        chain=args.chain,
        top_k=args.top_k,
        motif_weight=args.motif_weight,
        usage_weight=args.usage_weight,
        use_olga=not args.no_olga,
        use_sonia=not args.no_sonia,
        require_c_start=not args.allow_noncanonical_ends,
        require_f_or_w_end=not args.allow_noncanonical_ends,
        v_model=_load_v_model(args),
        v_model_weight=args.v_model_weight,
    )


def run_predict(args: argparse.Namespace) -> int:
    cfg = _make_predict_config(args)
    refs = load_reference(
        args.chain,
        v_file=args.v_ref,
        j_file=args.j_ref,
        include_nonfunctional=args.include_nonfunctional,
    )
    v_refs, j_refs = split_v_j(refs)
    result = predict(
        args.cdr3,
        args.chain,
        config=cfg,
        v_refs=v_refs,
        j_refs=j_refs,
        progress=not args.no_progress,
    )
    text = render(result, fmt=args.fmt, out=args.out)
    if args.out is None:
        print(text)
    else:
        print(f"wrote {len(result.candidates)} ranked candidates to {args.out}")
    return 0


def run_usage(args: argparse.Namespace) -> int:
    from supervdj.vdjdb import gene_usage_frequencies, load_vdjdb

    rows = load_vdjdb(args.vdjdb, chain=args.chain)
    v_frequency, j_frequency = gene_usage_frequencies(rows, chain=args.chain)
    v_df = (
        pd.DataFrame(sorted(v_frequency.items(), key=lambda kv: -kv[1]),
                     columns=["v_gene", "frequency"])
        .head(args.top)
    )
    j_df = (
        pd.DataFrame(sorted(j_frequency.items(), key=lambda kv: -kv[1]),
                     columns=["j_gene", "frequency"])
        .head(args.top)
    )
    print(f"{args.chain} V usage (top {args.top}, n={len(rows)} rows):")
    print(v_df.to_string(index=False))
    print()
    print(f"{args.chain} J usage (top {args.top}, n={len(rows)} rows):")
    print(j_df.to_string(index=False))
    return 0


def run_eval(args: argparse.Namespace) -> int:
    from supervdj.vdjdb import evaluate, load_vdjdb

    rows = load_vdjdb(args.vdjdb, chain=args.chain)
    cfg = PredictionConfig(
        chain=args.chain,
        top_k=args.top_k,
        use_olga=not args.no_olga,
        use_sonia=not args.no_sonia,
        v_model=_load_v_model(args),
        v_model_weight=args.v_model_weight,
    )
    refs = load_reference(args.chain)
    v_refs, j_refs = split_v_j(refs)

    summary = evaluate(
        rows,
        chain=args.chain,
        config=cfg,
        v_refs=v_refs,
        j_refs=j_refs,
        top_k=args.top_k,
        limit=args.limit,
        progress=not args.no_progress,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    if args.out_rows is not None:
        per_row = pd.DataFrame([r.__dict__ for r in summary.rows])
        per_row.to_csv(args.out_rows, index=False)
        print(f"wrote per-row details to {args.out_rows}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "predict":
        return run_predict(args)
    if args.command == "usage":
        return run_usage(args)
    if args.command == "eval":
        return run_eval(args)
    parser.error(f"Unknown command: {args.command}")
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
