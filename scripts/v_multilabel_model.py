"""Train/evaluate a linear multi-label V-gene model on real CDR3 datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supervdj.v_multilabel import (  # noqa: E402
    LinearVGeneModel,
    load_neotcr_v_rows,
    mean_reciprocal_rank,
    split_rows,
    topk_recall,
)


def _metric_rows(model: LinearVGeneModel, rows, split: str) -> list[dict[str, object]]:
    record = {
        "chain": model.chain,
        "split": split,
        "n": len(rows),
        "mrr": mean_reciprocal_rank(model, rows),
    }
    for k in (1, 3, 5, 10, 20, 50):
        record[f"top{k}"] = topk_recall(model, rows, k)
    return [record]


def cmd_train(args: argparse.Namespace) -> int:
    rows = load_neotcr_v_rows(args.data, args.chain)
    model = LinearVGeneModel.from_rows(
        rows,
        chain=args.chain,
        min_label_count=args.min_label_count,
        max_iter=args.max_iter,
    )
    model.save(args.outdir)
    print(f"trained {args.chain.upper()} model on {len(rows)} NeoTCR rows")
    print(f"wrote model to {args.outdir}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    rows = load_neotcr_v_rows(args.data, args.chain)
    train, test = split_rows(rows, test_fraction=args.test_fraction, seed=args.seed)
    model = LinearVGeneModel.from_rows(
        train,
        chain=args.chain,
        min_label_count=args.min_label_count,
        max_iter=args.max_iter,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    model.save(args.outdir / f"linear_v_multilabel_{args.chain.upper()}")
    metrics = pd.DataFrame(
        _metric_rows(model, train, "train") + _metric_rows(model, test, "test")
    )
    out = args.outdir / f"neotcr_v_multilabel_{args.chain.upper()}.tsv"
    metrics.to_csv(out, sep="\t", index=False)
    print(f"wrote metrics to {out}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = LinearVGeneModel.load(args.model_dir)
    for rank, (gene, score) in enumerate(model.rank(args.cdr3)[: args.top_k], start=1):
        print(f"{rank}\t{gene}\t{score:.6f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", type=Path, default=Path("supervdj/data/NeoTCR.xlsx"))
    common.add_argument("--chain", choices=["TRA", "TRB"], default="TRB")
    common.add_argument("--min-label-count", type=int, default=2)
    common.add_argument("--max-iter", type=int, default=1000)

    train = sub.add_parser("train", parents=[common], help="Train on all NeoTCR rows.")
    train.add_argument("--outdir", type=Path, default=Path("results/neotcr_v_multilabel/model"))
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser(
        "evaluate",
        parents=[common],
        help="Train/test split evaluation on NeoTCR real rows.",
    )
    evaluate.add_argument("--test-fraction", type=float, default=0.2)
    evaluate.add_argument("--seed", type=int, default=1)
    evaluate.add_argument("--outdir", type=Path, default=Path("results/neotcr_v_multilabel"))
    evaluate.set_defaults(func=cmd_evaluate)

    predict = sub.add_parser("predict", help="Rank V genes for one CDR3.")
    predict.add_argument("--model-dir", type=Path, required=True)
    predict.add_argument("--cdr3", required=True)
    predict.add_argument("--top-k", type=int, default=10)
    predict.set_defaults(func=cmd_predict)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
