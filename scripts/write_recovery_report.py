"""Write a short markdown report from VDJdb recovery benchmark outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


def _fmt_fraction(value: object) -> str:
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return ""


def _metric_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in ("top1", "top3", "top5", "top10", "top20", "top50") if c in df.columns]
    return cols


def _markdown_table(df: pd.DataFrame, fraction_cols: Sequence[str]) -> str:
    if df.empty:
        return "_No rows available._\n"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in headers:
            value = row[col]
            if col in fraction_cols:
                cells.append(_fmt_fraction(value))
            elif isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(results_dir: Path) -> Path:
    by_chain = pd.read_csv(results_dir / "recovery_by_chain.tsv", sep="\t")
    per_seq = pd.read_csv(results_dir / "per_sequence_results.tsv", sep="\t")
    metric_cols = _metric_columns(by_chain)

    dataset_size = (
        per_seq.groupby("chain")
        .size()
        .reset_index(name="n_sequences")
        .sort_values("chain")
    )
    exact = by_chain[by_chain["category"].isin(["V exact", "J exact", "V+J exact"])]
    family = by_chain[by_chain["category"].isin(["V family", "J family", "V+J family"])]
    rank_stats = by_chain[["chain", "category", "mrr", "median_rank", "recovered_at_top_k_max"]]

    out = results_dir / "recovery_report.md"
    parts = [
        "# VDJdb Recovery Benchmark Report",
        "",
        "## Dataset Size by Chain",
        "",
        _markdown_table(dataset_size, fraction_cols=[]),
        "",
        "## Top-k Exact Recovery",
        "",
        _markdown_table(exact[["chain", "category", *metric_cols]], fraction_cols=metric_cols),
        "",
        "## Top-k Family Recovery",
        "",
        _markdown_table(family[["chain", "category", *metric_cols]], fraction_cols=metric_cols),
        "",
        "## MRR and Median Rank",
        "",
        _markdown_table(
            rank_stats,
            fraction_cols=["recovered_at_top_k_max"],
        ),
        "",
        "## Interpretation",
        "",
        (
            "J gene usage is highly recoverable from CDR3 amino acid sequences, "
            "while exact V gene recovery is more ambiguous. Family-level recovery "
            "provides a more biologically appropriate view of V-gene ambiguity."
        ),
        "",
    ]
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a markdown recovery report.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/vdjdb_recovery"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = write_report(args.results_dir)
    print(f"wrote report to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
