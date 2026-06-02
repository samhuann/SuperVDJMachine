"""Tabular formatting for prediction results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from supervdj.scoring import PredictionResult

OUTPUT_COLUMNS = [
    "rank",
    "chain",
    "cdr3",
    "v_gene",
    "j_gene",
    "log_pgen",
    "log_selection",
    "motif_score",
    "gene_usage_prior",
    "penalty",
    "final_log_score",
    "posterior_probability",
    "explanation",
]


def result_to_dataframe(result: PredictionResult) -> pd.DataFrame:
    """Convert a :class:`PredictionResult` to a ranked DataFrame."""
    rows = []
    for rank, c in enumerate(result.candidates, start=1):
        rows.append(
            {
                "rank": rank,
                "chain": result.chain,
                "cdr3": result.cdr3,
                "v_gene": c.v_gene,
                "j_gene": c.j_gene,
                "log_pgen": round(c.log_pgen, 4),
                "log_selection": round(c.log_selection, 4),
                "motif_score": round(c.motif_score, 4),
                "gene_usage_prior": round(c.gene_usage_prior, 4),
                "penalty": round(c.penalty, 4),
                "final_log_score": round(c.final_log_score, 4),
                "posterior_probability": round(c.posterior_probability, 6),
                "explanation": c.explanation,
            }
        )
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return df


def _format_pretty(df: pd.DataFrame, confidence: str, warnings: Iterable[str]) -> str:
    """Render a result as a human-readable terminal table."""
    if df.empty:
        body = "(no candidates)"
    else:
        cols = [
            "rank",
            "v_gene",
            "j_gene",
            "posterior_probability",
            "final_log_score",
            "motif_score",
            "log_pgen",
            "log_selection",
            "explanation",
        ]
        body = df[cols].to_string(index=False)
    parts = [f"confidence: {confidence}", body]
    warn_list = [w for w in warnings if w]
    if warn_list:
        parts.append("warnings:")
        parts.extend(f"  - {w}" for w in warn_list)
    parts.append(
        "note: CDR3 amino acid alone rarely uniquely identifies V/J - "
        "rows above are *ranked plausible candidates*, not gene calls."
    )
    return "\n".join(parts)


def render(
    result: PredictionResult,
    fmt: str = "pretty",
    out: Optional[Path] = None,
) -> str:
    """Render result in ``pretty``, ``csv``, ``tsv``, or ``json`` form.

    If ``out`` is provided, the rendered text is written to that path; the
    rendered text is also returned to the caller.
    """
    fmt = fmt.lower()
    df = result_to_dataframe(result)
    if fmt == "csv":
        text = df.to_csv(index=False)
    elif fmt == "tsv":
        text = df.to_csv(index=False, sep="\t")
    elif fmt == "json":
        text = df.to_json(orient="records", indent=2)
    elif fmt == "pretty":
        text = _format_pretty(df, result.confidence, result.warnings)
    else:
        raise ValueError(f"Unknown format: {fmt!r}")
    if out is not None:
        Path(out).write_text(text, encoding="utf-8")
    return text
