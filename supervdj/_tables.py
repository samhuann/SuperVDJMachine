"""Shared helpers for reading VDJdb-style gene/cdr3/v/j tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def strip_allele(gene: str) -> str:
    """``TRBV19*01`` -> ``TRBV19``; leaves None / empty unchanged."""
    if not isinstance(gene, str) or not gene:
        return ""
    return gene.split("*", 1)[0].strip().upper()


def resolve_column(df: pd.DataFrame, *names: str) -> str:
    """Return the first column in ``df`` matching any of ``names`` (case-insensitive)."""
    by_lower = {c.lower(): c for c in df.columns}
    for name in names:
        found = by_lower.get(name.lower())
        if found is not None:
            return found
    raise ValueError(f"Table is missing one of these columns: {names}")


def read_table(path: Path) -> pd.DataFrame:
    """Read a ``.csv`` (comma) or ``.tsv``/other (tab) table as strings."""
    sep = "," if Path(path).suffix.lower() == ".csv" else "\t"
    return pd.read_csv(path, sep=sep, dtype=str, low_memory=False)
