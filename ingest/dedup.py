"""Deduplication and cross-source overlap.

A *duplicate* is the full tuple ``(cdr3_aa, v_gene, j_gene, chain)`` -- never
``cdr3_aa`` alone, because the same CDR3 with different V/J is a distinct
rearrangement and the object of study.

* **Intra-dataset**: collapse identical tuples within one source to one row.
* **Inter-dataset**: keep one canonical row per unique tuple, recording the set
  of sources it appeared in (``sources`` field) so overlap stays reportable.

Because the canonical schema keeps only the key columns plus ``source``, there
are no *other* fields that could disagree across sources; the conflict check is
therefore over any non-key, non-``source`` columns present and is reported
explicitly (vacuous for the canonical schema).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import pandas as pd

from ingest.schema import DEDUP_KEY


def intra_dedup(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Collapse identical tuples within one source. Returns (df, n_collapsed)."""
    if df.empty:
        return df, 0
    deduped = df.drop_duplicates(subset=DEDUP_KEY, keep="first").reset_index(drop=True)
    return deduped, len(df) - len(deduped)


def inter_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """One canonical row per unique tuple, with a ``sources`` set recorded."""
    grouped = (
        df.groupby(DEDUP_KEY, sort=False)["source"]
        .agg(lambda s: sorted(set(s)))
        .reset_index()
    )
    grouped["sources"] = grouped["source"].apply(lambda xs: ";".join(xs))
    grouped["n_sources"] = grouped["source"].apply(len)
    return grouped.drop(columns="source")


def pairwise_overlap(unique_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise counts of unique tuples shared between each source pair."""
    counts: Dict[Tuple[str, str], int] = {}
    for xs in unique_df["sources"].str.split(";"):
        for a, b in combinations(sorted(set(xs)), 2):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    rows = [{"source_a": a, "source_b": b, "shared_tuples": n}
            for (a, b), n in sorted(counts.items())]
    return pd.DataFrame(rows, columns=["source_a", "source_b", "shared_tuples"])


def find_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Tuples carrying differing values in any kept non-key, non-source field."""
    extra = [c for c in df.columns if c not in DEDUP_KEY and c != "source"]
    if not extra:
        return pd.DataFrame(columns=DEDUP_KEY + ["field", "values"])
    rows: List[Dict[str, object]] = []
    for key, g in df.groupby(DEDUP_KEY, sort=False):
        for col in extra:
            vals = sorted(set(g[col].dropna().astype(str)))
            if len(vals) > 1:
                rec = dict(zip(DEDUP_KEY, key if isinstance(key, tuple) else (key,)))
                rec["field"] = col
                rec["values"] = "|".join(vals)
                rows.append(rec)
    return pd.DataFrame(rows, columns=DEDUP_KEY + ["field", "values"])
