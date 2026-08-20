"""Shared normalization: raw adapter rows -> canonical rows + rejected rows.

This is intentionally NOT format-specific: every adapter parses its own native
format into uniform raw rows ``{cdr3_raw, v_raw, j_raw, chain}``, and this
module applies the two correctness steps in order:

1. **Gene-name reconciliation** onto the OLGA gene set (:mod:`ingest.gene_names`).
2. **CDR3 boundary normalization** to OLGA's junction convention
   (:mod:`ingest.imgt_boundaries`).

Rows that fail either step are returned as rejected rows with a reason.
Deduplication happens later (:mod:`ingest.dedup`), strictly after this.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import pandas as pd

from ingest.gene_names import GeneReconciler
from ingest.imgt_boundaries import ImgtBoundaries
from ingest.schema import CANONICAL_COLUMNS, REJECTED_COLUMNS


def normalize_rows(
    raw_rows: Iterable[Dict[str, str]],
    source: str,
    reconciler: GeneReconciler,
    boundaries: ImgtBoundaries,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize one source's raw rows into ``(kept_df, rejected_df)``."""
    kept: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []

    for r in raw_rows:
        chain = str(r.get("chain", "")).strip().upper()
        cdr3_raw = r.get("cdr3_raw")
        v_raw = r.get("v_raw")
        j_raw = r.get("j_raw")

        def reject(reason: str) -> None:
            rejected.append({
                "source": source, "chain": chain,
                "cdr3_raw": cdr3_raw, "v_raw": v_raw, "j_raw": j_raw,
                "reason": reason,
            })

        if chain not in ("TRA", "TRB"):
            reject(f"bad_chain:{chain!r}")
            continue

        v_gene, v_reason = reconciler.map_v(v_raw, chain)
        if v_gene is None:
            reject(f"v_{v_reason}")
            continue
        j_gene, j_reason = reconciler.map_j(j_raw, chain)
        if j_gene is None:
            reject(f"j_{j_reason}")
            continue

        cdr3_aa, b_reason = boundaries.normalize_cdr3(str(cdr3_raw or ""), j_gene)
        if cdr3_aa is None:
            reject(b_reason)
            continue

        kept.append({
            "cdr3_aa": cdr3_aa, "v_gene": v_gene, "j_gene": j_gene,
            "chain": chain, "source": source,
        })

    kept_df = pd.DataFrame(kept, columns=CANONICAL_COLUMNS)
    rejected_df = pd.DataFrame(rejected, columns=REJECTED_COLUMNS)
    return kept_df, rejected_df
