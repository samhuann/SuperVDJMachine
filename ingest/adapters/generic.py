"""Adapter: any CDR3 table with chain, CDR3, V and J columns.

This is the adapter to use for your own data. Column names are resolved the same
way :func:`supervdj.io.load_dataset` resolves them, so ``Gene/CDR3/V/J``,
AIRR-style ``locus/junction_aa/v_call/j_call`` and plain lowercase
``chain/cdr3/v/j`` all work. Everything downstream (gene reconciliation, IMGT
junction boundaries, dedup, the Pgen filter) is shared by every adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator

import pandas as pd

SOURCE = "generic"

_CHAIN = ("chain", "gene", "locus")
_CDR3 = ("cdr3", "cdr3_aa", "junction_aa", "cdr3.aa", "cdr3b", "cdr3a")
_V = ("v", "v_gene", "v_call", "v.segm", "vgene")
_J = ("j", "j_gene", "j_call", "j.segm", "jgene")


def _resolve(df: pd.DataFrame, names) -> str | None:
    by_lower = {c.lower(): c for c in df.columns}
    return next((by_lower[n] for n in names if n in by_lower), None)


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    sep = "," if path.suffix.lower() == ".csv" else "\t"
    return pd.read_csv(path, sep=sep, dtype=str)


def read_raw(path: Path) -> Iterator[Dict[str, str]]:
    df = _read(path)
    cols = {k: _resolve(df, n) for k, n in
            (("chain", _CHAIN), ("cdr3", _CDR3), ("v", _V), ("j", _J))}
    missing = [k for k, c in cols.items() if c is None]
    if missing:
        raise SystemExit(
            f"{path}: could not find column(s) for {', '.join(missing)}.\n"
            f"Columns present: {', '.join(map(str, df.columns))}"
        )
    for chain, cdr3, v, j in df[[cols["chain"], cols["cdr3"], cols["v"],
                                 cols["j"]]].itertuples(index=False, name=None):
        yield {"cdr3_raw": cdr3, "v_raw": v, "j_raw": j,
               "chain": str(chain).strip().upper()}
