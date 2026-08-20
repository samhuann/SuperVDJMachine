"""Load annotated CDR3 datasets (CDR3 + chain + true V + true J).

Salvaged in spirit from the old ``supervdj.vdjdb`` / ``_tables`` loaders but
slimmed to exactly what the posterior pipeline needs.  Column names are
resolved case-insensitively against common VDJdb-style headers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from supervdj.models import strip_allele

_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class CDR3Row:
    """One annotated CDR3 record."""

    seq_id: str
    chain: str          # TRA / TRB
    cdr3: str
    true_v: str         # allele-stripped, e.g. TRBV19
    true_j: str         # allele-stripped


def _resolve(df: pd.DataFrame, *names: str) -> Optional[str]:
    by_lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    return None


def _read(path: Path) -> pd.DataFrame:
    sep = "," if Path(path).suffix.lower() == ".csv" else "\t"
    return pd.read_csv(path, sep=sep, dtype=str, low_memory=False)


def valid_cdr3(cdr3: str) -> bool:
    """True for a non-empty amino-acid CDR3 over the 20 standard residues."""
    return bool(cdr3) and set(cdr3) <= _VALID_AA


def load_dataset(
    path: Path,
    chain: Optional[str] = None,
    species: str = "HomoSapiens",
) -> List[CDR3Row]:
    """Load a CDR3 table into :class:`CDR3Row` records.

    Args:
        path: TSV/CSV with chain, CDR3, V, and J columns (VDJdb-style headers
            ``Gene/CDR3/V/J`` or lowercase ``chain/cdr3/v/j`` are accepted).
        chain: Optional ``TRA``/``TRB`` filter.
        species: Restrict to this ``Species`` value when the column exists.

    Returns:
        Rows with a valid CDR3 and both V and J annotated.  Allele suffixes
        are stripped to gene level.
    """
    df = _read(path)
    chain_col = _resolve(df, "chain", "gene", "Gene")
    cdr3_col = _resolve(df, "cdr3", "CDR3", "junction_aa", "cdr3_aa")
    v_col = _resolve(df, "v", "V", "v.segm", "v_gene", "v_call")
    j_col = _resolve(df, "j", "J", "j.segm", "j_gene", "j_call")
    id_col = _resolve(df, "id", "seq_id", "complex.id")
    species_col = _resolve(df, "species")

    missing = [n for n, c in
               [("chain", chain_col), ("cdr3", cdr3_col), ("v", v_col), ("j", j_col)]
               if c is None]
    if missing:
        raise ValueError(f"{path}: could not resolve required column(s): {missing}")

    if species_col and species:
        df = df[df[species_col] == species]
    df = df.dropna(subset=[cdr3_col, v_col, j_col, chain_col])

    rows: List[CDR3Row] = []
    for i, r in enumerate(df.to_dict("records")):
        row_chain = str(r[chain_col]).strip().upper()
        if row_chain not in {"TRA", "TRB"}:
            continue
        if chain is not None and row_chain != chain.upper():
            continue
        cdr3 = str(r[cdr3_col]).strip().upper()
        if not valid_cdr3(cdr3):
            continue
        seq_id = str(r[id_col]).strip() if id_col else f"row{i}"
        rows.append(
            CDR3Row(
                seq_id=seq_id,
                chain=row_chain,
                cdr3=cdr3,
                true_v=strip_allele(str(r[v_col])),
                true_j=strip_allele(str(r[j_col])),
            )
        )
    return rows
