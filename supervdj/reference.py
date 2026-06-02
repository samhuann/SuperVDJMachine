"""Load V/J gene references.

Default path: real IMGT germline FASTAs shipped at ``supervdj/data/imgt/``
(see :mod:`supervdj.imgt`). Default references use uniform within-chain
priors so validation data cannot influence prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from supervdj.imgt import load_imgt_reference
from supervdj.models import GeneReference

REQUIRED_TSV_COLUMNS = ("gene", "chain", "gene_type", "anchor")


def _read_tsv(path: Path) -> List[GeneReference]:
    """Read a user-supplied TSV with columns ``gene/chain/gene_type/anchor``."""
    df = pd.read_csv(path, sep="\t")
    missing = [c for c in REQUIRED_TSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Reference file {path} is missing required columns: {missing}"
        )
    if "functional" not in df.columns:
        df["functional"] = "F"
    if "usage_prior" not in df.columns:
        df["usage_prior"] = 0.0
    refs: List[GeneReference] = []
    for row in df.itertuples(index=False):
        refs.append(
            GeneReference(
                gene=str(row.gene).strip(),
                chain=str(row.chain).strip().upper(),
                gene_type=str(row.gene_type).strip().upper(),
                functional=str(row.functional).strip(),
                anchor=str(row.anchor).strip().upper(),
                usage_prior=float(row.usage_prior or 0.0),
            )
        )
    return refs


def load_default_reference(
    chain: str,
    include_nonfunctional: bool = False,
) -> List[GeneReference]:
    """Load the packaged IMGT reference for ``TRA`` or ``TRB``."""
    v_refs, j_refs = load_imgt_reference(
        chain, include_nonfunctional=include_nonfunctional
    )
    refs: List[GeneReference] = list(v_refs) + list(j_refs)

    # Uniform prior keeps the log-prior term well defined without using
    # validation-set frequency information.
    n_v = sum(1 for r in refs if r.gene_type == "V") or 1
    n_j = sum(1 for r in refs if r.gene_type == "J") or 1
    uniform_v = 1.0 / n_v
    uniform_j = 1.0 / n_j
    return [
        GeneReference(
            gene=r.gene,
            chain=r.chain,
            gene_type=r.gene_type,
            functional=r.functional,
            anchor=r.anchor,
            usage_prior=uniform_v if r.gene_type == "V" else uniform_j,
        )
        for r in refs
    ]


def load_reference(
    chain: str,
    v_file: Optional[Path] = None,
    j_file: Optional[Path] = None,
    include_nonfunctional: bool = False,
) -> List[GeneReference]:
    """Load reference for ``chain``.

    Behaviour:
        * If neither ``v_file`` nor ``j_file`` is provided, load packaged
          IMGT references with uniform within-chain priors.
        * Otherwise, load the user-provided TSV(s); priors are taken from
          the file as-is.
    """
    if v_file is None and j_file is None:
        return load_default_reference(
            chain,
            include_nonfunctional=include_nonfunctional,
        )
    refs: List[GeneReference] = []
    for p in (v_file, j_file):
        if p is not None:
            refs.extend(_read_tsv(Path(p)))
    return refs


def split_v_j(
    refs: Sequence[GeneReference],
) -> Tuple[List[GeneReference], List[GeneReference]]:
    """Partition a reference list into V and J entries."""
    v = [r for r in refs if r.gene_type == "V"]
    j = [r for r in refs if r.gene_type == "J"]
    return v, j
