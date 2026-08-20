"""Resolution: collapse a gene-level posterior to a group-level posterior.

The pipeline always computes the posterior at individual-gene resolution
(one OLGA/SONNIA evaluation per candidate gene, fully cached).  A *resolution*
then maps genes to labels:

* ``gene`` resolution is the identity (label == gene).
* a *group* resolution applies a user-supplied gene->group mapping and sums
  posterior mass within each group.

Because ``Pgen`` is additive over the OLGA gene mask, summing per-gene
posterior mass is exactly the grouped-mask posterior, so both resolutions are
served from one gene-level computation.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Optional

import pandas as pd

_FAMILY_RE = re.compile(r"^(TR[AB][VJ]\d+)(?:-\d+)?((?:/DV\d+)?)$")


def load_group_mapping(path: Path) -> Dict[str, str]:
    """Load a gene->group mapping from a 2-column TSV/CSV (``gene``, ``group``)."""
    sep = "," if Path(path).suffix.lower() == ".csv" else "\t"
    df = pd.read_csv(path, sep=sep, dtype=str)
    cols = {c.lower(): c for c in df.columns}
    gene_c = cols.get("gene") or df.columns[0]
    group_c = cols.get("group") or df.columns[1]
    return {
        str(g).strip(): str(grp).strip()
        for g, grp in zip(df[gene_c], df[group_c])
        if str(g).strip() and str(grp).strip()
    }


def family_of(gene: str) -> str:
    """Built-in germline-family label, e.g. ``TRBV7-2`` -> ``TRBV7``.

    Provided as a convenient default group mapping; a user-supplied mapping
    overrides it.
    """
    m = _FAMILY_RE.match(gene)
    return f"{m.group(1)}{m.group(2)}" if m else gene


def gene_to_label(
    gene: str,
    resolution: str,
    mapping: Optional[Mapping[str, str]],
    unmapped: str = "self",
) -> str:
    """Map one gene to its label under ``resolution``.

    Args:
        resolution: ``gene``, ``family``, or ``group``.
        mapping: required for ``group``; gene -> group.
        unmapped: for ``group``, genes absent from ``mapping`` become their own
            singleton label (``self``) or are dropped (``drop`` -> ``""``).
    """
    if resolution == "gene":
        return gene
    if resolution == "family":
        return family_of(gene)
    if resolution == "group":
        if mapping is None:
            raise ValueError("resolution='group' requires a gene->group mapping")
        if gene in mapping:
            return mapping[gene]
        return gene if unmapped == "self" else ""
    raise ValueError(f"Unknown resolution: {resolution!r}")


def aggregate_posterior(
    gene_posterior: Mapping[str, float],
    resolution: str,
    mapping: Optional[Mapping[str, str]] = None,
    unmapped: str = "self",
) -> Dict[str, float]:
    """Sum a gene-level posterior into a label-level posterior."""
    if resolution == "gene":
        return dict(gene_posterior)
    out: Dict[str, float] = defaultdict(float)
    for gene, prob in gene_posterior.items():
        label = gene_to_label(gene, resolution, mapping, unmapped)
        if label:
            out[label] += prob
    return dict(out)
