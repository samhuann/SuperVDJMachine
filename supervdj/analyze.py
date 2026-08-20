"""Turn posteriors into the per-sequence table that is the single source of truth.

One input CDR3 yields several rows: one per (axis V/J) x (model pre/post[mode])
x (resolution).  Each row records the spec-required summary -- top-1 label,
true-gene rank, posterior mass on the top label, posterior entropy -- plus the
full posterior as JSON so every later figure can be derived from this one
table without rerunning OLGA or SONNIA.
"""

from __future__ import annotations

import json
from typing import Dict, List, Mapping, Optional, Sequence

from supervdj.cache import ValueCache
from supervdj.io import CDR3Row
from supervdj.models import ChainModels
from supervdj.posterior import (
    postselection_posterior,
    preselection_posterior,
    summarize,
    _olga_pgen,
)
from supervdj.resolution import aggregate_posterior, gene_to_label


def _record(row: CDR3Row, axis: str, model: str, mode: str, resolution: str,
            true_gene: str, true_label: str, posterior: Dict[str, float],
            status: str, pgen_total: float) -> Dict[str, object]:
    stats = summarize(posterior, true_label)
    return {
        "seq_id": row.seq_id,
        "chain": row.chain,
        "cdr3": row.cdr3,
        "axis": axis,
        "model": model,
        "mode": mode,
        "resolution": resolution,
        "true_gene": true_gene,
        "true_label": true_label,
        "top1_label": stats.top1_label,
        "top1_mass": stats.top1_mass,
        "true_rank": stats.true_rank,
        "entropy_nats": stats.entropy_nats,
        "n_candidates": stats.n_candidates,
        "pgen_total": pgen_total,
        "status": status,
        "posterior_json": json.dumps(
            {k: round(v, 8) for k, v in sorted(
                posterior.items(), key=lambda kv: kv[1], reverse=True)},
            separators=(",", ":"),
        ),
    }


def analyze_sequence(
    models: ChainModels,
    cache: ValueCache,
    row: CDR3Row,
    *,
    run_pre: bool = True,
    post_modes: Sequence[str] = ("grid", "fixed"),
    resolutions: Sequence[str] = ("gene",),
    mapping: Optional[Mapping[str, str]] = None,
    unmapped: str = "self",
) -> List[Dict[str, object]]:
    """Compute all requested posteriors for one CDR3 and emit table rows."""
    pgen_total = _olga_pgen(models, cache, row.cdr3, None, None)
    impossible = pgen_total <= 0.0
    records: List[Dict[str, object]] = []

    for axis in ("V", "J"):
        true_gene = row.true_v if axis == "V" else row.true_j
        cond_gene = row.true_j if axis == "V" else row.true_v  # for fixed mode

        gene_posteriors: List[tuple] = []  # (model, mode, gene_posterior, status)
        if run_pre:
            pre = {} if impossible else preselection_posterior(
                models, cache, row.cdr3, axis)
            gene_posteriors.append(("pre", "", pre, "impossible" if impossible else "ok"))
        for mode in post_modes:
            if impossible:
                gene_posteriors.append(("post", mode, {}, "impossible"))
            else:
                post, status = postselection_posterior(
                    models, cache, row.cdr3, axis, mode, cond_gene)
                gene_posteriors.append(("post", mode, post, status))

        for model, mode, gene_post, status in gene_posteriors:
            for resolution in resolutions:
                label_post = aggregate_posterior(
                    gene_post, resolution, mapping, unmapped)
                true_label = gene_to_label(
                    true_gene, resolution, mapping, unmapped)
                records.append(_record(
                    row, axis, model, mode, resolution,
                    true_gene, true_label, label_post, status, pgen_total))
    return records
