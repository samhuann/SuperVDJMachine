"""Core posterior computation over V and J genes for a CDR3.

Pre-selection (OLGA only):

    P(V | CDR3) = Pgen(CDR3, V) / Pgen(CDR3)        [J marginalized by OLGA]
    P(J | CDR3) = Pgen(CDR3, J) / Pgen(CDR3)        [V marginalized by OLGA]

Post-selection (OLGA Pgen x SONNIA selection factor Q):

    Ppost(CDR3, V, J) proportional to  Pgen(CDR3, V, J) * Q(CDR3, V, J)

with two J-handling modes, selectable per run:

* ``grid``  -- marginalize the other axis:  P_post(V|CDR3) propto
  sum_J Pgen(CDR3,V,J) Q(CDR3,V,J).  Consistent with the pre-selection
  definition (which also marginalizes J).  Costs a V x J grid of evaluations.
* ``fixed`` -- condition on the annotated other gene: P_post(V|CDR3, J=J_true)
  propto Pgen(CDR3,V,J_true) Q(CDR3,V,J_true).  Cheaper; a different quantity.

All posteriors are normalized within the OLGA candidate gene set, so they sum
to 1 by construction; ``Pgen(CDR3)`` is still recorded as a cross-check and to
flag impossible sequences.
"""

from __future__ import annotations

import math
import os
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from supervdj.cache import ValueCache
from supervdj.models import ChainModels
from supervdj.resolution import aggregate_posterior, gene_to_label

# ---------------------------------------------------------------------------
# Cached single-model evaluations
# ---------------------------------------------------------------------------


def _olga_pgen(models: ChainModels, cache: ValueCache, cdr3: str,
               v: Optional[str], j: Optional[str]) -> float:
    """Cached OLGA Pgen(cdr3, V_mask, J_mask); ``None`` marginalizes that axis."""
    k = cache.key("pgen", models.chain, cdr3, v, j)
    val = cache.get(k)
    if val is not None:
        return val
    p = models.olga.compute_aa_CDR3_pgen(cdr3, v, j, print_warnings=False)
    p = float(p) if p and p > 0 else 0.0
    cache.set(k, p)
    return p


def _sonia_q(models: ChainModels, cache: ValueCache, cdr3: str,
             pairs: Sequence[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
    """Cached SONNIA selection factor Q for each (V, J) pair of one CDR3."""
    out: Dict[Tuple[str, str], float] = {}
    misses: List[Tuple[str, str]] = []
    for v, j in pairs:
        k = cache.key("q", models.chain, cdr3, v, j)
        val = cache.get(k)
        if val is None:
            misses.append((v, j))
        else:
            out[(v, j)] = val
    if misses:
        seqs = [[cdr3, v, j] for v, j in misses]
        with open(os.devnull, "w") as devnull, \
                redirect_stdout(devnull), redirect_stderr(devnull):
            q_vals = models.sonia.evaluate_selection_factors(seqs)
        for (v, j), q in zip(misses, list(q_vals)):
            q = float(q) if q and q > 0 else 0.0
            cache.set(cache.key("q", models.chain, cdr3, v, j), q)
            out[(v, j)] = q
    return out


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {g: w / total for g, w in weights.items()}


# ---------------------------------------------------------------------------
# Posteriors (gene-level, before resolution aggregation)
# ---------------------------------------------------------------------------


def preselection_posterior(models: ChainModels, cache: ValueCache,
                           cdr3: str, axis: str) -> Dict[str, float]:
    """Gene-level pre-selection posterior over ``axis`` (``V`` or ``J``)."""
    axis = axis.upper()
    genes = models.candidates(axis)
    if axis == "V":
        weights = {g: _olga_pgen(models, cache, cdr3, g, None) for g in genes}
    else:
        weights = {g: _olga_pgen(models, cache, cdr3, None, g) for g in genes}
    return _normalize(weights)


def postselection_posterior(models: ChainModels, cache: ValueCache, cdr3: str,
                            axis: str, mode: str,
                            cond_gene: Optional[str]) -> Tuple[Dict[str, float], str]:
    """Gene-level post-selection posterior over ``axis``.

    Returns ``(posterior, status)``.  ``mode='fixed'`` conditions on
    ``cond_gene`` (the annotated gene of the *other* axis); if that gene is
    not an OLGA candidate the posterior is undefined and status reflects it.
    """
    axis = axis.upper()
    if models.sonia is None:
        return {}, "sonia_off"
    v_genes, j_genes = models.v_genes, models.j_genes

    if mode == "fixed":
        other = j_genes if axis == "V" else v_genes
        if cond_gene not in other:
            return {}, "cond_gene_not_candidate"
        target = v_genes if axis == "V" else j_genes
        pairs = [(g, cond_gene) if axis == "V" else (cond_gene, g) for g in target]
        q = _sonia_q(models, cache, cdr3, pairs)
        weights: Dict[str, float] = {}
        for g in target:
            v, j = (g, cond_gene) if axis == "V" else (cond_gene, g)
            weights[g] = _olga_pgen(models, cache, cdr3, v, j) * q.get((v, j), 0.0)
        return _normalize(weights), "ok"

    if mode == "grid":
        pairs = [(v, j) for v in v_genes for j in j_genes]
        q = _sonia_q(models, cache, cdr3, pairs)
        target = v_genes if axis == "V" else j_genes
        weights = {g: 0.0 for g in target}
        for v, j in pairs:
            ppost = _olga_pgen(models, cache, cdr3, v, j) * q.get((v, j), 0.0)
            weights[v if axis == "V" else j] += ppost
        return _normalize(weights), "ok"

    raise ValueError(f"Unknown post-selection mode: {mode!r}")


# ---------------------------------------------------------------------------
# Per-sequence summary statistics
# ---------------------------------------------------------------------------


@dataclass
class PosteriorStats:
    """Summary of one posterior distribution for one sequence."""

    top1_label: str
    top1_mass: float
    true_rank: Optional[int]
    entropy_nats: float
    n_candidates: int


def summarize(posterior: Dict[str, float], true_label: str) -> PosteriorStats:
    """Top-1 label, mass on it, rank of the true label, and entropy (nats)."""
    if not posterior:
        return PosteriorStats("", float("nan"), None, float("nan"), 0)
    ranked = sorted(posterior.items(), key=lambda kv: kv[1], reverse=True)
    top1_label, top1_mass = ranked[0]
    true_rank: Optional[int] = None
    for rank, (label, _) in enumerate(ranked, start=1):
        if label == true_label:
            true_rank = rank
            break
    entropy = -sum(p * math.log(p) for _, p in ranked if p > 0)
    return PosteriorStats(top1_label, top1_mass, true_rank, entropy, len(ranked))
