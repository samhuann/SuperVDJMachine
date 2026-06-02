"""Log-space motif rewards for germline-compatible CDR3 endpoints."""

from __future__ import annotations

import math
from typing import Tuple

from supervdj.models import GeneReference, PredictionConfig

_MATCH_BONUS = math.log(4.0)
_MISMATCH_COST = math.log(0.5)


def _score_window(seq_a: str, seq_b: str) -> Tuple[float, int, int]:
    """Score two equal-length windows residue by residue."""
    score = 0.0
    matches = 0
    n = min(len(seq_a), len(seq_b))
    for i in range(n):
        if seq_a[i] == seq_b[i]:
            score += _MATCH_BONUS
            matches += 1
        else:
            score += _MISMATCH_COST
    return score, matches, n


def v_motif_score(
    cdr3: str, v: GeneReference, config: PredictionConfig
) -> Tuple[float, str]:
    """Score how well the V anchor explains the CDR3 N-terminus."""
    k = min(config.n_anchor_match, len(cdr3), len(v.anchor))
    if k == 0:
        return 0.0, "no overlap"
    score, matches, n = _score_window(cdr3[:k], v.anchor[:k])
    expl = f"V-anchor {matches}/{n} match ({v.anchor[:k]} vs {cdr3[:k]})"
    return config.motif_weight * score, expl


def j_motif_score(
    cdr3: str, j: GeneReference, config: PredictionConfig
) -> Tuple[float, str]:
    """Score how well the J anchor explains the CDR3 C-terminus."""
    k = min(config.c_anchor_match, len(cdr3), len(j.anchor))
    if k == 0:
        return 0.0, "no overlap"
    score, matches, n = _score_window(cdr3[-k:], j.anchor[-k:])
    expl = f"J-anchor {matches}/{n} match ({j.anchor[-k:]} vs {cdr3[-k:]})"
    return config.motif_weight * score, expl


def usage_prior_log(
    v: GeneReference, j: GeneReference, config: PredictionConfig
) -> float:
    """Combine V and J usage priors in log space, with a small floor."""
    eps = 1e-6
    return config.usage_weight * (
        math.log(max(v.usage_prior, eps)) + math.log(max(j.usage_prior, eps))
    )


def boundary_penalty(
    cdr3: str, v: GeneReference, j: GeneReference, config: PredictionConfig
) -> float:
    """Soft penalty applied when V/J anchors collectively over-explain CDR3."""
    overlap = len(v.anchor) + len(j.anchor) - len(cdr3)
    if overlap <= 0:
        return 0.0
    return config.soft_mismatch_penalty * float(overlap)
