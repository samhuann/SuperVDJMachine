"""Combine motif, usage, OLGA, SONIA terms into ranked V/J candidate scores."""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

from supervdj.filters import (
    chain_compatible,
    filter_j_genes,
    filter_v_genes,
    normalize_cdr3,
    validate_cdr3,
)
from supervdj.models import (
    CandidatePair,
    CandidateScore,
    GeneReference,
    PredictionConfig,
)
from supervdj.motif import (
    boundary_penalty,
    j_motif_score,
    usage_prior_log,
    v_motif_score,
)
from supervdj.olga_wrapper import compute_log_pgen
from supervdj.reference import load_default_reference, split_v_j
from supervdj.sonia_wrapper import compute_log_selection


@dataclass
class PredictionResult:
    """Top-level prediction result for a single CDR3.

    ``candidates`` is the joint (V, J) ranking. ``v_ranking`` and
    ``j_ranking`` are per-gene marginal rankings, each a list of
    ``(gene, posterior_probability)`` sorted descending.
    """

    cdr3: str
    chain: str
    candidates: List[CandidateScore]
    confidence: str
    warnings: List[str]
    relaxed: bool
    v_ranking: List[Tuple[str, float]] = field(default_factory=list)
    j_ranking: List[Tuple[str, float]] = field(default_factory=list)


def _marginalize(
    scores: Sequence[CandidateScore],
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Sum joint posteriors over the other axis to get V and J marginals."""
    v_marg: Dict[str, float] = defaultdict(float)
    j_marg: Dict[str, float] = defaultdict(float)
    for s in scores:
        v_marg[s.v_gene] += s.posterior_probability
        j_marg[s.j_gene] += s.posterior_probability
    v_ranking = sorted(v_marg.items(), key=lambda kv: kv[1], reverse=True)
    j_ranking = sorted(j_marg.items(), key=lambda kv: kv[1], reverse=True)
    return v_ranking, j_ranking


def _stable_softmax(values: Sequence[float]) -> List[float]:
    """Numerically stable softmax in log space."""
    if not values:
        return []
    m = max(values)
    if math.isinf(m) and m < 0:
        n = len(values)
        return [1.0 / n] * n
    exps = [math.exp(v - m) for v in values]
    s = sum(exps)
    if s <= 0:
        n = len(values)
        return [1.0 / n] * n
    return [e / s for e in exps]


def _single_usage_prior_log(ref: GeneReference, config: PredictionConfig) -> float:
    eps = 1e-6
    return config.usage_weight * math.log(max(ref.usage_prior, eps))


def _ranking_from_log_scores(
    scores: Sequence[Tuple[str, float]]
) -> List[Tuple[str, float]]:
    probs = _stable_softmax([s for _, s in scores])
    return sorted(
        [(gene, prob) for (gene, _), prob in zip(scores, probs)],
        key=lambda kv: kv[1],
        reverse=True,
    )


def _rank_v_genes(
    cdr3: str,
    v_refs: Sequence[GeneReference],
    config: PredictionConfig,
) -> List[Tuple[str, float]]:
    """Rank V genes from CDR3-side evidence only: the germline anchor motif, the
    V usage prior, and the supervised V model.

    OLGA and SONIA are deliberately excluded from V discrimination. OLGA's
    P(CDR3 | V) encodes essentially the same germline signal the supervised
    model already captures, at ~10^4x the cost per CDR3, and does not improve V
    recall. They remain in the joint (V, J) candidate scoring and the Pgen == 0
    impossibility filter (see :func:`_score_one`), where Pgen matters.
    """
    v_scores: Dict[str, float] = {}
    for v in v_refs:
        v_score, _ = v_motif_score(cdr3, v, config)
        v_scores[v.gene] = v_score + _single_usage_prior_log(v, config)

    if config.v_model is not None and config.v_model_weight > 0:
        genes = config.v_model.expand_genes(cdr3, list(v_scores))
        for gene in genes:
            v_scores.setdefault(gene, 0.0)
        model_scores = config.v_model.log_scores(cdr3, genes)
        for gene, model_score in model_scores.items():
            v_scores[gene] += config.v_model_weight * model_score

    return _ranking_from_log_scores(list(v_scores.items()))


def _confidence_label(
    posteriors: Sequence[float],
    motif_scores: Sequence[float],
    olga_available: bool,
    sonia_available: bool,
) -> str:
    """Heuristic confidence: high / medium / low."""
    if not posteriors:
        return "low"
    top = posteriors[0]
    second = posteriors[1] if len(posteriors) > 1 else 0.0
    strong_motif = max(motif_scores) > 4.0 if motif_scores else False
    integrations_off = not (olga_available or sonia_available)
    if integrations_off:
        return "low"
    if top > 0.5 and top - second > 0.2 and strong_motif:
        return "high"
    if top > 0.2:
        return "medium"
    return "low"


def _score_one(
    cdr3: str,
    v: GeneReference,
    j: GeneReference,
    chain: str,
    config: PredictionConfig,
) -> Tuple[Optional[CandidateScore], bool, bool, Optional[str], Optional[str]]:
    """Score a single V/J candidate.

    Returns ``(score_or_None, olga_ok, sonia_ok, warn_olga, warn_sonia)``.
    A score of ``None`` means OLGA (or SONIA, in the rare ``Q == 0`` case)
    judged the candidate biologically impossible and the caller should
    drop it before softmax.
    """
    pair = CandidatePair(chain=chain, cdr3=cdr3, v=v, j=j)
    v_score, v_expl = v_motif_score(cdr3, v, config)
    j_score, j_expl = j_motif_score(cdr3, j, config)
    motif_score = v_score + j_score
    usage = usage_prior_log(v, j, config)
    penalty = boundary_penalty(cdr3, v, j, config)

    olga_warn: Optional[str] = None
    sonia_warn: Optional[str] = None

    if config.use_olga:
        olga = compute_log_pgen(cdr3, v, j, chain)
        if olga.impossible:
            return None, olga.available, False, olga.warning, None
        log_pgen = config.olga_weight * olga.log_pgen
        olga_warn = olga.warning
        olga_available = olga.available
    else:
        log_pgen = 0.0
        olga_available = False

    if config.use_sonia:
        sonia = compute_log_selection(cdr3, v, j, chain)
        if sonia.impossible:
            return None, olga_available, sonia.available, olga_warn, sonia.warning
        log_selection = config.sonia_weight * sonia.log_selection
        sonia_warn = sonia.warning
        sonia_available = sonia.available
    else:
        log_selection = 0.0
        sonia_available = False

    final = log_pgen + log_selection + motif_score + usage - penalty
    explanation = "; ".join([v_expl, j_expl])
    return (
        CandidateScore(
            pair=pair,
            log_pgen=log_pgen,
            log_selection=log_selection,
            motif_score=motif_score,
            gene_usage_prior=usage,
            penalty=penalty,
            final_log_score=final,
            explanation=explanation,
        ),
        olga_available,
        sonia_available,
        olga_warn,
        sonia_warn,
    )


def score_candidates(
    cdr3: str,
    chain: str,
    v_refs: Sequence[GeneReference],
    j_refs: Sequence[GeneReference],
    config: PredictionConfig,
    progress: bool = False,
) -> Tuple[List[CandidateScore], bool, bool, List[str], int]:
    """Score the full V x J grid.

    Returns ``(scores, olga_any, sonia_any, warnings, n_excluded)`` where
    ``n_excluded`` counts candidates dropped because OLGA or SONIA judged
    them biologically impossible.
    """
    scores: List[CandidateScore] = []
    olga_any = False
    sonia_any = False
    seen_warnings: List[str] = []
    n_excluded = 0
    total = len(v_refs) * len(j_refs)
    bar = tqdm(
        total=total,
        disable=not progress,
        desc=f"Scoring {chain} V×J",
        unit="pair",
        leave=False,
    )
    for v in v_refs:
        for j in j_refs:
            score, olga_ok, sonia_ok, w_olga, w_sonia = _score_one(
                cdr3, v, j, chain, config
            )
            bar.update(1)
            olga_any = olga_any or olga_ok
            sonia_any = sonia_any or sonia_ok
            for w in (w_olga, w_sonia):
                if w and w not in seen_warnings:
                    seen_warnings.append(w)
            if score is None:
                n_excluded += 1
                continue
            scores.append(score)
    bar.close()
    return scores, olga_any, sonia_any, seen_warnings, n_excluded


def _attach_posteriors(scores: List[CandidateScore]) -> None:
    posts = _stable_softmax([s.final_log_score for s in scores])
    for s, p in zip(scores, posts):
        s.posterior_probability = p


def predict(
    cdr3: str,
    chain: str,
    config: Optional[PredictionConfig] = None,
    v_refs: Optional[Sequence[GeneReference]] = None,
    j_refs: Optional[Sequence[GeneReference]] = None,
    progress: bool = False,
) -> PredictionResult:
    """Predict and rank V/J candidates for a single CDR3.

    Args:
        cdr3: CDR3 amino-acid sequence.
        chain: ``TRA`` or ``TRB``.
        config: Optional ``PredictionConfig``; defaults are used otherwise.
        v_refs: Optional V gene references (overrides packaged defaults).
        j_refs: Optional J gene references (overrides packaged defaults).

    Returns:
        ``PredictionResult`` carrying ranked candidates, confidence, and
        any warnings emitted by the OLGA/SONIA integrations.
    """
    chain = chain.upper()
    config = config or PredictionConfig(chain=chain)
    cdr3 = normalize_cdr3(cdr3)
    warns: List[str] = []
    issues = validate_cdr3(cdr3, config)
    warns.extend(issues)

    if v_refs is None or j_refs is None:
        refs = load_default_reference(chain)
        v_default, j_default = split_v_j(refs)
        v_refs = v_refs if v_refs is not None else v_default
        j_refs = j_refs if j_refs is not None else j_default

    v_refs = chain_compatible(v_refs, chain)
    j_refs = chain_compatible(j_refs, chain)

    v_survivors = filter_v_genes(cdr3, v_refs, config)
    j_survivors = filter_j_genes(cdr3, j_refs, config)

    relaxed = False
    if (not v_survivors or not j_survivors) and config.relax_filters_if_empty:
        warns.append("No candidates survived strict filters; relaxing.")
        v_survivors = filter_v_genes(cdr3, v_refs, config, relaxed=True)
        j_survivors = filter_j_genes(cdr3, j_refs, config, relaxed=True)
        relaxed = True

    if not v_survivors or not j_survivors:
        warnings.warn("No V/J candidates survived even relaxed filters.")
        return PredictionResult(
            cdr3=cdr3,
            chain=chain,
            candidates=[],
            confidence="low",
            warnings=warns + ["No candidates"],
            relaxed=relaxed,
        )

    scores, olga_any, sonia_any, int_warns, n_excluded = score_candidates(
        cdr3, chain, v_survivors, j_survivors, config, progress=progress
    )
    warns.extend(int_warns)
    if n_excluded:
        warns.append(
            f"{n_excluded} candidate(s) excluded by OLGA/SONIA as biologically impossible."
        )
    if not olga_any and config.use_olga:
        warns.append("OLGA log Pgen unavailable for all candidates.")
    if not sonia_any and config.use_sonia:
        warns.append("SONIA log selection unavailable for all candidates.")

    if not scores:
        warnings.warn("All candidates were excluded by OLGA/SONIA as impossible.")
        return PredictionResult(
            cdr3=cdr3,
            chain=chain,
            candidates=[],
            confidence="low",
            warnings=warns + ["All candidates impossible per OLGA/SONIA"],
            relaxed=relaxed,
        )

    scores.sort(key=lambda s: s.final_log_score, reverse=True)
    _attach_posteriors(scores)
    _, j_ranking = _marginalize(scores)
    v_rank_refs = v_refs if config.v_model is not None else v_survivors
    v_ranking = _rank_v_genes(cdr3, v_rank_refs, config)
    top = scores[: config.top_k]

    posteriors = [s.posterior_probability for s in top]
    motif_scores = [s.motif_score for s in top]
    confidence = _confidence_label(posteriors, motif_scores, olga_any, sonia_any)
    if relaxed:
        confidence = "low"

    return PredictionResult(
        cdr3=cdr3,
        chain=chain,
        candidates=top,
        confidence=confidence,
        warnings=warns,
        relaxed=relaxed,
        v_ranking=v_ranking,
        j_ranking=j_ranking,
    )
