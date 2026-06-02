"""Dataclasses describing reference genes, candidate pairs, scores, and config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class GeneReference:
    """A single V or J gene reference entry.

    Attributes:
        gene: IMGT-style gene name (e.g., ``TRBV6-5``).
        chain: ``TRA`` or ``TRB``.
        gene_type: ``V`` or ``J``.
        functional: IMGT functional status (``F``, ``P``, ``ORF``).
        anchor: The germline amino acids that bound the CDR3 on this gene's
            side. For V genes this is the CDR3 N-terminal contribution
            (including the conserved Cys). For J genes this is the CDR3
            C-terminal contribution (ending with the conserved F or W).
        usage_prior: Prior probability of observing this gene in a naive
            repertoire. Used as a soft Bayesian prior.
    """

    gene: str
    chain: str
    gene_type: str
    functional: str
    anchor: str
    usage_prior: float = 0.0


@dataclass(frozen=True)
class CandidatePair:
    """A V/J gene pair to be scored against a CDR3."""

    chain: str
    cdr3: str
    v: GeneReference
    j: GeneReference


@dataclass
class CandidateScore:
    """All component scores for one candidate. All scores are in log space."""

    pair: CandidatePair
    log_pgen: float = 0.0
    log_selection: float = 0.0
    motif_score: float = 0.0
    gene_usage_prior: float = 0.0
    penalty: float = 0.0
    final_log_score: float = 0.0
    posterior_probability: float = 0.0
    explanation: str = ""

    @property
    def v_gene(self) -> str:
        return self.pair.v.gene

    @property
    def j_gene(self) -> str:
        return self.pair.j.gene


@dataclass
class PredictionConfig:
    """Tunable parameters for the prediction pipeline."""

    chain: str = "TRB"
    top_k: int = 20
    motif_weight: float = 1.0
    usage_weight: float = 1.0
    olga_weight: float = 0.1
    sonia_weight: float = 0.1
    min_cdr3_length: int = 5
    max_cdr3_length: int = 30
    relax_filters_if_empty: bool = True
    use_olga: bool = True
    use_sonia: bool = True
    n_anchor_match: int = 5
    c_anchor_match: int = 4
    soft_mismatch_penalty: float = 1.0
    require_c_start: bool = True
    require_f_or_w_end: bool = True
    v_model: Optional[Any] = None
    v_model_weight: float = 1.0
