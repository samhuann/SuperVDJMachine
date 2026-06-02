"""Biological compatibility filters and CDR3 sanity checks."""

from __future__ import annotations

from typing import List, Sequence

from supervdj.models import GeneReference, PredictionConfig

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def normalize_cdr3(cdr3: str) -> str:
    """Strip whitespace and upper-case a CDR3 sequence."""
    return cdr3.strip().upper()


def validate_cdr3(cdr3: str, config: PredictionConfig) -> List[str]:
    """Return a list of human-readable validation issues; empty if valid."""
    issues: List[str] = []
    if not cdr3:
        issues.append("CDR3 is empty")
        return issues
    if len(cdr3) < config.min_cdr3_length:
        issues.append(f"CDR3 shorter than {config.min_cdr3_length} residues")
    if len(cdr3) > config.max_cdr3_length:
        issues.append(f"CDR3 longer than {config.max_cdr3_length} residues")
    bad = sorted(set(cdr3) - VALID_AA)
    if bad:
        issues.append(f"CDR3 contains non-standard residues: {''.join(bad)}")
    if config.require_c_start and not cdr3.startswith("C"):
        issues.append("CDR3 does not start with conserved Cys (C)")
    if config.require_f_or_w_end and cdr3[-1] not in ("F", "W"):
        issues.append("CDR3 does not end with conserved Phe/Trp (F/W)")
    return issues


def _has_match(window_a: str, window_b: str) -> bool:
    """True if at least one position matches in the compared window."""
    return any(a == b for a, b in zip(window_a, window_b))


def filter_v_genes(
    cdr3: str,
    v_refs: Sequence[GeneReference],
    config: PredictionConfig,
    relaxed: bool = False,
) -> List[GeneReference]:
    """Return V genes whose anchor is compatible with the CDR3 N-terminus."""
    k = max(2, config.n_anchor_match - (1 if relaxed else 0))
    survivors: List[GeneReference] = []
    for v in v_refs:
        window = min(k, len(cdr3), len(v.anchor))
        if relaxed or window == 0 or _has_match(cdr3[:window], v.anchor[:window]):
            survivors.append(v)
    return survivors


def filter_j_genes(
    cdr3: str,
    j_refs: Sequence[GeneReference],
    config: PredictionConfig,
    relaxed: bool = False,
) -> List[GeneReference]:
    """Return J genes whose anchor is compatible with the CDR3 C-terminus."""
    k = max(2, config.c_anchor_match - (1 if relaxed else 0))
    survivors: List[GeneReference] = []
    for j in j_refs:
        window = min(k, len(cdr3), len(j.anchor))
        if relaxed or window == 0 or _has_match(cdr3[-window:], j.anchor[-window:]):
            survivors.append(j)
    return survivors


def chain_compatible(
    refs: Sequence[GeneReference], chain: str
) -> List[GeneReference]:
    """Restrict reference entries to those matching the requested chain."""
    chain = chain.upper()
    return [r for r in refs if r.chain == chain]
