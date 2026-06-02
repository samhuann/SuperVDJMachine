"""Tests for filters and CDR3 validation."""

from __future__ import annotations

from supervdj.filters import (
    chain_compatible,
    filter_j_genes,
    filter_v_genes,
    normalize_cdr3,
    validate_cdr3,
)
from supervdj.models import PredictionConfig


def test_normalize_cdr3_uppercases_and_strips():
    assert normalize_cdr3(" cassirssyEqyf ") == "CASSIRSSYEQYF"


def test_validate_cdr3_canonical_pass():
    issues = validate_cdr3("CASSIRSSYEQYF", PredictionConfig())
    assert issues == []


def test_validate_cdr3_flags_non_canonical_ends():
    issues = validate_cdr3("ASSIRSSYEQYX", PredictionConfig())
    assert any("Cys" in i for i in issues)
    assert any("Phe/Trp" in i for i in issues)


def test_validate_cdr3_flags_bad_residues():
    issues = validate_cdr3("CASS*IRSSYEQYF", PredictionConfig())
    assert any("non-standard" in i for i in issues)


def test_validate_cdr3_too_short():
    issues = validate_cdr3("CASF", PredictionConfig())
    assert any("shorter" in i for i in issues)


def test_filter_v_genes_keeps_anchor_compatible(trbv_refs):
    # Strict filter is intentionally soft (at least one residue match in
    # the window). Hard exclusion only kicks in when zero positions match.
    cdr3 = "CASSIRSSYEQYF"
    cfg = PredictionConfig()
    survivors = filter_v_genes(cdr3, trbv_refs, cfg)
    names = {v.gene for v in survivors}
    assert "TRBV6-5" in names  # CASS perfect match
    # Now a CDR3 with no residue in common with any V anchor window:
    survivors_none = filter_v_genes("DDDDIRSSYEQYF", trbv_refs, cfg)
    assert survivors_none == []


def test_filter_j_genes_keeps_anchor_compatible(trbj_refs):
    cdr3 = "CASSIRSSYEQYF"
    cfg = PredictionConfig()
    survivors = filter_j_genes(cdr3, trbj_refs, cfg)
    names = {j.gene for j in survivors}
    assert "TRBJ2-7" in names  # ...EQYF perfect match
    # CDR3 whose C-terminus has zero overlap with any J anchor window:
    survivors_none = filter_j_genes("CASSIRSSSDDDD", trbj_refs, cfg)
    assert survivors_none == []


def test_relaxed_filter_recovers_candidates(trbv_refs):
    cdr3 = "CWWWIRSSYEQYF"  # nothing matches well
    cfg = PredictionConfig()
    strict = filter_v_genes(cdr3, trbv_refs, cfg)
    relaxed = filter_v_genes(cdr3, trbv_refs, cfg, relaxed=True)
    assert len(relaxed) >= len(strict)


def test_chain_compatible_filters_by_chain(trbv_refs, trav_refs):
    mixed = list(trbv_refs) + list(trav_refs)
    assert all(r.chain == "TRB" for r in chain_compatible(mixed, "TRB"))
    assert all(r.chain == "TRA" for r in chain_compatible(mixed, "TRA"))
