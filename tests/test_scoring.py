"""Tests for scoring pipeline and softmax behavior."""

from __future__ import annotations

import math

from supervdj.models import PredictionConfig
from supervdj.motif import boundary_penalty, j_motif_score, v_motif_score
from supervdj.scoring import _stable_softmax, predict


def test_v_motif_score_rewards_match(trbv_refs):
    cfg = PredictionConfig()
    matched, _ = v_motif_score("CASSIRSSYEQYF", trbv_refs[0], cfg)  # TRBV6-5 (CASS)
    mismatched, _ = v_motif_score("CWWWIRSSYEQYF", trbv_refs[0], cfg)
    assert matched > mismatched


def test_j_motif_score_rewards_match(trbj_refs):
    cfg = PredictionConfig()
    good, _ = j_motif_score("CASSIRSSYEQYF", trbj_refs[0], cfg)  # TRBJ2-7 (SYEQYF)
    bad, _ = j_motif_score("CASSIRSSYEAFF", trbj_refs[0], cfg)
    assert good > bad


def test_stable_softmax_sums_to_one():
    out = _stable_softmax([1.0, 2.0, 3.0, 1000.0])
    assert math.isclose(sum(out), 1.0, abs_tol=1e-9)
    assert max(out) == out[-1]


def test_stable_softmax_handles_empty():
    assert _stable_softmax([]) == []


def test_stable_softmax_handles_all_negative_infinity():
    out = _stable_softmax([-math.inf, -math.inf])
    assert out == [0.5, 0.5]


def test_boundary_weight_scales_penalty(trbv_refs, trbj_refs):
    cdr3 = "CASSIRSS"
    base = PredictionConfig(boundary_weight=1.0)
    off = PredictionConfig(boundary_weight=0.0)
    assert boundary_penalty(cdr3, trbv_refs[0], trbj_refs[0], off) == 0.0
    assert boundary_penalty(cdr3, trbv_refs[0], trbj_refs[0], base) >= 0.0


def test_predict_ranks_known_pair_top(trbv_refs, trbj_refs):
    cfg = PredictionConfig(use_olga=False, use_sonia=False, top_k=10)
    result = predict(
        "CASSIRSSYEQYF",
        "TRB",
        config=cfg,
        v_refs=trbv_refs,
        j_refs=trbj_refs,
    )
    assert result.candidates
    top = result.candidates[0]
    assert top.v_gene == "TRBV6-5"
    assert top.j_gene == "TRBJ2-7"


def test_predict_posteriors_sum_to_one(trbv_refs, trbj_refs):
    cfg = PredictionConfig(use_olga=False, use_sonia=False, top_k=100)
    result = predict(
        "CASSIRSSYEQYF",
        "TRB",
        config=cfg,
        v_refs=trbv_refs,
        j_refs=trbj_refs,
    )
    total = sum(c.posterior_probability for c in result.candidates)
    assert math.isclose(total, 1.0, abs_tol=1e-6)


def test_predict_tra(trav_refs, traj_refs):
    cfg = PredictionConfig(use_olga=False, use_sonia=False)
    result = predict(
        "CAVRDSNYQLIW",
        "TRA",
        config=cfg,
        v_refs=trav_refs,
        j_refs=traj_refs,
    )
    assert result.candidates
    top = result.candidates[0]
    assert top.v_gene == "TRAV1-1"
    assert top.j_gene == "TRAJ33"


def test_predict_relaxes_when_no_survivors(trbv_refs, trbj_refs):
    cfg = PredictionConfig(use_olga=False, use_sonia=False)
    # CDR3 that doesn't anchor-match any V or J in the toy set:
    result = predict(
        "CDDDDDDDDDDDDD",
        "TRB",
        config=cfg,
        v_refs=trbv_refs,
        j_refs=trbj_refs,
    )
    # We accept either: candidates returned after relaxation, or empty +
    # 'No candidates' warning. Both states must report low confidence.
    assert result.confidence == "low"
    assert any("relax" in w.lower() or "No candidates" in w for w in result.warnings)


def test_olga_wrapper_normalizes_allele_suffix():
    from supervdj.olga_wrapper import _normalize

    assert _normalize("TRBV19*01") == "TRBV19"
    assert _normalize("  trbv6-5*02 ") == "TRBV6-5"
    assert _normalize("TRAV14/DV4*01") == "TRAV14/DV4"


def test_sonia_wrapper_normalizes_allele_suffix():
    from supervdj.sonia_wrapper import _normalize

    assert _normalize("TRBJ2-7*01") == "TRBJ2-7"
    assert _normalize("traj33*02") == "TRAJ33"


def test_predict_with_external_integrations_returns_ranked_candidates(
    trbv_refs, trbj_refs
):
    # OLGA and SONIA are required runtime deps. Whether or not their default
    # model files are present in the test env, the wrapper degrades to
    # neutral scores with a warning rather than crashing — so predict() must
    # always return ranked candidates and a `warnings` field that is a list.
    cfg = PredictionConfig(use_olga=True, use_sonia=True)
    result = predict(
        "CASSIRSSYEQYF",
        "TRB",
        config=cfg,
        v_refs=trbv_refs,
        j_refs=trbj_refs,
    )
    assert result.candidates
    assert isinstance(result.warnings, list)
    # Posteriors must remain a valid distribution.
    assert abs(sum(c.posterior_probability for c in result.candidates) - 1.0) < 1e-6
