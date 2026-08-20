"""Tests for the calibrated candidate-set utility.

The lookups and set-building logic run without OLGA; the end-to-end test needs the
pretrained models and is skipped if they are unavailable.
"""

from __future__ import annotations

import pytest

from supervdj.candidates import (
    candidate_set,
    coverage_for_mass,
    group_breakdown,
    mass_for_coverage,
)


def test_candidate_set_is_smallest_reaching_target():
    post = {"A": 0.5, "B": 0.25, "C": 0.15, "D": 0.1}
    genes, mass = candidate_set(post, 0.7)
    assert [g for g, _ in genes] == ["A", "B"]      # 0.5 + 0.25 = 0.75 >= 0.7
    assert mass == pytest.approx(0.75)
    genes, mass = candidate_set(post, 1.0)
    assert len(genes) == 4 and mass == pytest.approx(1.0)


def test_calibration_lookup_is_monotone_and_inverts():
    for chain in ("TRA", "TRB"):
        masses = [mass_for_coverage(chain, c)[0] for c in (0.5, 0.7, 0.9)]
        assert masses == sorted(masses)
        # a mass read back through the curve returns roughly the coverage asked for
        m, reachable = mass_for_coverage(chain, 0.8)
        assert reachable
        assert coverage_for_mass(chain, m) == pytest.approx(0.8, abs=0.01)


def test_unreachable_coverage_is_flagged_not_faked():
    mass, reachable = mass_for_coverage("TRA", 0.999)
    assert not reachable
    assert mass <= 1.0


def test_alpha_posterior_is_overconfident():
    """The published finding the calibration exists for: nominal mass overstates
    coverage for TRA, while TRB is close to nominal."""
    assert coverage_for_mass("TRA", 0.9) < 0.85
    assert coverage_for_mass("TRB", 0.9) == pytest.approx(0.9, abs=0.02)


def test_group_breakdown_conserves_mass():
    genes = [("TRBV5-1", 0.3), ("TRBV7-9", 0.2), ("TRBV27", 0.1)]
    groups = group_breakdown("TRB", genes)
    assert sum(g["mass"] for g in groups) == pytest.approx(0.6)
    assert all(set(g["genes"]) <= {g0 for g0, _ in genes} for g in groups)


def test_end_to_end_candidate_set():
    pytest.importorskip("olga")
    from supervdj.candidates import calibrated_candidates

    r = calibrated_candidates("CASSLGQAYEQYF", "TRB", coverage=0.9)
    assert r["status"] == "ok"
    assert r["achieved_mass"] >= r["nominal_mass"]
    assert r["n_candidates"] == len(r["genes"])
    assert sum(g["mass"] for g in r["groups"]) == pytest.approx(r["achieved_mass"], abs=1e-4)
    assert r["expected_coverage"] == pytest.approx(0.9, abs=0.02)
