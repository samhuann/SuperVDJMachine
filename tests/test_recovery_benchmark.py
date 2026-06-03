"""Tests for manuscript recovery benchmark helpers."""

from __future__ import annotations

import inspect

from scripts import benchmark_vdjdb_recovery as bench


def test_strip_allele():
    assert bench.strip_allele("TRBV7-2*01") == "TRBV7-2"
    assert bench.strip_allele(" traj33*02 ") == "TRAJ33"
    assert bench.strip_allele("") == ""
    assert bench.strip_allele(None) == ""


def test_gene_family():
    assert bench.gene_family("TRAV12-2") == "TRAV12"
    assert bench.gene_family("TRAV12-3*01") == "TRAV12"
    assert bench.gene_family("TRBV5-1") == "TRBV5"
    assert bench.gene_family("TRBV5-6") == "TRBV5"
    assert bench.gene_family("TRBV20-1") == "TRBV20"
    assert bench.gene_family("TRAV1-1") == "TRAV1"
    assert bench.gene_family("TRBJ2-1") == "TRBJ2"
    assert bench.gene_family("TRBJ2-7") == "TRBJ2"
    assert bench.gene_family("TRAJ33") == "TRAJ33"


def test_topk_recovery_calculation():
    ranks = [1, 2, 5, None]
    assert bench.topk_recovery(ranks, 1) == 0.25
    assert bench.topk_recovery(ranks, 3) == 0.5
    assert bench.topk_recovery(ranks, 5) == 0.75


def test_mrr_calculation():
    ranks = [1, 2, 4, None]
    expected = (1.0 + 0.5 + 0.25 + 0.0) / 4
    assert bench.mrr(ranks) == expected


def test_match_helpers():
    assert bench.exact_gene_match("TRBV7-2*01", "TRBV7-2")
    assert not bench.exact_gene_match("TRBV7-2", "TRBV7-3")
    assert bench.family_match("TRBV7-2", "TRBV7-3")
    assert bench.family_match("TRBJ2-1", "TRBJ2-7")
    assert not bench.family_match("TRBJ1-1", "TRBJ2-7")


def test_vdjdb_evaluation_does_not_use_vdjdb_priors():
    signature = inspect.signature(bench.load_eval_references)
    assert list(signature.parameters) == ["chain"]
    source = inspect.getsource(bench.load_eval_references)
    assert "load_vdjdb" not in source
    assert "gene_usage" not in source
    assert "VdjdbRow" not in source
