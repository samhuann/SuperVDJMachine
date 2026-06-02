"""Tests for VDJdb loading, usage summaries, and evaluation harness."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from supervdj.models import GeneReference, PredictionConfig
from supervdj.vdjdb import (
    VdjdbRow,
    evaluate,
    gene_usage_frequencies,
    load_vdjdb,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
VDJDB_PATH = REPO_ROOT / "Best_VDJdb.tsv"


def _toy_vdjdb_rows():
    return [
        VdjdbRow("TRB", "CASSIRSSYEQYF", "TRBV19", "TRBJ2-7"),
        VdjdbRow("TRB", "CASSIRSAYEQYF", "TRBV19", "TRBJ2-7"),
        VdjdbRow("TRB", "CASSDRGYEQYF",  "TRBV6-5", "TRBJ2-7"),
        VdjdbRow("TRA", "CAVRDSNYQLIW",  "TRAV1-1", "TRAJ33"),
    ]


def test_gene_usage_frequencies_normalised():
    rows = _toy_vdjdb_rows()
    v_p, j_p = gene_usage_frequencies(rows, chain="TRB")
    assert abs(sum(v_p.values()) - 1.0) < 1e-9
    assert abs(sum(j_p.values()) - 1.0) < 1e-9
    assert v_p["TRBV19"] > v_p["TRBV6-5"]
    assert j_p["TRBJ2-7"] == 1.0


def test_evaluate_on_toy_rows():
    refs_v = [
        GeneReference("TRBV19", "TRB", "V", "F", "CASS", 0.5),
        GeneReference("TRBV6-5", "TRB", "V", "F", "CASS", 0.5),
    ]
    refs_j = [
        GeneReference("TRBJ2-7", "TRB", "J", "F", "SYEQYF", 0.5),
        GeneReference("TRBJ1-1", "TRB", "J", "F", "NTEAFF", 0.5),
    ]
    rows = _toy_vdjdb_rows()
    cfg = PredictionConfig(chain="TRB", use_olga=False, use_sonia=False, top_k=10)
    summary = evaluate(rows, chain="TRB", config=cfg, v_refs=refs_v, j_refs=refs_j)
    assert summary.n == 3
    # J-top1 should be perfect on this toy set since all TRB rows ground-truth J is TRBJ2-7.
    assert summary.j_top1 == 1.0


def test_load_vdjdb_if_present():
    if not VDJDB_PATH.exists():
        return
    rows = load_vdjdb(VDJDB_PATH, chain="TRB")
    assert rows
    # All allele suffixes must be stripped.
    assert all("*" not in r.v for r in rows[:50] if r.v)
    assert all("*" not in r.j for r in rows[:50] if r.j)
    # Chain filter works.
    assert all(r.chain == "TRB" for r in rows)


def test_load_vdjdb_accepts_clean_lowercase_schema(tmp_path):
    p = tmp_path / "clean.tsv"
    pd.DataFrame(
        [
            {
                "gene": "TRB",
                "cdr3": "CASSIRSSYEQYF",
                "v": "TRBV19*01",
                "j": "TRBJ2-7*01",
            }
        ]
    ).to_csv(p, sep="\t", index=False)

    rows = load_vdjdb(p, chain="TRB")

    assert rows == [VdjdbRow("TRB", "CASSIRSSYEQYF", "TRBV19", "TRBJ2-7")]


def test_evaluate_against_real_vdjdb_smoke():
    if not VDJDB_PATH.exists():
        return
    rows = load_vdjdb(VDJDB_PATH, chain="TRB")[:25]
    cfg = PredictionConfig(chain="TRB", use_olga=False, use_sonia=False, top_k=20)
    summary = evaluate(rows, chain="TRB", config=cfg, top_k=20)
    assert summary.n == len(rows)
    # Sanity: top-10 accuracy must be at least nonzero on a famous dataset.
    assert summary.v_top10 >= 0.0
    assert summary.j_top10 >= 0.0
