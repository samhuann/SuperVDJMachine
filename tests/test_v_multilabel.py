"""Tests for the real-data linear V-gene model utility."""

from __future__ import annotations

from pathlib import Path

from supervdj.v_multilabel import (
    LinearVGeneModel,
    load_neotcr_v_rows,
    mean_reciprocal_rank,
    split_rows,
    topk_recall,
)


NEOTCR = Path("supervdj/data/NeoTCR.xlsx")


def test_load_neotcr_v_rows_real_data():
    rows = load_neotcr_v_rows(NEOTCR, "TRB")
    assert len(rows) > 100
    cdr3, labels = rows[0]
    assert cdr3
    assert labels
    assert "*" not in labels[0]


def test_linear_v_model_trains_on_neotcr_rows():
    rows = load_neotcr_v_rows(NEOTCR, "TRB")[:120]
    train, test = split_rows(rows, test_fraction=0.2, seed=1)
    model = LinearVGeneModel.from_rows(train, chain="TRB", min_label_count=1, max_iter=200)
    assert model.expand_genes("CASSLVSPSEQFF", [])[:1]
    assert 0.0 <= topk_recall(model, test, 10) <= 1.0
    assert 0.0 <= mean_reciprocal_rank(model, test) <= 1.0
