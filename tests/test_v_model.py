"""Tests for the supervised V-gene CNN model."""

from __future__ import annotations

import numpy as np
import pytest

from supervdj.models import PredictionConfig
from supervdj.scoring import predict
from supervdj.v_model import VGeneCNN


def test_encode_is_tf_free_and_padded():
    x = VGeneCNN._encode(["CASS", "CASSIRSSYEQYF"], maxlen=24)
    assert x.shape == (2, 24)
    assert x[0, 0] != 0          # 'C' encoded
    assert x[0, 4] == 0          # padding after a length-4 sequence
    assert np.all(x[:, -1] == 0)  # both sequences shorter than maxlen


def _toy_rows(n=40):
    pairs = [
        ("CASSLAPGATNEKLFF", "TRBV7-9"),
        ("CASSLGQAYEQYF", "TRBV5-1"),
        ("CASSIRSSYEQYF", "TRBV20-1"),
    ]
    return pairs * n


def test_cnn_ranks_trained_gene_and_round_trips(trbv_refs, trbj_refs, tmp_path):
    pytest.importorskip("tensorflow")
    model = VGeneCNN.from_rows(_toy_rows(), chain="TRB", epochs=20)
    genes = ["TRBV20-1", "TRBV5-1", "TRBV7-9"]

    # the model's own V discrimination recovers the trained label
    assert model.rank("CASSIRSSYEQYF", genes)[0][0] == "TRBV20-1"

    # integrates into predict() without error and surfaces the gene
    cfg = PredictionConfig(
        chain="TRB", use_olga=False, use_sonia=False, top_k=10, v_model=model,
    )
    result = predict(
        "CASSIRSSYEQYF", "TRB", config=cfg, v_refs=trbv_refs, j_refs=trbj_refs,
    )
    assert "TRBV20-1" in [g for g, _ in result.v_ranking]

    # save / load round-trip preserves scores
    model.save(tmp_path / "cnn")
    reloaded = VGeneCNN.load(tmp_path / "cnn")
    assert model.rank("CASSIRSSYEQYF", genes) == reloaded.rank("CASSIRSSYEQYF", genes)
