"""Tests for the supervised V-gene CNN model."""

from __future__ import annotations

import numpy as np
import pytest

from supervdj.models import PredictionConfig
from supervdj.scoring import predict
from supervdj.v_model import VGeneCNN


def _toy_rows():
    return [
        ("CASSIRSSYEQYF", "TRBV6-5"),
        ("CASSLGQETQYF", "TRBV6-5"),
        ("CASSQDPQYF", "TRBV7-2"),
        ("CASSLAPGATNEKLFF", "TRBV7-2"),
    ]


def test_cnn_encode_shape_and_padding():
    x = VGeneCNN._encode(["CASS", "CASSIRSSYEQYF"], maxlen=24)
    assert x.shape == (2, 24)
    assert x.dtype == np.int32
    assert x[0, 4] == 0
    assert x[1, 0] > 0


def test_cnn_ranks_trained_gene_and_round_trips(trbv_refs, trbj_refs, tmp_path):
    pytest.importorskip("tensorflow")
    model = VGeneCNN.from_rows(_toy_rows(), chain="TRB", epochs=20)
    genes = ["TRBV6-5", "TRBV7-2"]
    ranking = model.rank("CASSIRSSYEQYF", genes)
    assert ranking[0][0] in genes

    cfg = PredictionConfig(
        chain="TRB",
        use_olga=False,
        use_sonia=False,
        top_k=10,
        v_model=model,
    )
    result = predict(
        "CASSIRSSYEQYF",
        "TRB",
        config=cfg,
        v_refs=trbv_refs,
        j_refs=trbj_refs,
    )
    assert result.v_ranking

    model.save(tmp_path / "cnn")
    reloaded = VGeneCNN.load(tmp_path / "cnn")
    assert reloaded.classes == model.classes
