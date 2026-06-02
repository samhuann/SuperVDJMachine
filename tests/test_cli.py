"""Tests for the CLI and output formatters."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pandas as pd
import pytest

from supervdj.cli import main
from supervdj.models import PredictionConfig
from supervdj.output import OUTPUT_COLUMNS, render, result_to_dataframe
from supervdj.scoring import predict


def _run_cli(args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(args)
    return code, buf.getvalue()


def test_cli_predict_pretty_runs():
    code, out = _run_cli(
        [
            "predict",
            "--chain",
            "TRB",
            "--cdr3",
            "CASSIRSSYEQYF",
            "--top-k",
            "5",
            "--no-olga",
            "--no-sonia",
        ]
    )
    assert code == 0
    assert "confidence:" in out
    assert "TRBV" in out and "TRBJ" in out


def test_cli_predict_csv_to_file(tmp_path):
    out_path = tmp_path / "result.csv"
    code, _ = _run_cli(
        [
            "predict",
            "--chain",
            "TRB",
            "--cdr3",
            "CASSIRSSYEQYF",
            "--top-k",
            "5",
            "--format",
            "csv",
            "--out",
            str(out_path),
            "--no-olga",
            "--no-sonia",
        ]
    )
    assert code == 0
    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert list(df.columns) == OUTPUT_COLUMNS
    assert len(df) > 0
    assert df.iloc[0]["v_gene"].startswith("TRBV")


def test_cli_rejects_unknown_chain():
    with pytest.raises(SystemExit):
        main(["predict", "--chain", "TRG", "--cdr3", "CASSIRSSYEQYF"])


def test_result_to_dataframe_columns_match_spec():
    cfg = PredictionConfig(use_olga=False, use_sonia=False, top_k=3)
    result = predict("CASSIRSSYEQYF", "TRB", config=cfg)
    df = result_to_dataframe(result)
    assert list(df.columns) == OUTPUT_COLUMNS
    assert (df["rank"].tolist() == sorted(df["rank"].tolist()))


def test_render_json_is_parseable():
    cfg = PredictionConfig(use_olga=False, use_sonia=False, top_k=3)
    result = predict("CASSIRSSYEQYF", "TRB", config=cfg)
    text = render(result, fmt="json")
    parsed = json.loads(text)
    assert isinstance(parsed, list)
    assert parsed[0]["v_gene"].startswith("TRBV")


def test_render_tsv_has_tabs():
    cfg = PredictionConfig(use_olga=False, use_sonia=False, top_k=3)
    result = predict("CASSIRSSYEQYF", "TRB", config=cfg)
    text = render(result, fmt="tsv")
    assert "\t" in text.splitlines()[0]
