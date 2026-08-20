"""Tests for the supervdj pipeline.

Fast unit tests (cache, resolution, io) run without OLGA/SONNIA.  The OLGA and
SONNIA smoke tests exercise the real pretrained models on a couple of TRB
sequences and are skipped automatically if those packages are unavailable.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from supervdj.cache import ValueCache
from supervdj.io import load_dataset, valid_cdr3
from supervdj.resolution import (
    aggregate_posterior,
    family_of,
    gene_to_label,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic_smoke.tsv"

olga = pytest.importorskip("olga", reason="OLGA not installed")
try:
    import sonnia  # noqa: F401
    HAS_SONNIA = True
except Exception:
    HAS_SONNIA = False


# ---- fast unit tests -------------------------------------------------------

def test_cache_roundtrip(tmp_path):
    path = tmp_path / "c.pkl"
    c = ValueCache(path)
    k = c.key("pgen", "TRB", "CASSF", "TRBV19", None)
    assert c.get(k) is None
    c.set(k, 1.5)
    assert c.get(k) == 1.5
    c.save()
    c2 = ValueCache(path)
    assert c2.get(k) == 1.5


def test_family_and_group_resolution():
    assert family_of("TRBV7-2") == "TRBV7"
    assert family_of("TRBJ2-7") == "TRBJ2"
    assert family_of("TRAV13-1") == "TRAV13"
    post = {"TRBV7-2": 0.5, "TRBV7-9": 0.3, "TRBV20-1": 0.2}
    fam = aggregate_posterior(post, "family")
    assert math.isclose(fam["TRBV7"], 0.8)
    assert math.isclose(fam["TRBV20"], 0.2)
    mapping = {"TRBV7-2": "G1", "TRBV7-9": "G1"}
    grp = aggregate_posterior(post, "group", mapping, unmapped="self")
    assert math.isclose(grp["G1"], 0.8)
    assert math.isclose(grp["TRBV20-1"], 0.2)  # unmapped -> singleton
    assert gene_to_label("TRBV7-2", "group", mapping) == "G1"


def test_load_dataset():
    rows = load_dataset(DATA)
    assert len(rows) == 6
    trb = [r for r in rows if r.chain == "TRB"]
    assert all(r.true_v.startswith("TRBV") for r in trb)
    assert all("*" not in r.true_v for r in rows)  # alleles stripped
    assert valid_cdr3("CASSIRSSYEQYF") and not valid_cdr3("CASS1F")


# ---- real-model smoke tests ------------------------------------------------

def test_olga_preselection_posterior():
    from supervdj.analyze import analyze_sequence
    from supervdj.models import load_chain_models

    models = load_chain_models("TRB", use_sonia=False)
    assert "TRBV19" in models.v_genes and "TRBJ2-7" in models.j_genes
    cache = ValueCache(None)
    rows = [r for r in load_dataset(DATA) if r.cdr3 == "CASSIRSSYEQYF"]
    recs = analyze_sequence(models, cache, rows[0], post_modes=(),
                            resolutions=("gene", "family"))
    pre_v_gene = [r for r in recs if r["axis"] == "V" and r["model"] == "pre"
                  and r["resolution"] == "gene"][0]
    assert pre_v_gene["status"] == "ok"
    assert pre_v_gene["true_rank"] is not None        # TRBV19 is a candidate
    assert 0.0 < pre_v_gene["top1_mass"] <= 1.0
    assert pre_v_gene["entropy_nats"] >= 0.0
    # both resolutions present for V-pre
    res = {r["resolution"] for r in recs if r["axis"] == "V" and r["model"] == "pre"}
    assert res == {"gene", "family"}


@pytest.mark.skipif(not HAS_SONNIA, reason="sonnia not installed")
def test_sonnia_postselection_fixed():
    from supervdj.analyze import analyze_sequence
    from supervdj.models import load_chain_models

    models = load_chain_models("TRB", use_sonia=True)
    cache = ValueCache(None)
    rows = [r for r in load_dataset(DATA) if r.cdr3 == "CASSIRSSYEQYF"]
    recs = analyze_sequence(models, cache, rows[0], post_modes=("fixed",),
                            resolutions=("gene",))
    post_v = [r for r in recs if r["axis"] == "V" and r["model"] == "post"][0]
    assert post_v["status"] == "ok"
    assert post_v["true_rank"] is not None
    assert 0.0 < post_v["top1_mass"] <= 1.0
