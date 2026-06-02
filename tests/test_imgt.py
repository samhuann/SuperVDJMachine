"""Tests for IMGT FASTA loading and CDR3-anchor extraction."""

from __future__ import annotations

import textwrap

from supervdj.imgt import (
    _j_anchor,
    _translate,
    _v_anchor,
    load_imgt_reference,
    parse_fasta,
    records_to_references,
)


def _write_fasta(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    return p


def test_translate_skips_gaps_and_respects_codon_start():
    # codon_start = 3 means skip the first 2 nt to land on a codon boundary.
    # NT below decodes "NTEAFFGQGTRLTVV" for TRBJ1-1.
    nt = "tgaacactgaagctttctttggacaaggcaccagactcacagttgtag"
    protein = _translate(nt, codon_start=3)
    assert protein.startswith("NTEAFF")
    assert "FGQG" in protein  # FGXG motif present


def test_translate_handles_dots():
    nt = "ATG......AAATTT"
    protein = _translate(nt, codon_start=1)
    assert protein == "MKF"


def test_v_anchor_takes_suffix_from_last_C():
    assert _v_anchor("AAAYLCASSQR", max_len=6) == "CASSQR"
    assert _v_anchor("AAAYL") is None  # no Cys present


def test_j_anchor_takes_prefix_up_to_FGXG():
    assert _j_anchor("NTEAFFGQGTRLTVV") == "NTEAFF"
    # W-G-X-G also recognised
    assert _j_anchor("SDSNYQLIWGAGTKLI") == "SDSNYQLIW"
    assert _j_anchor("NOMATCH") is None


def test_parse_fasta_basic(tmp_path):
    p = _write_fasta(
        tmp_path,
        "mini.fasta",
        """
        >K02545|TRBJ1-1*01|Homo sapiens|F|J-REGION|749..796|48 nt|3| | | | |48+0=48| | |
        tgaacactgaagctttctttggacaaggcaccagactcacagttgtag
        >M14158|TRBJ1-3*01|Homo sapiens|F|J-REGION|1499..1548|50 nt|2| | | | |50+0=50| | |
        ctctggaaacaccatatattttggagagggaagttggctcactgttgtag
        """,
    )
    records = parse_fasta(p)
    assert len(records) == 2
    assert records[0].gene == "TRBJ1-1*01"
    assert records[0].functional == "F"
    assert records[0].region == "J-REGION"
    assert records[0].codon_start == 3
    assert records[0].protein.startswith("NTEAFF")


def test_records_to_references_collapses_alleles():
    from supervdj.imgt import ImgtRecord

    records = [
        ImgtRecord("X", "TRBV6-5*01", "Homo sapiens", "F", "V-REGION", 1,
                   "", "AAAYLCASSQR"),
        ImgtRecord("Y", "TRBV6-5*02", "Homo sapiens", "F", "V-REGION", 1,
                   "", "AAAYLCASSQR"),
        ImgtRecord("Z", "TRBV6-1*01", "Homo sapiens", "F", "V-REGION", 1,
                   "", "AAAYLCASSQK"),
    ]
    refs = records_to_references(records, chain="TRB", gene_type="V")
    names = sorted(r.gene for r in refs)
    assert names == ["TRBV6-1", "TRBV6-5"]


def test_records_to_references_uses_longest_v_anchor():
    from supervdj.imgt import ImgtRecord

    records = [
        ImgtRecord("X", "TRBV6-5*01", "Homo sapiens", "F", "V-REGION", 1,
                   "", "AAAYLCASS"),
        ImgtRecord("Y", "TRBV6-5*02", "Homo sapiens", "F", "V-REGION", 1,
                   "", "AAAYLCASSQR"),
    ]

    refs = records_to_references(records, chain="TRB", gene_type="V")

    assert len(refs) == 1
    assert refs[0].gene == "TRBV6-5"
    assert refs[0].anchor == "CASSQR"


def test_records_to_references_filters_pseudogenes_by_default():
    from supervdj.imgt import ImgtRecord

    records = [
        ImgtRecord("X", "TRBV1*01", "Homo sapiens", "P", "V-REGION", 1,
                   "", "AAAYLCTSSQR"),
        ImgtRecord("Y", "TRBV2*01", "Homo sapiens", "F", "V-REGION", 1,
                   "", "AAAYLCASSQR"),
    ]
    refs = records_to_references(records, chain="TRB", gene_type="V")
    assert {r.gene for r in refs} == {"TRBV2"}
    refs_all = records_to_references(
        records, chain="TRB", gene_type="V", include_nonfunctional=True
    )
    assert {r.gene for r in refs_all} == {"TRBV1", "TRBV2"}


def test_load_imgt_reference_packaged_trb_has_known_genes():
    v, j = load_imgt_reference("TRB")
    v_names = {r.gene for r in v}
    j_names = {r.gene for r in j}
    # A handful of well-known functional TRBV / TRBJ genes must be present.
    for g in ("TRBV6-5", "TRBV19", "TRBV20-1"):
        assert g in v_names
    for g in ("TRBJ2-7", "TRBJ1-1", "TRBJ2-1"):
        assert g in j_names
    # All anchors must look like CDR3-bounding motifs.
    assert all(r.anchor.startswith("C") for r in v)
    assert all(r.anchor.endswith("F") or r.anchor.endswith("W") for r in j)


def test_load_imgt_reference_packaged_tra_has_known_genes():
    v, j = load_imgt_reference("TRA")
    v_names = {r.gene for r in v}
    j_names = {r.gene for r in j}
    assert "TRAV1-1" in v_names
    assert "TRAJ33" in j_names
    assert all(r.anchor.startswith("C") for r in v)
    assert all(r.anchor.endswith("F") or r.anchor.endswith("W") for r in j)
