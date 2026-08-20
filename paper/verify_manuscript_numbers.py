"""Recompute every number quoted in the manuscript from the committed inputs and
compare against what the text says. Prints one row per claim; exits non-zero if any
claim fails.

  PYTHONPATH=. python scripts/verify_manuscript_numbers.py            # deterministic claims
  PYTHONPATH=. python scripts/verify_manuscript_numbers.py --resample # + rerun the bootstraps

Inputs: results/posteriors.tsv, results/ingest/*.tsv, results/supp_tables/*.csv, and the
resampling artifacts (results/permutation_test_TRA_TRB.json, results/alpha_bootstrap.json,
results/matched_n_cross_chain.json). --resample recomputes the artifacts' contents in
place of trusting them, which takes ~25 min.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import supervdj.aggregate as A
from supervdj.resolution import family_of

ROWS = []


def check(claim, quoted, computed, tol=None, fmt="{}"):
    """tol=None -> exact equality; else |quoted - computed| <= tol."""
    ok = (quoted == computed) if tol is None else (abs(quoted - computed) <= tol)
    ROWS.append((claim, fmt.format(quoted), fmt.format(computed), "PASS" if ok else "FAIL"))
    return ok


def skip(claim, why):
    """Record a claim that could not be evaluated here, keeping the row count fixed so
    the reported total does not depend on which inputs happen to be present."""
    ROWS.append((f"{claim} [{why}]", "-", "-", "SKIP"))


# --------------------------------------------------------------------------- #
# ingest: raw -> canonical bookkeeping  (main.tex "Data and preprocessing")
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# inputs: everything this script reads, so a bare clone gets one clear message
# --------------------------------------------------------------------------- #
ZENODO = "https://doi.org/10.5281/zenodo.22024426"       # deposited analysis record

ARTIFACTS = [
    "results/ingest/canonical_all.tsv",
    "results/ingest/canonical_unique.tsv",
    "results/ingest/canonical_alpha_pooled.tsv",
    "results/ingest/overlap_pairwise.tsv",
    "results/ingest/rejected.tsv",
    "results/matched_n_cross_chain.json",
    "results/permutation_test_TRA_TRB.json",
    "results/alpha_bootstrap.json",
    "results/figures/prism",
    "results/supp_tables/tableS2_source_robustness.csv",
    "results/supp_tables/tableS4_redistribution.csv",
    "results/supp_tables/tableS4_redistribution_accounting.csv",
    "results/supp_tables/tableS5_dropped_weight.csv",
    "results/supp_tables/tableS6_heldout_cohorts.csv",
    "results/validation_cohort/cohort_metrics.tsv",
    "results/validation_cohort/held_out_calibration.json",
    "results/validation_cohort/usage_reweight.json",
]


def preflight(posteriors):
    """Report every missing input at once instead of dying on the first read."""
    missing = [p for p in [posteriors, *ARTIFACTS] if not os.path.exists(p)]
    if not missing:
        return
    print("Cannot verify: the following inputs are missing.\n", file=sys.stderr)
    for p in missing:
        print(f"  {p}", file=sys.stderr)
    print(f"\nAll of them are in the deposited analysis record: {ZENODO}\n"
          "Download it, unpack it so that results/ sits at the repository root, and "
          "run this again. If the posterior table is somewhere else, pass "
          "--posteriors /path/to/posteriors.tsv.", file=sys.stderr)
    raise SystemExit(2)


def check_ingest():
    # beta (and the original McPAS-only alpha) live in canonical_unique/all; the pooled
    # alpha set (McPAS + VDJdb + NeoTCR) is canonical_alpha_pooled, added later.
    allrows = pd.read_csv("results/ingest/canonical_all.tsv", sep="\t")
    uniq = pd.read_csv("results/ingest/canonical_unique.tsv", sep="\t")
    pool = pd.read_csv("results/ingest/canonical_alpha_pooled.tsv", sep="\t")
    rej = pd.read_csv("results/ingest/rejected.tsv", sep="\t")

    # canonical_all/canonical_unique stay pinned to the beta+McPAS-alpha ingest whose row
    # order fixes the beta seq_ids; rejected.tsv and overlap_pairwise.tsv cover all six
    # sources (scripts/regen_ingest.sh).
    per_source = allrows.source.value_counts()
    for src, quoted in [("vdjdb_beta", 63237), ("mcpas_beta", 17666), ("neotcr_beta", 727)]:
        check(f"intra-source dedup tuples, {src}", quoted, int(per_source[src]))
    member = pool.sources.str.split(";")
    for src, quoted in [("mcpas_alpha", 6118), ("vdjdb_alpha", 33081), ("neotcr_alpha", 267)]:
        check(f"intra-source dedup tuples, {src}", quoted,
              int(member.apply(lambda L: src in L).sum()))

    n_beta = int((uniq.chain == "TRB").sum())
    check("unique TRB rearrangements", 80961, n_beta)
    check("unique TRA rearrangements", 38971, int(len(pool)))
    check("unique rearrangements, both chains", 119932, n_beta + int(len(pool)))

    check("TRB tuples in >1 source", 665, int(((uniq.chain == "TRB") & (uniq.n_sources > 1)).sum()))
    check("TRA tuples in >1 source", 494, int((pool.n_sources > 1).sum()))

    ov = pd.read_csv("results/ingest/overlap_pairwise.tsv", sep="\t").set_index(["source_a", "source_b"])
    for a, b, quoted in [("mcpas_beta", "vdjdb_beta", 553), ("neotcr_beta", "vdjdb_beta", 94),
                         ("mcpas_beta", "neotcr_beta", 26), ("mcpas_alpha", "vdjdb_alpha", 403),
                         ("neotcr_alpha", "vdjdb_alpha", 91), ("mcpas_alpha", "neotcr_alpha", 2)]:
        check(f"tuples shared, {a} n {b}", quoted, int(ov.loc[(a, b), "shared_tuples"]))

    check("raw records rejected", 6134, int(len(rej)))
    per_rej = rej.source.value_counts()
    for src, quoted in [("mcpas_beta", 2429), ("mcpas_alpha", 2277), ("neotcr_alpha", 698),
                        ("neotcr_beta", 229), ("vdjdb_alpha", 410), ("vdjdb_beta", 91)]:
        check(f"rejected, {src}", quoted, int(per_rej.get(src, 0)))
    reason = rej.reason.astype(str)
    check("rejected for unmapped V", 4936, int(reason.str.startswith("v_unmapped").sum()))
    check("rejected for unmapped J", 182, int(reason.str.startswith("j_unmapped").sum()))
    check("rejected for CDR3 boundary", 862, int(reason.str.startswith("boundary").sum()))
    check("rejected for non-standard residues", 154,
          int(reason.str.startswith("cdr3:nonstandard_residue").sum()))


# --------------------------------------------------------------------------- #
# canonical set, entropies, curves  (Results: recoverability / top-k / confidence)
# --------------------------------------------------------------------------- #
def check_canonical(df):
    ns = A._canonical_ns(df)
    check("canonical N, TRA", 37687, ns["TRA"])
    check("canonical N, TRB", 80409, ns["TRB"])
    uniq = pd.read_csv("results/ingest/canonical_unique.tsv", sep="\t")
    pool = pd.read_csv("results/ingest/canonical_alpha_pooled.tsv", sep="\t")
    check("zero-Pgen sequences removed, TRA", 1284, int(len(pool)) - ns["TRA"])
    check("zero-Pgen sequences removed, TRB", 552, int((uniq.chain == "TRB").sum()) - ns["TRB"])


def check_entropies(df):
    def mean_H(chain, axis, resolution="gene"):
        return float(A._slice(df, chain=chain, axis=axis, model="pre",
                              resolution=resolution)["entropy_nats"].mean())

    check("mean V-gene entropy, TRA", 1.2925, mean_H("TRA", "V"), tol=5e-5, fmt="{:.4f}")
    check("mean V-gene entropy, TRB", 2.7055, mean_H("TRB", "V"), tol=5e-5, fmt="{:.4f}")
    check("mean J-gene entropy, TRA", 0.0115, mean_H("TRA", "J"), tol=5e-5, fmt="{:.4f}")
    check("mean J-gene entropy, TRB", 0.0474, mean_H("TRB", "J"), tol=5e-5, fmt="{:.4f}")
    check("mean V-family entropy, TRA", 1.1848, mean_H("TRA", "V", "family"), tol=5e-5, fmt="{:.4f}")


def check_topk(df):
    for chain, k, quoted in [("TRA", 1, 0.49), ("TRA", 10, 0.85), ("TRA", 20, 0.92),
                             ("TRB", 1, 0.20), ("TRB", 10, 0.61), ("TRB", 20, 0.88)]:
        r = A._slice(df, chain=chain, axis="V", model="pre", resolution="gene")["true_rank"].dropna()
        check(f"top-{k} V coverage, {chain}", quoted, float((r <= k).mean()), tol=5e-3, fmt="{:.3f}")


def check_confidence(df):
    def frac(chain, res, thr):
        m = A._slice(df, chain=chain, axis="V", model="pre", resolution=res)["top1_mass"].dropna()
        return float((m >= thr).mean())

    for chain, res, thr, quoted in [("TRB", "gene", 0.5, 0.13), ("TRB", "gene", 0.99, 0.04),
                                    ("TRA", "gene", 0.5, 0.57), ("TRA", "gene", 0.99, 0.13),
                                    ("TRB", "family", 0.5, 0.19)]:
        check(f"high-confidence fraction, {chain} {res} @ {thr}", quoted, frac(chain, res, thr),
              tol=5e-3, fmt="{:.3f}")


def check_selfmass(df):
    """Genes whose mean posterior self-mass (confusion diagonal) is >= 0.5 -- fig S3."""
    for chain, quoted in [("TRA", 18), ("TRB", 7)]:
        keep, M = A.build_leakage(df, chain, 20)
        check(f"genes with self-mass >= 0.5, {chain}", quoted, int((np.diag(M) >= 0.5).sum()))


def check_gene_sets(df):
    """Model candidate-set sizes after collapsing alleles: union of posterior keys.

    Also checks the Methods claim that every analyzed sequence's annotated gene is a
    candidate the posterior ranges over -- the ground truth is never scored against a
    label the model could not have produced.
    """
    missing = 0
    for chain, axis, quoted in [("TRA", "V", 47), ("TRA", "J", 61), ("TRB", "V", 59), ("TRB", "J", 13)]:
        sl = A._slice(df, chain=chain, axis=axis, model="pre", resolution="gene")
        genes = set()
        for true, js in zip(sl["true_gene"].astype(str), sl["posterior_json"]):
            post = json.loads(js)
            genes.update(post.keys())
            missing += true not in post
        check(f"{axis} genes in the OLGA {chain} model", quoted, len(genes))
    check("annotated genes outside the posterior support", 0, missing)


def check_mcpas_null(df):
    """Methods: the estimator's finite-sample bias at the McPAS-only n (-0.0071)."""
    pool = pd.read_csv("results/ingest/canonical_alpha_pooled.tsv", sep="\t")
    ids = set(pool.loc[pool.sources.str.split(";").apply(lambda L: "mcpas_alpha" in L), "seq_id"])
    f = A._selection_frame(df[df.seq_id.isin(ids)], "TRA")
    _, null = A._permutation_null(f, n_perm=2000, seed=0)
    check("permutation-null center, McPAS-only alpha", -0.0071, float(null.mean()), tol=5e-5, fmt="{:+.4f}")


def check_ari(df):
    g = A.grouping_comparison(df, chains=("TRA", "TRB"), min_count=20)
    check("grouping ARI vs IMGT families, TRA", 0.0519, float(g["TRA"]["ari"]), tol=5e-5, fmt="{:.4f}")
    check("grouping ARI vs IMGT families, TRB", 0.214, float(g["TRB"]["ari"]), tol=5e-4, fmt="{:.4f}")


# --------------------------------------------------------------------------- #
# selection: shifts, redistribution, dropped weight
# --------------------------------------------------------------------------- #
def check_selection(df):
    res = A.selection_comparison(df, "gene", verbose=False)
    check("usage-controlled shift, TRA", -0.063, res["TRA"]["ctrl_shift"], tol=5e-4, fmt="{:+.4f}")
    check("usage-controlled shift, TRB", -0.281, res["TRB"]["ctrl_shift"], tol=5e-4, fmt="{:+.4f}")
    check("raw shift, TRA", 0.0054, res["TRA"]["raw_shift"], tol=5e-5, fmt="{:+.4f}")
    perm = json.load(open("results/permutation_test_TRA_TRB.json"))
    check("bias-corrected shift, TRA", -0.062,
          res["TRA"]["ctrl_shift"] - perm["TRA"]["null_mean"], tol=5e-4, fmt="{:+.4f}")
    check("dropped post weight, TRB", 0.072, 1 - res["TRB"]["frac_post_kept"], tol=5e-4, fmt="{:.4f}")
    check("dropped post weight, TRA", 0.0015, 1 - res["TRA"]["frac_post_kept"], tol=5e-4, fmt="{:.4f}")


def check_redistribution(df):
    tb, n_trb = A._redistribution_table(df, "TRB")
    ta, _ = A._redistribution_table(df, "TRA")
    tb, ta = tb.set_index("gene"), ta.set_index("gene")

    elim = tb[(tb.pre_top1 > 0) & (tb.post_top1 == 0)]
    check("TRB genes eliminated as argmax", 4, int(len(elim)))
    check("TRB sequences on eliminated genes", 20685, int(elim.pre_top1.sum()))
    check("TRBV13 pre-selection top-1 share", 0.1597,
          float(tb.loc["TRBV13", "pre_top1_share"]), tol=5e-5, fmt="{:.4f}")
    created = tb[(tb.pre_top1 == 0) & (tb.post_top1 > 0)]
    check("TRB genes created as argmax from zero", 6, int(len(created)))
    check("TRBV2 sequences created as argmax", 3838, int(tb.loc["TRBV2", "post_top1"]))
    erased_a = ta[(ta.pre_top1 > 0) & (ta.post_top1 == 0)]
    check("TRA sequences erased as argmax", 1, int(erased_a.pre_top1.sum()))
    check("TRA genes created as argmax from zero", 0,
          int(len(ta[(ta.pre_top1 == 0) & (ta.post_top1 > 0)])))
    check("TRAV9-2 pre-selection top-1 share", 0.0138,
          float(ta.loc["TRAV9-2", "pre_top1_share"]), tol=5e-5, fmt="{:.4f}")
    check("TRAV9-2 post-selection top-1 share", 0.0913,
          float(ta.loc["TRAV9-2", "post_top1_share"]), tol=5e-5, fmt="{:.4f}")


def check_cells(df):
    """Promotion/suppression cell accounting (Results: 26.1% vs 7.2%; TRA 0.40%/0.15%)."""
    for chain, quoted_sup, quoted_pro in [("TRB", 0.261, 0.072), ("TRA", 0.0040, 0.0015)]:
        f = A._selection_frame(df, chain)
        prek = f["cdr3_len"].astype(str) + "|" + f["pre_top1"]
        postk = f["cdr3_len"].astype(str) + "|" + f["post_top1"]
        pro = int((~postk.isin(set(prek))).sum())          # post cell absent pre -> dropped
        sup = int((~prek.isin(set(postk))).sum())          # pre cell absent post
        check(f"pre-cell-absent-post sequences, {chain}", quoted_sup, sup / len(f), tol=5e-4, fmt="{:.4f}")
        check(f"post-cell-absent-pre sequences, {chain}", quoted_pro, pro / len(f), tol=5e-4, fmt="{:.4f}")


def check_dropped_weight(df):
    t, n, nd, _ = A._dropped_weight_table(df, "TRB")
    check("TRBV2+TRBV28 share of TRB dropped weight", 0.94,
          float(t["pct_of_dropped"].reindex(["TRBV2", "TRBV28"]).sum()), tol=5e-3, fmt="{:.3f}")
    csv = pd.read_csv("results/supp_tables/tableS5_dropped_weight.csv").set_index("gene")
    bad = 0
    for g, row in csv.iterrows():
        bad += int(row["dropped"]) != int(t.loc[g, "dropped"])
        bad += abs(row["pct_of_dropped"] - 100 * t.loc[g, "pct_of_dropped"]) > 5e-2
        bad += abs(row["gene_drop_rate"] - 100 * t.loc[g, "gene_drop_rate"]) > 5e-2
    check("table S4 mismatched cells", 0, bad)


# --------------------------------------------------------------------------- #
# resampling claims: artifacts, or --resample to recompute them
# --------------------------------------------------------------------------- #
def check_heldout_cohorts():
    """Table S6 / the held-out-cohort Results subsection must match the run artifact."""
    art = pd.read_csv("results/validation_cohort/cohort_metrics.tsv", sep="\t")
    art = art.set_index(["cohort", "chain"])
    csv = pd.read_csv("results/supp_tables/tableS6_heldout_cohorts.csv").set_index("quantity")
    pairs = [("NSCLC", "TRA", "NSCLC_alpha"), ("NSCLC", "TRB", "NSCLC_beta"),
             ("HNSCC", "TRB", "HNSCC_beta")]
    fields = [("V-gene conditional entropy (nats)", "v_gene_entropy"),
              ("J-gene conditional entropy (nats)", "j_gene_entropy"),
              ("top-1 V coverage", "top1"), ("top-10 V coverage", "top10"),
              ("top-20 V coverage", "top20"),
              ("fraction with top-1 mass >= 0.5", "conf_0.5"),
              ("fraction with top-1 mass >= 0.99", "conf_0.99"),
              ("ARI vs analysis-set confusion grouping", "ari_vs_paper_groups")]
    bad = 0
    for cohort, chain, col in pairs:
        row = art.loc[(cohort, chain)]
        bad += int(csv.loc["n scored", col]) != int(row["n_scored"])
        for label, key in fields:
            bad += abs(float(csv.loc[label, col]) - float(row[key])) > 5e-5
    check("table S6 cells disagreeing with the cohort run", 0, bad)
    # the two claims the Results subsection makes about alpha reproducing
    a = art.loc[("NSCLC", "TRA")]
    check("held-out alpha V entropy", 1.2851, float(a["v_gene_entropy"]), tol=5e-5, fmt="{:.4f}")
    check("held-out alpha grouping ARI", 0.9594, float(a["ari_vs_paper_groups"]), tol=5e-5, fmt="{:.4f}")


def check_resampling(df, rerun):
    """Matched-n and equal-n are recomputed every run (~2 min); the 2,000-permutation
    test and the B=1,000 bootstrap are read from their artifacts unless --resample."""
    mn_json = json.load(open("results/matched_n_cross_chain.json"))
    mn = A.matched_n_selection(df, n_boot=500, seed=0, make_fig=False)
    cc = A.cross_chain_pvalue(df, n_boot=500, seed=0)
    en = A.equal_n_test(df, n_boot=500, seed=0, make_fig=False)

    check("beta matched-n entropy CI low (gene)", 2.6986, float(en["gene"]["ci"][0]), tol=5e-5, fmt="{:.4f}")
    check("beta matched-n entropy CI high (gene)", 2.7127, float(en["gene"]["ci"][1]), tol=5e-5, fmt="{:.4f}")
    check("beta matched-n entropy CI low (family)", 1.9272, float(en["family"]["ci"][0]), tol=5e-5, fmt="{:.4f}")
    check("beta matched-n entropy CI high (family)", 1.9370, float(en["family"]["ci"][1]), tol=5e-5, fmt="{:.4f}")
    check("beta matched-n shift CI low", -0.2858, float(mn["ci"][0]), tol=5e-5, fmt="{:+.4f}")
    check("beta matched-n shift CI high", -0.2774, float(mn["ci"][1]), tol=5e-5, fmt="{:+.4f}")
    check("cross-chain z", 99.4, float(cc["z"]), tol=0.05, fmt="{:.1f}")
    check("matched_n_cross_chain.json still matches a fresh run", True,
          abs(mn_json["beta_matched_ci"][0] - mn["ci"][0]) < 5e-9
          and abs(mn_json["cross_chain_z"] - cc["z"]) < 5e-9, fmt="{}")

    if rerun:
        perm = A.permutation_test(df, chains=("TRA", "TRB"), n_perm=2000, seed=0)
        boot = A.alpha_bootstrap(df, B=1000, seed=0)
    else:
        perm = json.load(open("results/permutation_test_TRA_TRB.json"))
        boot = json.load(open("results/alpha_bootstrap.json"))

    check("permutation p, TRA (floor 1/2001)", 0.0005, float(perm["TRA"]["p"]), tol=5e-7, fmt="{:.2e}")
    check("permutation p, TRB (floor 1/2001)", 0.0005, float(perm["TRB"]["p"]), tol=5e-7, fmt="{:.2e}")
    check("permutations as extreme, TRA", 0, int(perm["TRA"]["n_extreme"]))
    check("permutations as extreme, TRB", 0, int(perm["TRB"]["n_extreme"]))
    check("sd separation from null, TRA", 36.9, abs(float(perm["TRA"]["z"])), tol=0.05, fmt="{:.1f}")
    check("sd separation from null, TRB", 206.5, abs(float(perm["TRB"]["z"])), tol=0.05, fmt="{:.1f}")

    for key, quoted_lo, quoted_hi in [("v_gene", 1.2834, 1.3013), ("v_family", 1.1764, 1.1931),
                                      ("j_gene", 0.0106, 0.0123), ("shift_raw", -0.0740, -0.0563),
                                      ("shift_corrected", -0.0726, -0.0549),
                                      ("grouping_ari", 0.844, 1.000),
                                      ("beta_v_gene", 2.6995, 2.7121)]:
        _, lo, hi = boot[key]
        tol = 5e-4 if key == "grouping_ari" else 5e-5      # ari is quoted to 3 decimals
        check(f"alpha bootstrap CI low, {key}", quoted_lo, float(lo), tol=tol, fmt="{:+.4f}")
        check(f"alpha bootstrap CI high, {key}", quoted_hi, float(hi), tol=tol, fmt="{:+.4f}")
    check("bias-corrected shift point estimate", -0.0617, float(boot["points"]["corr"]),
          tol=5e-5, fmt="{:+.4f}")


WORD_COUNT_CLAIMS = ("stated body word count", "stated abstract word count",
                     "abstract within the 350-word limit")


def check_word_counts():
    """The metrics block on page 1 must match a fresh count, or it silently goes
    stale every time the manuscript is edited.

    These are the only checks that read the LaTeX sources rather than the deposited
    data. When those sources are absent, each claim is still emitted, as a skip, so
    the total stays the same number a reader is told to expect."""
    import shutil
    try:
        import count_manuscript_words as W
        available = bool(shutil.which("texcount")) and W.MAIN.exists()
    except ImportError:          # manuscript tooling is not part of the public repo
        available = False
    if not available:
        for claim in WORD_COUNT_CLAIMS:
            skip(claim, "manuscript sources absent")
        return
    said_body, said_abstract = W.stated()
    check(WORD_COUNT_CLAIMS[0], said_body, W.texcount(W.BODY))
    check(WORD_COUNT_CLAIMS[1], said_abstract, W.abstract_words())
    check(WORD_COUNT_CLAIMS[2], True, W.abstract_words() <= 350, fmt="{}")


def check_usage_reweight_and_calibration():
    """The two Results claims that rest on their own artifacts: the V-usage
    reweighting that explains the cohort entropy gap, and the held-out
    calibration split behind the candidate-set utility's coverage."""
    rw = json.load(open("results/validation_cohort/usage_reweight.json"))
    for key, quoted, lo, hi in [("NSCLC_TRB", 2.4181, 2.4064, 2.4299),
                                ("HNSCC_TRB", 2.4280, 2.4163, 2.4400)]:
        d = rw[key]
        check(f"reweighted beta entropy, {key}", quoted, float(d["reweighted_H"]), tol=5e-5, fmt="{:.4f}")
        check(f"reweighted CI low, {key}", lo, float(d["ci"][0]), tol=5e-5, fmt="{:.4f}")
        check(f"reweighted CI high, {key}", hi, float(d["ci"][1]), tol=5e-5, fmt="{:.4f}")
    check("cohort gap closed by usage, NSCLC beta", 0.269, float(rw["NSCLC_TRB"]["gap"]), tol=5e-4, fmt="{:.3f}")
    check("cohort gap closed by usage, HNSCC beta", 0.249, float(rw["HNSCC_TRB"]["gap"]), tol=5e-4, fmt="{:.3f}")
    for key, resid in [("NSCLC_TRB", 0.018), ("HNSCC_TRB", 0.029)]:
        d = rw[key]
        check(f"overshoot residual, {key}", resid,
              round(d["cohort_H"] - d["reweighted_H"], 3), tol=5e-4, fmt="{:.3f}")
    check("alpha reweighting moves entropy by", 0.0045, float(rw["NSCLC_TRA"]["closed"]),
          tol=5e-5, fmt="{:.4f}")
    check("alpha cohort gap", 0.0074, float(rw["NSCLC_TRA"]["gap"]), tol=5e-5, fmt="{:.4f}")
    check("usage reweighting overshoots both beta cohorts", True,
          all(rw[k]["frac_closed"] > 1 for k in ("NSCLC_TRB", "HNSCC_TRB")), fmt="{}")

    cal = json.load(open("results/validation_cohort/held_out_calibration.json"))
    check("largest held-out calibration deviation", 0.0039, float(cal["worst_abs_error"]),
          tol=5e-5, fmt="{:.4f}")
    check("held-out calibration within 0.4 pp everywhere", True,
          float(cal["worst_abs_error"]) <= 0.004, fmt="{}")
    at95 = {r["chain"]: r["held_out_coverage"] for r in cal["rows"] if r["target_coverage"] == 0.95}
    for chain in ("TRA", "TRB"):
        check(f"held-out coverage at requested 0.95, {chain}", 0.9486, float(at95[chain]),
              tol=5e-5, fmt="{:.4f}")

    # the cohort-replication claim the abstract-adjacent Discussion sentence makes
    m = pd.read_csv("results/validation_cohort/cohort_metrics.tsv", sep="\t")
    a = m[(m.cohort == "NSCLC") & (m.chain == "TRA")].iloc[0]
    check("alpha cohort entropy delta (quoted as within 0.008 nats)", True,
          abs(1.2925 - float(a.v_gene_entropy)) <= 0.008, fmt="{}")
    worst_pp = max(abs(q - float(getattr(a, c))) for q, c in
                   [(0.493, "top1"), (0.847, "top10"), (0.921, "top20"), (0.570, "conf_0.5")])
    check("alpha cohort coverage/confidence delta (quoted as within 2.5 pp)", True,
          worst_pp <= 0.025, fmt="{}")



# --------------------------------------------------------------------------- #
# supplementary tables: the CSV must equal a fresh recomputation
# --------------------------------------------------------------------------- #
def check_prism(df):
    """The Prism CSVs behind the publication figures must equal a fresh export
    (row order aside -- the committed fig6/fig7 exports predate a sort flip)."""
    import filecmp
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        A.export_prism(df, outdir=tmp)
        bad = []
        for name in sorted(os.listdir("results/figures/prism")):
            if not name.endswith(".csv"):
                continue
            new_p, old_p = os.path.join(tmp, name), f"results/figures/prism/{name}"
            if filecmp.cmp(new_p, old_p, shallow=False):
                continue
            o, n = pd.read_csv(old_p), pd.read_csv(new_p)
            key = o.columns[0]
            o, n = o.set_index(key).sort_index(), n.set_index(key).sort_index()
            num = o.select_dtypes(include="number").columns
            if not (set(o.index) == set(n.index) and list(o.columns) == list(n.columns)
                    and float((o[num] - n.loc[o.index, num]).abs().max().max()) < 5e-9):
                bad.append(name)
        check("prism CSVs that disagree with a fresh export", 0, len(bad) and bad or 0)


def check_figS2_clusters(df):
    """figS2 captions: 7 data-driven confusion clusters of >=2 genes in each chain."""
    from scipy.cluster.hierarchy import fcluster
    for chain, quoted in [("TRA", 7), ("TRB", 7)]:
        keep, M = A.build_leakage(df, chain, 20)
        order, Z = A._confusion_order(M)
        K = len(set(family_of(g) for g in keep))
        lab = fcluster(Z, t=K, criterion="maxclust")[order]
        blocks, start = 0, 0
        for i in range(1, len(lab) + 1):
            if i == len(lab) or lab[i] != lab[start]:
                blocks += (i - start) >= 2
                start = i
        check(f"figS2 confusion clusters of >=2 genes, {chain}", quoted, blocks)


def check_tableS2(df):
    csv = pd.read_csv("results/supp_tables/tableS2_source_robustness.csv").set_index("source")
    pool = pd.read_csv("results/ingest/canonical_alpha_pooled.tsv", sep="\t")
    s = A._slice(df, chain="TRA", axis="V", model="pre", resolution="gene")
    for row, src in [("McPAS-alpha", "mcpas_alpha"), ("VDJdb-alpha", "vdjdb_alpha"), ("Pooled-alpha", None)]:
        if src is None:
            sub, ids = df, set(s.seq_id)
        else:
            ids = set(pool.loc[pool.sources.str.split(";").apply(lambda L: src in L), "seq_id"])
            sub = df[df.seq_id.isin(ids)]
        sl = s[s.seq_id.isin(ids)]
        check(f"table S2 n, {row}", int(csv.loc[row, "n"]), int(sl.seq_id.nunique()))
        check(f"table S2 v_gene_entropy, {row}", float(csv.loc[row, "v_gene_entropy"]),
              float(sl.entropy_nats.mean()), tol=5e-4, fmt="{:.4f}")
        f = A._selection_frame(sub, "TRA")
        check(f"table S2 ctrl_selection_shift, {row}", float(csv.loc[row, "ctrl_selection_shift"]),
              A._controlled_shift(f)["ctrl_shift"], tol=5e-4, fmt="{:+.4f}")

    # the caption's reconciliation of the three source counts against the pool
    p = pool[pool.seq_id.isin(set(s.seq_id))]
    memberships = p.sources.str.split(";")
    check("table S2 caption, canonical NeoTCR-alpha", 260,
          int(memberships.apply(lambda L: "neotcr_alpha" in L).sum()))
    check("table S2 caption, canonical rows in >1 source", 486,
          int((p.sources.str.count(";") > 0).sum()))
    check("table S2 caption, duplicate memberships", 487,
          int(memberships.apply(len).sum() - len(p)))
    check("table S2 caption, source counts reconcile to the pool", 37687,
          5829 + 32085 + 260 - 487)


def check_tableS4(df):
    csv = pd.read_csv("results/supp_tables/tableS4_redistribution.csv")
    acc = pd.read_csv("results/supp_tables/tableS4_redistribution_accounting.csv").set_index("chain")
    for chain in ("TRA", "TRB"):
        t, N = A._redistribution_table(df, chain)
        t = t.set_index("gene")
        sub = csv[csv.chain == chain]
        bad = 0
        for _, row in sub.iterrows():
            r = t.loc[row.gene]
            bad += int(row.pre_top1) != int(r.pre_top1)
            bad += int(row.post_top1) != int(r.post_top1)
            bad += abs(row.delta_top1_share - r.delta_top1_share) > 5e-6
            bad += int(row.N) != N
        check(f"table S3 mismatched cells, {chain}", 0, bad)

        f = A._selection_frame(df, chain)
        prek = f["cdr3_len"].astype(str) + "|" + f["pre_top1"]
        postk = f["cdr3_len"].astype(str) + "|" + f["post_top1"]
        pro_seq = int((~postk.isin(set(prek))).sum())
        sup_seq = int((~prek.isin(set(postk))).sum())
        a = acc.loc[chain]
        check(f"table S3 accounting n, {chain}", int(a.n), int(len(f)))
        check(f"table S3 accounting promotion_seqs, {chain}", int(a.promotion_seqs), pro_seq)
        check(f"table S3 accounting suppression_seqs, {chain}", int(a.suppression_seqs), sup_seq)
        check(f"table S3 accounting dropped_total, {chain}", int(a.dropped_total), pro_seq)
        check(f"table S3 accounting promotion_bins, {chain}", int(a.promotion_bins),
              len(set(postk) - set(prek)))
        check(f"table S3 accounting suppression_bins, {chain}", int(a.suppression_bins),
              len(set(prek) - set(postk)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--posteriors", default=A.PATH, metavar="TSV",
                    help=f"per-sequence posterior table (default: {A.PATH})")
    ap.add_argument("--resample", action="store_true",
                    help="recompute the permutation test and bootstrap instead of "
                         "reading their stored artifacts")
    args = ap.parse_args()

    preflight(args.posteriors)
    rerun = args.resample
    df = A.load(args.posteriors)
    check_ingest()
    check_canonical(df)
    check_entropies(df)
    check_topk(df)
    check_confidence(df)
    check_selfmass(df)
    check_gene_sets(df)
    check_ari(df)
    check_mcpas_null(df)
    check_selection(df)
    check_redistribution(df)
    check_cells(df)
    check_dropped_weight(df)
    check_prism(df)
    check_figS2_clusters(df)
    check_tableS2(df)
    check_tableS4(df)
    check_heldout_cohorts()
    check_resampling(df, rerun)
    check_usage_reweight_and_calibration()
    check_word_counts()

    w = max(len(r[0]) for r in ROWS)
    print(f"\n{'claim':<{w}}  {'manuscript':>12}  {'recomputed':>12}  status")
    print("-" * (w + 42))
    for claim, quoted, computed, status in ROWS:
        print(f"{claim:<{w}}  {quoted:>12}  {computed:>12}  {status}")
    n_fail = sum(r[3] == "FAIL" for r in ROWS)
    n_skip = sum(r[3] == "SKIP" for r in ROWS)
    line = f"\n{len(ROWS) - n_fail}/{len(ROWS)} claims reproduce"
    if n_skip:
        line += f" ({n_skip} skipped, inputs not present here)"
    if n_fail:
        line += f"; {n_fail} FAIL"
    print(line)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
