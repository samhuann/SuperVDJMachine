"""Controls on what the CDR3 posterior recovers.

Each subcommand asks whether the measured recoverability is explained by something other
than information in the CDR3, or checks it against an independent record.

    PYTHONPATH=. python scripts/recoverability_controls.py cache      # one pass over posteriors.tsv
    PYTHONPATH=. python scripts/recoverability_controls.py usage      # gene usage prior vs the CDR3
    PYTHONPATH=. python scripts/recoverability_controls.py ambiguity  # CDR3s annotated to several V genes
    PYTHONPATH=. python scripts/recoverability_controls.py pairing    # does the partner chain help
    PYTHONPATH=. python scripts/recoverability_controls.py antigen    # does an epitope label help
    PYTHONPATH=. python scripts/recoverability_controls.py germline   # germline similarity and trimming

`cache` must run first; it reduces the posterior table to dense per-gene arrays the others
reuse. Results are written to results/ as JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supervdj.models import (candidate_genes_from_olga, load_olga,  # noqa: E402
                             strip_allele, _olga_model_dir)

POSTERIORS = Path("results/posteriors.tsv")
OUT = Path("results/tier3")
CACHE = OUT / "posteriors_pre_gene.pkl"
COHORT_DIR = Path(os.environ.get("SUPERVDJ_COHORT_DIR", "data/cohorts"))
CHAINS = ("TRA", "TRB")
PSEUDO = 0.5          # additive smoothing for association tables
MIN_COUNT = 20        # same gene threshold the manuscript's leakage matrices use


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def entropy_rows(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(P > 0, P * np.log(P), 0.0).sum(1)


def true_rank(P, true):
    """1-based rank of the annotated gene in each row (ties broken pessimistically)."""
    p_true = P[np.arange(len(true)), true]
    return 1 + (P > p_true[:, None]).sum(1)


def summarize(P, true):
    r = true_rank(P, true)
    return {"n": int(len(true)), "mean_entropy_nats": float(entropy_rows(P).mean()),
            "top1": float((r == 1).mean()), "top10": float((r <= 10).mean())}


def half(key: str) -> int:
    """Deterministic 50/50 split on a string key."""
    return zlib.crc32(key.encode()) & 1


def normalize_rows(P):
    s = P.sum(1, keepdims=True)
    return np.divide(P, s, out=np.zeros_like(P), where=s > 0)


def bayes_update(P, partner, A, base):
    """Combine a CDR3-only posterior with a partner feature.

    P        (n, T)  CDR3-only posterior over target genes
    partner  (n, S)  distribution over the partner feature (one-hot if annotated)
    A        (T, S)  P(target | partner value), estimated on the training split
    base     (T,)    the same table's implied marginal, sum_s A[t,s] P_train(s)

    p'(t) ∝ P(t) * sum_s A[t,s] q(s) / base(t). Dividing by the table's *own*
    marginal rather than a separately counted prior makes the update exactly the
    identity when the partner carries no information, including for target genes
    absent from the training split. Counting the prior separately gave those genes
    a large spurious boost (A[t,s] is smoothing-dominated while the prior is not),
    which the shuffled-pairing control exposed.
    """
    L = (partner @ A.T) / base[None, :]
    return normalize_rows(P * L)


def association(target_idx, partner_idx, T, S):
    """P(target | partner) with additive smoothing, plus the implied target marginal."""
    C = np.zeros((T, S))
    np.add.at(C, (target_idx, partner_idx), 1.0)
    A = (C + PSEUDO) / (C.sum(0, keepdims=True) + PSEUDO * T)
    p_s = (C.sum(0) + PSEUDO * T) / (C.sum() + PSEUDO * T * S)
    return A, A @ p_s


def _selfcheck():
    rng = np.random.default_rng(0)
    # an uninformative partner must leave P unchanged
    P = normalize_rows(rng.random((5, 4)))
    prior = np.array([.1, .2, .3, .4])
    A = np.tile(prior[:, None], (1, 3))
    assert np.allclose(bayes_update(P, normalize_rows(rng.random((5, 3))), A, A @ np.full(3, 1 / 3)), P)
    # and still unchanged when estimated from independent draws in which some target
    # genes never occur -- the failure the shuffled-pairing control caught
    T, S, n = 12, 6, 20000
    t_idx = rng.integers(0, T - 4, n)          # last 4 targets absent from training
    s_idx = rng.integers(0, S, n)              # partner independent of target
    A2, base2 = association(t_idx, s_idx, T, S)
    # the Methods state A[t,s] = P(t | s). The table is column-normalized, so its columns
    # must sum to one; this is what the external review caught written backwards in the text.
    assert abs(A2.sum(axis=0) - 1.0).max() < 1e-9, "A[t,s] must be P(t|s)"
    P2 = normalize_rows(rng.random((200, T)))
    q2 = np.eye(S)[rng.integers(0, S, 200)]
    moved = np.abs(bayes_update(P2, q2, A2, base2) - P2).max()
    assert moved < 0.02, moved
    # a perfectly informative partner must move the posterior to the implied target
    t3 = rng.integers(0, T, n)
    A3, base3 = association(t3, t3 % S, T, S)
    assert np.abs(bayes_update(P2, np.eye(S)[np.zeros(200, int)], A3, base3) - P2).max() > 0.05
    # rank of the annotated gene
    assert list(true_rank(np.array([[.5, .3, .2], [.1, .6, .3]]), np.array([0, 2]))) == [1, 2]
    # split is deterministic and roughly balanced
    assert half("CASSF") == half("CASSF")
    assert 0.4 < np.mean([half(f"k{i}") for i in range(2000)]) < 0.6
    assert identity("CASSL", "CASSL") == 1.0 and identity("CASS", "CAVR") == 0.5


def olga_parts(chain):
    """Gene-level pieces of the OLGA model: candidate genes, P(V gene), the germline
    V segment from the conserved cysteine onward, and mean 3' V deletion."""
    import olga.load_model as lm
    f = _olga_model_dir(chain)
    if chain == "TRA":
        g, m = lm.GenomicDataVJ(), lm.GenerativeModelVJ()
    else:
        g, m = lm.GenomicDataVDJ(), lm.GenerativeModelVDJ()
    g.load_igor_genomic_data(os.path.join(f, "model_params.txt"),
                             os.path.join(f, "V_gene_CDR3_anchors.csv"),
                             os.path.join(f, "J_gene_CDR3_anchors.csv"))
    m.load_and_process_igor_model(os.path.join(f, "model_marginals.txt"))
    pv_allele = m.PVJ.sum(1) if chain == "TRA" else m.PV
    alleles = [x[0] for x in g.genV]
    # deletion index i <-> i - max_delV_palindrome nt removed (negative = palindromic)
    dels = np.arange(m.PdelV_given_V.shape[0]) - g.max_delV_palindrome
    mean_del_allele = dels @ m.PdelV_given_V            # (n_alleles,)
    pv, md_num, tail = defaultdict(float), defaultdict(float), {}
    for i, a in enumerate(alleles):
        gene = strip_allele(a)
        pv[gene] += pv_allele[i]
        md_num[gene] += pv_allele[i] * mean_del_allele[i]
        seg = g.genV[i][1]
        if seg and (gene not in tail or a.endswith("*01")):
            tail[gene] = seg
    mean_del = {k: md_num[k] / pv[k] for k in pv if pv[k] > 0}
    return dict(pv), mean_del, tail


def load_cache():
    with CACHE.open("rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------------------------------- #
# cache: one streaming pass over posteriors.tsv
# --------------------------------------------------------------------------- #
def cmd_cache(_args):
    genes = {}
    for chain in CHAINS:
        v, j = candidate_genes_from_olga(load_olga(chain))
        genes[(chain, "V")], genes[(chain, "J")] = v, j
    idx = {k: {g: i for i, g in enumerate(v)} for k, v in genes.items()}
    acc = {k: {"cdr3": [], "true": [], "rows": [], "ent_file": []} for k in genes}
    with POSTERIORS.open(newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if (r["model"] != "pre" or r["resolution"] != "gene" or r["status"] != "ok"
                    or (r["mode"] or "") != ""):
                continue
            k = (r["chain"], r["axis"])
            ix = idx[k]
            row = np.zeros(len(ix), dtype=np.float32)
            for g, mass in json.loads(r["posterior_json"]).items():
                row[ix[g]] = mass
            a = acc[k]
            a["cdr3"].append(r["cdr3"])
            a["true"].append(ix[r["true_gene"]])
            a["rows"].append(row)
            a["ent_file"].append(float(r["entropy_nats"]))
    out = {}
    for k, a in acc.items():
        P = np.vstack(a["rows"])
        out[k] = {"genes": genes[k], "cdr3": a["cdr3"],
                  "true": np.array(a["true"], dtype=np.int32), "P": P,
                  "ent_file": np.array(a["ent_file"])}
        drift = np.abs(entropy_rows(P.astype(np.float64)) - out[k]["ent_file"]).max()
        print(f"[{k[0]} {k[1]}] {len(a['true']):,} sequences x {len(genes[k])} genes; "
              f"max |recomputed - file| entropy = {drift:.2e}")
    OUT.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as fh:
        pickle.dump(out, fh, protocol=4)
    print(f"wrote {CACHE}")


# --------------------------------------------------------------------------- #
# 3A  R2 major 2: V usage versus information carried by the CDR3
# --------------------------------------------------------------------------- #
def cmd_3a(_args):
    data, report = load_cache(), {}
    for chain in CHAINS:
        d = data[(chain, "V")]
        genes, P, true = d["genes"], d["P"].astype(np.float64), d["true"]
        n, G = len(true), len(genes)
        native = summarize(P, true)

        # usage-only baseline, empirical: the best a CDR3-blind guess can do on this data
        freq = np.bincount(true, minlength=G) / n
        order = np.argsort(-freq)
        rank_of = np.empty(G, int)
        rank_of[order] = np.arange(1, G + 1)
        r = rank_of[true]
        h_emp = float(-(freq[freq > 0] * np.log(freq[freq > 0])).sum())
        usage_emp = {"n": n, "entropy_nats": h_emp,
                     "top1": float((r == 1).mean()), "top10": float((r <= 10).mean())}

        # usage-only baseline, model: the OLGA prior P(V) the posterior is built on
        pv, _, _ = olga_parts(chain)
        prior = np.array([pv.get(g, 0.0) for g in genes])
        prior = prior / prior.sum()
        orderm = np.argsort(-prior)
        rank_m = np.empty(G, int)
        rank_m[orderm] = np.arange(1, G + 1)
        rm = rank_m[true]
        h_mod = float(-(prior[prior > 0] * np.log(prior[prior > 0])).sum())
        usage_mod = {"n": n, "entropy_nats": h_mod,
                     "top1": float((rm == 1).mean()), "top10": float((rm <= 10).mean())}

        # prior-equalized: posterior is P(CDR3|V)P(V) normalized; divide P(V) back out
        ratio = np.divide(P, prior[None, :], out=np.zeros_like(P), where=prior[None, :] > 0)
        eq_rows = ratio.sum(1) > 0
        equalized = summarize(normalize_rows(ratio[eq_rows]), true[eq_rows])

        report[chain] = {
            "native_posterior": native,
            "usage_only_empirical": usage_emp,
            "usage_only_model_prior": usage_mod,
            "prior_equalized_posterior": equalized,
            "information_from_cdr3_nats": {
                "vs_empirical_usage": h_emp - native["mean_entropy_nats"],
                "fraction_of_empirical_usage_entropy":
                    (h_emp - native["mean_entropy_nats"]) / h_emp},
        }
        print(f"[{chain}] {n:,} sequences, {G} V genes")
        for name, s in (("native posterior", native), ("usage only (empirical)", usage_emp),
                        ("usage only (OLGA prior)", usage_mod), ("prior-equalized", equalized)):
            ent = s.get("mean_entropy_nats", s.get("entropy_nats"))
            print(f"  {name:26s} entropy {ent:.4f}  top-1 {s['top1']:.3f}  top-10 {s['top10']:.3f}")
        info = report[chain]["information_from_cdr3_nats"]
        print(f"  information from the CDR3 above empirical usage: {info['vs_empirical_usage']:.4f} nats "
              f"({100 * info['fraction_of_empirical_usage_entropy']:.1f}% of usage entropy)")
    _write("3a_usage_vs_cdr3", report)


# --------------------------------------------------------------------------- #
# 3B  R2 major 3, second half: are identical CDR3s observed with different V genes,
#     and does that empirical ambiguity track the predicted entropy?
# --------------------------------------------------------------------------- #
def cmd_3b(_args):
    data, report = load_cache(), {}
    canon = {"TRA": pd.read_csv("results/ingest/canonical_alpha_pooled.tsv", sep="\t"),
             "TRB": pd.read_csv("results/ingest/canonical_unique.tsv", sep="\t")}
    canon["TRB"] = canon["TRB"][canon["TRB"].chain == "TRB"]
    for chain in CHAINS:
        d = data[(chain, "V")]
        # the posterior depends only on the CDR3, so every row of a CDR3 carries the same one
        ent = defaultdict(list)
        for c, e in zip(d["cdr3"], d["ent_file"]):
            ent[c].append(e)
        spread = max(max(v) - min(v) for v in ent.values())
        ent1 = {c: v[0] for c, v in ent.items()}

        df = canon[chain][canon[chain].cdr3_aa.isin(ent1)]
        g = df.groupby("cdr3_aa").agg(
            n_v=("v_gene", "nunique"), n_rows=("v_gene", "size"),
            n_src=("sources", lambda s: len({x for v in s for x in str(v).split(";")})))
        g["entropy"] = g.index.map(ent1)
        # a CDR3 seen once can only ever show one V gene: restrict the test to CDR3s
        # observed independently more than once (more than one annotation, or one
        # annotation reported by more than one database)
        multi = g[(g.n_rows >= 2) | (g.n_src >= 2)]
        amb = multi[multi.n_v >= 2]
        agree = multi[multi.n_v == 1]
        u = mannwhitneyu(amb.entropy, agree.entropy, alternative="two-sided")
        auc = u.statistic / (len(amb) * len(agree))
        rho = spearmanr(multi.entropy, multi.n_v)
        report[chain] = {
            "distinct_cdr3s": int(len(g)),
            "cdr3s_with_more_than_one_v": int((g.n_v >= 2).sum()),
            "fraction_with_more_than_one_v": float((g.n_v >= 2).mean()),
            "max_v_genes_for_one_cdr3": int(g.n_v.max()),
            "multiply_observed_cdr3s": int(len(multi)),
            "of_which_ambiguous": int(len(amb)),
            "of_which_agreeing": int(len(agree)),
            "median_predicted_entropy_ambiguous": float(amb.entropy.median()),
            "median_predicted_entropy_agreeing": float(agree.entropy.median()),
            "auc_entropy_separates_ambiguous_from_agreeing": float(auc),
            "mannwhitney_p": float(u.pvalue),
            "spearman_entropy_vs_n_distinct_v": float(rho.correlation),
            "spearman_p": float(rho.pvalue),
            "max_entropy_spread_within_a_cdr3": float(spread),
        }
        r = report[chain]
        print(f"[{chain}] {r['distinct_cdr3s']:,} CDR3s; {r['cdr3s_with_more_than_one_v']:,} "
              f"({100 * r['fraction_with_more_than_one_v']:.1f}%) seen with >1 V gene")
        print(f"  multiply observed: {r['multiply_observed_cdr3s']:,} "
              f"({r['of_which_ambiguous']:,} ambiguous, {r['of_which_agreeing']:,} agreeing)")
        print(f"  median predicted entropy: ambiguous {r['median_predicted_entropy_ambiguous']:.3f} "
              f"vs agreeing {r['median_predicted_entropy_agreeing']:.3f}; AUC "
              f"{r['auc_entropy_separates_ambiguous_from_agreeing']:.3f} (p={r['mannwhitney_p']:.2g}); "
              f"Spearman {r['spearman_entropy_vs_n_distinct_v']:.3f} (p={r['spearman_p']:.2g})")
    _write("3b_empirical_ambiguity", report)


# --------------------------------------------------------------------------- #
# 3C  R1 comment 7: does pairing help V/J inference? does antigen specificity?
# --------------------------------------------------------------------------- #
def _normalizers():
    from ingest.gene_names import GeneReconciler
    from ingest.imgt_boundaries import ImgtBoundaries
    return GeneReconciler.from_olga(), ImgtBoundaries.from_dir(Path("data/imgt"))


def _norm_one(rec, bnd, chain, cdr3, v, j):
    vg, _ = rec.map_v(v, chain)
    jg, _ = rec.map_j(j, chain)
    if vg is None or jg is None:
        return None
    c, _ = bnd.normalize_cdr3(str(cdr3 or ""), jg)
    return None if c is None else (c, vg, jg)


_W = {}


def _w_init(chain):
    from supervdj.cache import ValueCache
    from supervdj.models import load_chain_models
    _W["m"] = load_chain_models(chain, use_sonia=False)
    _W["c"] = ValueCache(None)


def _w_post(cdr3):
    from supervdj.posterior import preselection_posterior
    v = preselection_posterior(_W["m"], _W["c"], cdr3, "V")
    if not v or sum(v.values()) == 0:
        return None
    return v, preselection_posterior(_W["m"], _W["c"], cdr3, "J")


def _dense(dicts, genes):
    ix = {g: i for i, g in enumerate(genes)}
    P = np.zeros((len(dicts), len(genes)))
    for k, dct in enumerate(dicts):
        for g, m in dct.items():
            P[k, ix[g]] = m
    return P


def _compare(P, true, Pn):
    a, b = summarize(P, true), summarize(Pn, true)
    return {"cdr3_only": a, "with_partner": b,
            "delta_top1": b["top1"] - a["top1"], "delta_top10": b["top10"] - a["top10"],
            "delta_entropy": b["mean_entropy_nats"] - a["mean_entropy_nats"]}


def cmd_3c_pairing(args):
    import multiprocessing as mp
    raw = pd.read_csv(COHORT_DIR / "data-nsclc/tcr.csv.gz",
                      usecols=["TRA_cdr3", "TRA_v_gene", "TRA_j_gene",
                               "TRB_cdr3", "TRB_v_gene", "TRB_j_gene"]).dropna().drop_duplicates()
    rec, bnd = _normalizers()
    pairs = []
    for r in raw.itertuples(index=False):
        a = _norm_one(rec, bnd, "TRA", r.TRA_cdr3, r.TRA_v_gene, r.TRA_j_gene)
        b = _norm_one(rec, bnd, "TRB", r.TRB_cdr3, r.TRB_v_gene, r.TRB_j_gene)
        if a and b:
            pairs.append(a + b)
    pairs = pd.DataFrame(pairs, columns=["ca", "va", "ja", "cb", "vb", "jb"]).drop_duplicates()
    print(f"NSCLC: {len(raw):,} distinct raw pairs -> {len(pairs):,} pairs with both chains normalized")

    test = pairs.sample(n=min(args.n, len(pairs)), random_state=0)
    # training pairs share no CDR3 with any test pair, on either chain
    train = pairs[~pairs.ca.isin(set(test.ca)) & ~pairs.cb.isin(set(test.cb))]
    print(f"  test {len(test):,} pairs (OLGA posteriors); train {len(train):,} pairs (annotations only)")

    pcache = OUT / f"pairing_posteriors_n{len(test)}.pkl"
    if pcache.exists():
        with pcache.open("rb") as fh:
            post = pickle.load(fh)
    else:
        post = {}
        for chain, col in (("TRA", "ca"), ("TRB", "cb")):
            with mp.get_context("spawn").Pool(args.workers, initializer=_w_init, initargs=(chain,)) as pool:
                post[chain] = pool.map(_w_post, list(test[col]), chunksize=16)
            print(f"  {chain}: posteriors for {sum(x is not None for x in post[chain]):,} of {len(test):,}")
        OUT.mkdir(parents=True, exist_ok=True)
        with pcache.open("wb") as fh:
            pickle.dump(post, fh)

    ok = np.array([a is not None and b is not None for a, b in zip(post["TRA"], post["TRB"])])
    test = test[ok]
    pa = [x for x, k in zip(post["TRA"], ok) if k]
    pb = [x for x, k in zip(post["TRB"], ok) if k]
    genes = {}
    for chain in CHAINS:
        v, j = candidate_genes_from_olga(load_olga(chain))
        genes[(chain, "V")], genes[(chain, "J")] = v, j

    report = {"n_test_pairs": int(len(test)), "n_train_pairs": int(len(train))}
    rng = np.random.default_rng(0)
    # target chain, axis -> (target column, partner chain, partner column, posteriors)
    for t_chain, axis, t_col, p_chain, p_col in (("TRA", "V", "va", "TRB", "vb"),
                                                 ("TRB", "V", "vb", "TRA", "va"),
                                                 ("TRA", "J", "ja", "TRB", "jb"),
                                                 ("TRB", "J", "jb", "TRA", "ja")):
        tg, sg = genes[(t_chain, axis)], genes[(p_chain, axis)]
        ti, si = {g: i for i, g in enumerate(tg)}, {g: i for i, g in enumerate(sg)}
        k_ax = 0 if axis == "V" else 1
        P = _dense([x[k_ax] for x in (pa if t_chain == "TRA" else pb)], tg)
        Q = _dense([x[k_ax] for x in (pb if t_chain == "TRA" else pa)], sg)
        true = np.array([ti[g] for g in test[t_col]])
        part_true = np.array([si[g] for g in test[p_col]])
        m_t, m_s = train[t_col].map(ti), train[p_col].map(si)
        both = m_t.notna() & m_s.notna()
        tr_t, tr_s = m_t[both].astype(int).to_numpy(), m_s[both].astype(int).to_numpy()
        A, prior = association(tr_t, tr_s, len(tg), len(sg))
        oracle = np.eye(len(sg))[part_true]                   # annotated partner gene
        A_shuf, base_shuf = association(tr_t, rng.permutation(tr_s), len(tg), len(sg))
        # a single shuffle gives one point, which cannot say whether an observed delta is
        # inside the null; repeat it so the control has a range to compare against
        null = []
        for _ in range(args.pair_permutations):
            As, bs = association(tr_t, rng.permutation(tr_s), len(tg), len(sg))
            null.append(_compare(P, true, bayes_update(P, oracle, As, bs))["delta_top1"])
        null = np.array(null)
        name = f"{t_chain}_{axis}_given_partner_{p_chain}_{axis}"
        report[name] = {
            "partner_from_its_cdr3": _compare(P, true, bayes_update(P, Q, A, prior)),
            "partner_annotated_gene": _compare(P, true, bayes_update(P, oracle, A, prior)),
            "control_shuffled_pairing": _compare(P, true, bayes_update(P, oracle, A_shuf, base_shuf)),
            "shuffled_null_delta_top1": {
                "permutations": int(args.pair_permutations),
                "mean": float(null.mean()), "sd": float(null.std(ddof=1)),
                "p2.5": float(np.percentile(null, 2.5)),
                "p97.5": float(np.percentile(null, 97.5)),
                "min": float(null.min()), "max": float(null.max()),
                # one-sided empirical p for each observed delta against this null
                "p_cdr3": float((1 + (null >= _compare(
                    P, true, bayes_update(P, Q, A, prior))["delta_top1"]).sum())
                    / (len(null) + 1)),
                "p_gene": float((1 + (null >= _compare(
                    P, true, bayes_update(P, oracle, A, prior))["delta_top1"]).sum())
                    / (len(null) + 1))},
        }
        r = report[name]
        print(f"[{name}]  CDR3-only top-1 {r['partner_from_its_cdr3']['cdr3_only']['top1']:.3f}")
        for lab in ("partner_from_its_cdr3", "partner_annotated_gene", "control_shuffled_pairing"):
            x = r[lab]
            print(f"    {lab:26s} top-1 {x['delta_top1']:+.4f}  top-10 {x['delta_top10']:+.4f}  "
                  f"entropy {x['delta_entropy']:+.4f}")
    _write("3c_pairing", report)


def cmd_3c_antigen(_args):
    data = load_cache()
    vdj = pd.read_csv("data/VDJdb Greater Than 0.tsv", sep="\t", dtype=str, low_memory=False)
    vdj = vdj[vdj.Gene.isin(CHAINS) & (vdj.Species == "HomoSapiens")]
    rec, bnd = _normalizers()
    rows = []
    for r in vdj.itertuples(index=False):
        n = _norm_one(rec, bnd, r.Gene, r.CDR3, r.V, r.J)
        if n:
            rows.append((r.Gene,) + n + (r.Epitope, str(r.Reference)))
    df = pd.DataFrame(rows, columns=["chain", "cdr3", "v", "j", "epitope", "ref"])
    # an unambiguous specificity label: CDR3s annotated to exactly one epitope
    n_epi = df.groupby(["chain", "cdr3"]).epitope.nunique()
    single = n_epi[n_epi == 1].index
    df = df.set_index(["chain", "cdr3"]).loc[single].reset_index()
    df = df.drop_duplicates(["chain", "cdr3", "v", "j", "epitope"])
    print(f"VDJdb human: {len(df):,} (CDR3, V, J, epitope) instances with a single epitope")

    report = {}
    for chain in CHAINS:
        for axis, col in (("V", "v"), ("J", "j")):
            d = data[(chain, axis)]
            first = {}
            for k, c in enumerate(d["cdr3"]):
                first.setdefault(c, k)
            sub = df[(df.chain == chain) & df.cdr3.isin(first)].copy()
            genes = d["genes"]
            gi = {g: i for i, g in enumerate(genes)}
            sub = sub[sub[col].isin(gi)]
            P_all = d["P"].astype(np.float64)
            res = {}
            # split by CDR3, and, as the check on study confounding, by publication
            for split, key in (("cdr3_heldout", "cdr3"), ("publication_heldout", "ref")):
                is_test = sub[key].map(half).to_numpy() == 1
                tr, te = sub[~is_test], sub[is_test]
                epis = sorted(sub.epitope.unique())
                ei = {e: i for i, e in enumerate(epis)}
                A, prior = association(tr[col].map(gi).to_numpy(), tr.epitope.map(ei).to_numpy(),
                                       len(genes), len(epis))
                n_tr = Counter(tr.epitope)
                P = P_all[[first[c] for c in te.cdr3]]
                true = te[col].map(gi).to_numpy()
                onehot = np.eye(len(epis))[te.epitope.map(ei).to_numpy()]
                Pn = bayes_update(P, onehot, A, prior)
                # epitopes with too little training data get no update at all
                has = te.epitope.map(lambda e: n_tr[e] >= MIN_COUNT).to_numpy()
                Pn[~has] = P[~has]
                res[split] = {"n_train": int(len(tr)), "n_test": int(len(te)),
                              "n_test_with_update": int(has.sum()),
                              "all_test": _compare(P, true, Pn),
                              "updated_subset": _compare(P[has], true[has], Pn[has])}
            name = f"{chain}_{axis}"
            report[name] = res
            for split, x in res.items():
                u = x["updated_subset"]
                print(f"[{name} {split:20s}] test {x['n_test']:,}, updated {x['n_test_with_update']:,}: "
                      f"top-1 {u['cdr3_only']['top1']:.3f} -> {u['with_partner']['top1']:.3f} "
                      f"({u['delta_top1']:+.4f})  entropy {u['delta_entropy']:+.4f}")
    _write("3c_antigen", report)


# --------------------------------------------------------------------------- #
# 3D  R2 major 4, second half: is V confusion explained by CDR3-proximal germline
#     similarity and by V-gene trimming?
# --------------------------------------------------------------------------- #
def identity(a: str, b: str) -> float:
    """Amino-acid identity over the shared length, both aligned at the conserved
    cysteine (position 0). Trimming removes residues from the 3' end, so the
    shared prefix is what both genes can contribute to a CDR3."""
    n = min(len(a), len(b))
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n if n else float("nan")


def cmd_3d(args):
    from ingest.imgt_boundaries import _translate
    data, report = load_cache(), {}
    rng = np.random.default_rng(0)
    for chain in CHAINS:
        d = data[(chain, "V")]
        genes, P, true = d["genes"], d["P"].astype(np.float64), d["true"]
        cnt = np.bincount(true, minlength=len(genes))
        keep = [i for i in range(len(genes)) if cnt[i] >= MIN_COUNT]
        _, mean_del, tail_nt = olga_parts(chain)
        keep = [i for i in keep if genes[i] in tail_nt and genes[i] in mean_del]
        names = [genes[i] for i in keep]
        sel = np.isin(true, keep)
        remap = {g: k for k, g in enumerate(keep)}
        M = np.zeros((len(keep), len(keep)))
        np.add.at(M, np.array([remap[t] for t in true[sel]]), P[sel][:, keep])
        M /= cnt[keep][:, None]
        S = (M + M.T) / 2.0                                    # symmetric confusion
        aa = {g: _translate(tail_nt[g], 1) for g in names}
        Iden = np.array([[identity(aa[a], aa[b]) for b in names] for a in names])
        iu = np.triu_indices(len(names), 1)
        rho = spearmanr(S[iu], Iden[iu]).correlation
        null = []
        for _ in range(args.pair_permutations):
            p = rng.permutation(len(names))
            null.append(spearmanr(S[iu], Iden[np.ix_(p, p)][iu]).correlation)
        null = np.array(null)
        p_mantel = (1 + (np.abs(null) >= abs(rho)).sum()) / (len(null) + 1)

        leak_out = 1.0 - np.diag(M)                            # mass leaving the true gene
        md = np.array([mean_del[g] for g in names])
        tr = spearmanr(md, leak_out)
        report[chain] = {
            "n_genes": len(names),
            "germline_similarity": {
                "metric": "amino-acid identity of the germline V segment from the conserved "
                          "cysteine, over the shared length",
                "spearman_confusion_vs_identity": float(rho),
                "mantel_p": float(p_mantel), "permutations": int(args.permutations)},
            "trimming": {
                "per_gene": "mean 3' V deletion under the OLGA model vs 1 - self posterior mass",
                "spearman": float(tr.correlation), "p": float(tr.pvalue)},
        }
        print(f"[{chain}] {len(names)} V genes")
        print(f"  confusion vs CDR3-proximal germline identity: Spearman {rho:.3f}, "
              f"Mantel p = {p_mantel:.4f} ({args.permutations} permutations)")
        print(f"  mean V trimming vs posterior mass leaving the true gene: Spearman "
              f"{tr.correlation:.3f}, p = {tr.pvalue:.3g}")
    _write("3d_germline_and_trimming", report)


# --------------------------------------------------------------------------- #
def _write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1))
    print(f"wrote {path}")


def main():
    _selfcheck()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "usage", "ambiguity", "pairing", "antigen", "germline"])
    ap.add_argument("--n", type=int, default=10000, help="pairing: test pairs")
    ap.add_argument("--workers", type=int, default=64, help="pairing: OLGA workers")
    ap.add_argument("--permutations", type=int, default=9999, help="germline: Mantel permutations")
    ap.add_argument("--pair-permutations", type=int, default=200,
                    help="pairing: shuffled-pairing replicates for the null")
    args = ap.parse_args()
    {"cache": cmd_cache, "usage": cmd_3a, "ambiguity": cmd_3b, "pairing": cmd_3c_pairing,
     "antigen": cmd_3c_antigen, "germline": cmd_3d}[args.cmd](args)


if __name__ == "__main__":
    main()
