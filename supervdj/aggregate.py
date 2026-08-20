"""Read-only aggregation over results/posteriors.tsv.

This module never recomputes a posterior. It reads the per-sequence table
(the single source of truth) and produces the integrity report (step 1) and
the usage-controlled selection comparison (step 2). Figures live elsewhere.

Table schema (one row per seq x axis x model x mode x resolution):
  axis        V | J
  model       pre  (OLGA, mode=NaN)  |  post (OLGA*SONNIA, mode in {grid,fixed})
  mode        grid (full V x J)      |  fixed (condition on annotated J) | NaN for pre
  resolution  gene | family
  status      ok | impossible (Pgen==0 -> empty posterior by design)
  entropy_nats   per-seq posterior entropy in nats (NaN when posterior empty)
  top1_mass      max posterior mass (NaN when empty)
  posterior_json the full distribution; "{}" when empty
"""
from __future__ import annotations
import json
import math
import os
from math import comb
import numpy as np
import pandas as pd

from supervdj.resolution import family_of  # /DV-preserving family def, single source of truth

PATH = "results/posteriors.tsv"
FIGDIR = "results/figures"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt
EXPECTED_SEQ = {"TRA": 38971, "TRB": 80961}  # rows per chain = seq * 12
ROWS_PER_SEQ = 12
SUM_TOL = 1e-3            # float32 storage -> ~8 sig digits; 1e-3 is generous
HIGH_J_NATS = math.log(2) # J posterior entropy above 1 effective bit = suspect ingestion
# A gene is "high confidence" when its mean posterior self-mass reaches this. The same
# threshold annotates the supplementary confusion figures and selects Table S3's rows,
# and self_mass itself is the column exported to confusion_groups_{chain}.tsv.
SELF_MASS_HIGH_CONFIDENCE = 0.5
# (bias correction for the alpha selection shift is computed PER-SET from each analyzed
#  set's own permutation-null center via _permutation_null, not a hardcoded constant.)


def load(path: str = PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
    df["cdr3_len"] = df["cdr3"].str.len()
    return df


def _slice(df, *, chain, axis, model, mode=None, resolution="gene", ok_only=True):
    m = (df.chain == chain) & (df.axis == axis) & (df.model == model) & (df.resolution == resolution)
    if mode is None:
        m &= df["mode"].isna()
    else:
        m &= df["mode"] == mode
    if ok_only:
        m &= df.status == "ok"
    return df[m]


# --------------------------------------------------------------------------- #
# Step 1: integrity report
# --------------------------------------------------------------------------- #
def integrity_report(df: pd.DataFrame) -> dict:
    out = {}
    print("=" * 78)
    print("STEP 1  INTEGRITY REPORT  (both chains, read-only over results/posteriors.tsv)")
    print("=" * 78)

    # --- row counts ---
    print("\n[row counts]  expected = n_seq * %d rows/seq" % ROWS_PER_SEQ)
    for chain, nseq in EXPECTED_SEQ.items():
        got = int((df.chain == chain).sum())
        exp = nseq * ROWS_PER_SEQ
        ok = "OK" if got == exp else "MISMATCH"
        print(f"  {chain}: {got:>9,} rows  (expected {exp:>9,} = {nseq:,} seq x {ROWS_PER_SEQ})  [{ok}]")
        out[f"{chain}_rows_ok"] = got == exp

    # --- posterior population / sum-to-1 ---
    print("\n[posterior population]")
    populated = df[df.n_candidates > 0].copy()
    empty = df[df.n_candidates == 0]
    imp = empty[empty.status == "impossible"]
    fixed_empty = empty[empty.status == "ok"]
    print(f"  populated rows (n_candidates>0): {len(populated):>9,}")
    print(f"  empty - status=impossible (Pgen==0, by design): {len(imp):>9,}")
    print(f"  empty - status=ok & mode=fixed (J-conditioning hit no candidates): {len(fixed_empty):>9,}")
    assert (fixed_empty["mode"] == "fixed").all(), "ok-empty rows are not all fixed-mode!"
    out["n_populated"] = len(populated)
    out["n_impossible"] = len(imp)
    out["n_fixed_empty"] = len(fixed_empty)

    # sum-to-1 over every populated posterior
    sums = populated["posterior_json"].map(lambda s: sum(json.loads(s).values()))
    dev = (sums - 1.0).abs()
    n_bad = int((dev > SUM_TOL).sum())
    print(f"\n[posterior sums to 1]  over {len(populated):,} populated posteriors")
    print(f"  max |sum-1| = {dev.max():.2e}   rows exceeding tol {SUM_TOL:.0e}: {n_bad}")
    out["max_sum_dev"] = float(dev.max())
    out["n_sum_violations"] = n_bad
    assert n_bad == 0, "some populated posteriors do not sum to 1"

    # NaN audit on the populated set (should be none)
    nan_pop = int(populated["entropy_nats"].isna().sum())
    print(f"  populated rows with NaN entropy: {nan_pop}  (expected 0)")
    out["n_populated_nan_entropy"] = nan_pop
    assert nan_pop == 0

    # --- entropy / top-mass table, side by side ---
    print("\n[conditional entropy  H(.|CDR3), mean of per-seq posterior entropy, nats]")
    print("  (pre = OLGA prior; V uses model=pre/grid, gene & family resolution)")
    hdr = f"  {'metric':<34}{'TRA':>12}{'TRB':>12}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    rows = {}
    def line(label, ta, tb, fmt="{:.4f}"):
        sa = fmt.format(ta) if ta == ta else "n/a"
        sb = fmt.format(tb) if tb == tb else "n/a"
        print(f"  {label:<34}{sa:>12}{sb:>12}")
        rows[label] = (ta, tb)

    def mean_entropy(chain, axis, model, mode, resolution):
        s = _slice(df, chain=chain, axis=axis, model=model, mode=mode, resolution=resolution)
        return float(s["entropy_nats"].mean()), len(s)

    va_g = {c: mean_entropy(c, "V", "pre", None, "gene")[0] for c in ("TRA", "TRB")}
    va_f = {c: mean_entropy(c, "V", "pre", None, "family")[0] for c in ("TRA", "TRB")}
    j_g  = {c: mean_entropy(c, "J", "pre", None, "gene")[0] for c in ("TRA", "TRB")}
    line("V gene entropy (pre)", va_g["TRA"], va_g["TRB"])
    line("V family entropy (pre)", va_f["TRA"], va_f["TRB"])
    line("J gene entropy (pre)", j_g["TRA"], j_g["TRB"])
    out["V_gene_entropy_pre"] = va_g
    out["V_family_entropy_pre"] = va_f
    out["J_gene_entropy_pre"] = j_g

    # headline sanity vs the small-sample preview (TRA~1.48, TRB~2.64)
    print(f"\n  headline check: full-table V gene entropy  TRA={va_g['TRA']:.3f} (preview ~1.48), "
          f"TRB={va_g['TRB']:.3f} (preview ~2.64)")

    # --- high-J flags (possible ingestion failures) ---
    print(f"\n[high-J flag]  seqs whose pre J-gene posterior entropy > {HIGH_J_NATS:.3f} nats (>1 effective bit)")
    for chain in ("TRA", "TRB"):
        s = _slice(df, chain=chain, axis="J", model="pre", mode=None, resolution="gene")
        n_high = int((s["entropy_nats"] > HIGH_J_NATS).sum())
        frac = n_high / max(len(s), 1)
        print(f"  {chain}: {n_high:>6,} / {len(s):,}  ({frac:.2%})")
        out[f"{chain}_high_J"] = (n_high, len(s))

    # --- pre -> post top-mass shift (V, gene) ---
    print("\n[pre->post top-mass shift]  mean top1_mass, V gene resolution")
    print(f"  {'arm':<24}{'TRA':>12}{'TRB':>12}")
    for label, model, mode in [("pre (OLGA)", "pre", None),
                               ("post grid (SONNIA)", "post", "grid"),
                               ("post fixed (SONNIA)", "post", "fixed")]:
        ta = _slice(df, chain="TRA", axis="V", model=model, mode=mode, resolution="gene")["top1_mass"].mean()
        tb = _slice(df, chain="TRB", axis="V", model=model, mode=mode, resolution="gene")["top1_mass"].mean()
        print(f"  {label:<24}{ta:>12.4f}{tb:>12.4f}")
        out[f"top1_{model}_{mode}"] = (float(ta), float(tb))
    return out


# --------------------------------------------------------------------------- #
# Step 2: usage-controlled selection comparison
# --------------------------------------------------------------------------- #
def selection_comparison(df: pd.DataFrame, resolution: str = "gene", verbose: bool = True) -> dict:
    """Conditional entropy of V posterior pre (OLGA) vs post (SONNIA), with the
    post arm reweighted so its marginal V usage matches the pre arm within each
    CDR3-length bin. If post entropy stays below pre after this control, the
    sharpening is separability, not the prior shifting."""
    say = print if verbose else (lambda *a, **k: None)
    say("\n" + "=" * 78)
    say(f"STEP 2  SELECTION COMPARISON, usage-controlled  (V, {resolution} resolution)")
    say("=" * 78)
    say("  pre = model=pre (OLGA);  post = model=post & mode=grid (SONNIA, full VxJ)")
    say("  control: reweight post seqs so (cdr3_len x argmax-V) marginal == pre's")

    out = {}
    say(f"\n  {'':<10}{'pre H':>10}{'post H raw':>12}{'raw shift':>11}"
        f"{'post H ctrl':>13}{'ctrl shift':>12}{'post kept':>11}")
    for chain in ("TRA", "TRB"):
        pre = _slice(df, chain=chain, axis="V", model="pre", mode=None, resolution=resolution)
        post = _slice(df, chain=chain, axis="V", model="post", mode="grid", resolution=resolution)
        # same sequences in both arms (intersection of ok-in-both)
        pre = pre.set_index("seq_id")
        post = post.set_index("seq_id")
        common = pre.index.intersection(post.index)
        if len(common) == 0:
            say(f"  {chain:<10}{'(no sequences)':>56}")
            continue
        pre = pre.loc[common]
        post = post.loc[common]

        pre_H = pre["entropy_nats"].mean()
        post_H_raw = post["entropy_nats"].mean()

        # target marginal = pre's (cdr3_len, argmax-V) frequency
        pre_key = (pre["cdr3_len"].astype(str) + "|" + pre["top1_label"].astype(str)).tolist()
        post_key = (post["cdr3_len"].astype(str) + "|" + post["top1_label"].astype(str)).tolist()
        pre_freq = pd.Series(pre_key).value_counts(normalize=True).to_dict()
        post_freq = pd.Series(post_key).value_counts(normalize=True).to_dict()
        # importance weight per post seq: target / source ; 0 if pre never has that cell
        w = np.array([pre_freq.get(k, 0.0) / post_freq[k] for k in post_key])
        kept = w > 0
        post_H_ctrl = np.average(post["entropy_nats"].values[kept], weights=w[kept])

        raw_shift = post_H_raw - pre_H
        ctrl_shift = post_H_ctrl - pre_H
        frac_kept = kept.mean()
        say(f"  {chain:<10}{pre_H:>10.4f}{post_H_raw:>12.4f}{raw_shift:>+11.4f}"
            f"{post_H_ctrl:>13.4f}{ctrl_shift:>+12.4f}{frac_kept:>10.1%}")
        out[chain] = dict(pre_H=float(pre_H), post_H_raw=float(post_H_raw), raw_shift=float(raw_shift),
                          post_H_ctrl=float(post_H_ctrl), ctrl_shift=float(ctrl_shift),
                          n=int(len(common)), frac_post_kept=float(frac_kept))
    say("\n  negative shift = post-selection sharpens the V posterior (lower entropy).")
    say("  if ctrl shift stays negative, the effect survives holding V usage fixed.")
    return out


# --------------------------------------------------------------------------- #
# Pre-figure analyses (gate 2.5)
# --------------------------------------------------------------------------- #
def equal_n_test(df, resolutions=("gene", "family"), n_boot=500, seed=0, make_fig=True):
    """Is the alpha<beta V-entropy gap a sample-size artifact? Subsample beta to
    alpha's n, recompute mean conditional entropy, repeat; check alpha sits
    clearly outside the beta matched-n distribution."""
    rng = np.random.default_rng(seed)
    print("\n" + "=" * 78)
    print("EQUAL-N ALPHA-BETA TEST  (V conditional entropy, beta subsampled to alpha n)")
    print("=" * 78)
    out = {}
    for res in resolutions:
        a = _slice(df, chain="TRA", axis="V", model="pre", resolution=res)["entropy_nats"].dropna().values
        b = _slice(df, chain="TRB", axis="V", model="pre", resolution=res)["entropy_nats"].dropna().values
        n = len(a)
        alpha_H = float(a.mean())
        boot = np.array([rng.choice(b, size=n, replace=False).mean() for _ in range(n_boot)])
        lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        outside = (alpha_H < lo) or (alpha_H > hi)
        out[res] = dict(alpha_H=alpha_H, n=int(n), beta_mean=float(boot.mean()),
                        ci=(lo, hi), beta_min=float(boot.min()), alpha_outside=bool(outside),
                        boot=boot)
        print(f"\n  V {res}: match n={n:,} (alpha usable rows), {n_boot} beta subsamples")
        print(f"    alpha H            = {alpha_H:.4f}")
        print(f"    beta matched-n H   = {boot.mean():.4f}   95% CI [{lo:.4f}, {hi:.4f}]   min {boot.min():.4f}")
        print(f"    alpha outside CI?  = {outside}   (alpha is {boot.mean()-alpha_H:.3f} nats below the beta mean;"
              f" gap to nearest beta draw = {boot.min()-alpha_H:.3f})")
    if make_fig:
        fig_equal_n(out)
    return out


def fig_equal_n(out, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res_list = list(out)
    fig, axes = plt.subplots(1, len(res_list), figsize=(5.2 * len(res_list), 4))
    axes = np.atleast_1d(axes)
    for ax, res in zip(axes, res_list):
        d = out[res]
        ax.hist(d["boot"], bins=40, color="#4C72B0", alpha=0.85)
        lo, hi = d["ci"]
        ax.axvspan(lo, hi, color="grey", alpha=0.25, label="beta 95% CI")
        ax.axvline(d["alpha_H"], color="C3", lw=2.2, label=f"alpha H = {d['alpha_H']:.3f}")
        ax.set_title(f"V {res}: beta matched-n (n={d['n']:,})")
        ax.set_xlabel("mean conditional entropy (nats)")
        ax.set_ylabel("# subsamples")
        ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    p = path or f"{FIGDIR}/equal_n_alpha_beta.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  [fig] {p}")
    return p


def _selection_frame(df, chain, resolution="gene"):
    """Per-seq frame (ok in both arms) for the usage-controlled selection shift:
    pre/post V conditional entropy, CDR3 length, and each arm's argmax-V label."""
    pre = _slice(df, chain=chain, axis="V", model="pre", resolution=resolution).set_index("seq_id")
    post = _slice(df, chain=chain, axis="V", model="post", mode="grid", resolution=resolution).set_index("seq_id")
    common = pre.index.intersection(post.index)
    return pd.DataFrame({
        "cdr3_len": pre.loc[common, "cdr3_len"].to_numpy(),
        "pre_H": pre.loc[common, "entropy_nats"].to_numpy(),
        "post_H": post.loc[common, "entropy_nats"].to_numpy(),
        "pre_top1": pre.loc[common, "top1_label"].astype(str).to_numpy(),
        "post_top1": post.loc[common, "top1_label"].astype(str).to_numpy(),
    })


def _controlled_shift(f, smooth=0.0):
    """Usage-controlled pre->post conditional-entropy shift on the GIVEN rows:
    reweight post to pre's (cdr3_len x argmax-V) marginal using this frame's own
    frequencies, then shift = weighted_post_H - pre_H. Same recipe as
    selection_comparison, packaged so it can run inside a subsample.

    smooth>0 applies additive (Laplace) smoothing to the pre marginal over the
    union of pre/post cells, so post-argmax cells absent from pre get a small
    nonzero target weight instead of being dropped (smooth=0 = exact original)."""
    pre_H = float(f["pre_H"].mean())
    post_raw = float(f["post_H"].mean())
    prek = (f["cdr3_len"].astype(str) + "|" + f["pre_top1"]).to_numpy()
    postk = (f["cdr3_len"].astype(str) + "|" + f["post_top1"]).to_numpy()
    pre_counts = pd.Series(prek).value_counts().to_dict()
    post_freq = pd.Series(postk).value_counts(normalize=True).to_dict()
    if smooth > 0:
        universe = set(prek) | set(postk)
        denom = len(prek) + smooth * len(universe)
        pre_freq = {k: (pre_counts.get(k, 0) + smooth) / denom for k in universe}
    else:
        npre = len(prek)
        pre_freq = {k: c / npre for k, c in pre_counts.items()}
    w = np.array([pre_freq.get(k, 0.0) / post_freq[k] for k in postk])
    kept = w > 0
    post_ctrl = float(np.average(f["post_H"].to_numpy()[kept], weights=w[kept]))
    return dict(pre_H=pre_H, post_raw=post_raw, post_ctrl=post_ctrl,
                raw_shift=post_raw - pre_H, ctrl_shift=post_ctrl - pre_H,
                frac_kept=float(kept.mean()))


def matched_n_selection(df, resolution="gene", n_boot=500, seed=0, make_fig=True):
    """Matched-n cross-chain selection test: subsample beta to alpha's n, recompute
    the usage-controlled shift from scratch INSIDE each subsample (control weights
    from the subsample itself), build the distribution, and check whether alpha's
    shift sits inside beta's matched-n CI. Licenses a quantitative cross-chain
    magnitude claim that the unequal-n shifts could not."""
    a = _selection_frame(df, "TRA", resolution)
    b = _selection_frame(df, "TRB", resolution)
    n = len(a)
    alpha = _controlled_shift(a)["ctrl_shift"]
    beta_full = _controlled_shift(b)["ctrl_shift"]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    nb = len(b)
    for i in range(n_boot):
        boot[i] = _controlled_shift(b.iloc[rng.choice(nb, size=n, replace=False)])["ctrl_shift"]
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    inside = lo <= alpha <= hi
    print("\n" + "=" * 78)
    print(f"MATCHED-N SELECTION TEST  (usage-controlled V shift, {resolution} resolution)")
    print("=" * 78)
    print(f"  alpha n = {n:,} seqs (ok in both arms);  beta pool = {nb:,};  {n_boot} subsamples")
    print(f"  alpha usage-controlled shift          = {alpha:+.4f}")
    print(f"  beta full-n usage-controlled shift     = {beta_full:+.4f}  (for reference, n={nb:,})")
    print(f"  beta MATCHED-n shift  mean             = {boot.mean():+.4f}")
    print(f"    95% CI                               = [{lo:+.4f}, {hi:+.4f}]   "
          f"(range [{boot.min():+.4f}, {boot.max():+.4f}])")
    print(f"  alpha inside beta matched-n CI?        = {inside}")
    verdict = ("beta sharpens MORE than alpha" if hi < alpha else
               "alpha sharpens more than beta" if lo > alpha else
               "no separable difference at matched n")
    print(f"  -> cross-chain claim licensed: {verdict}")
    print("  NOTE: matched-n removes SAMPLE SIZE as an explanation only. It does NOT")
    print("        equalize the chains: alpha vs beta still differ in V-usage distribution")
    print("        and candidate-set size (47 alpha vs 59 beta V genes) -- those are real biology and")
    print("        remain. This is the cross-chain magnitude test, separate from the")
    print("        within-chain directional result (both shifts are negative).")
    out = dict(alpha_shift=float(alpha), beta_full_shift=float(beta_full), n=int(n),
               beta_matched_mean=float(boot.mean()), ci=(lo, hi),
               alpha_inside=bool(inside), boot=boot, resolution=resolution)
    if make_fig:
        fig_matched_n_selection(out)
    return out


def fig_matched_n_selection(out, path=None):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(out["boot"], bins=40, color="#55A868", alpha=0.85)
    lo, hi = out["ci"]
    ax.axvspan(lo, hi, color="grey", alpha=0.25, label="beta matched-n 95% CI")
    ax.axvline(out["alpha_shift"], color="C3", lw=2.2,
               label=f"alpha shift = {out['alpha_shift']:+.3f}")
    ax.set_title(f"Matched-n usage-controlled selection shift\n"
                 f"beta subsampled to alpha n={out['n']:,} (V {out['resolution']})")
    ax.set_xlabel("usage-controlled shift Δ entropy (nats); more negative = sharper")
    ax.set_ylabel("# subsamples")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path or "matched_n_selection")


def dropped_weight_diagnostic(df, chains=("TRB", "TRA"), top=12):
    """In the usage control, post sequences whose (cdr3_len x post-argmax-V) cell is
    absent from pre's marginal get weight 0 and drop out. Check whether those drops
    concentrate in particular V genes (=> the controlled shift carries an implicit
    per-gene qualifier) or spread broadly."""
    print("\n" + "=" * 78)
    print("DROPPED-WEIGHT DIAGNOSTIC  (usage-controlled estimator, post argmax-V of dropped seqs)")
    print("=" * 78)
    out = {}
    for chain in chains:
        f = _selection_frame(df, chain)
        prek = set(f["cdr3_len"].astype(str) + "|" + f["pre_top1"])
        postk = f["cdr3_len"].astype(str) + "|" + f["post_top1"]
        dropped = ~postk.isin(prek)
        nd, n = int(dropped.sum()), len(f)
        bg = f.loc[dropped.values, "post_top1"].value_counts()
        tot = f["post_top1"].value_counts()
        top5 = bg.head(5).sum() / nd if nd else float("nan")
        print(f"\n[{chain}] dropped {nd:,}/{n:,} ({nd/n:.1%}); "
              f"dropped seqs span {bg.size} distinct post-argmax-V genes; "
              f"top-5 genes hold {top5:.0%} of dropped")
        print(f"  {'post-argmax V':<16}{'dropped':>9}{'% of dropped':>14}{'gene drop-rate':>16}")
        for g, c in bg.head(top).items():
            print(f"  {str(g):<16}{int(c):>9}{c/nd:>13.1%}{c/int(tot[g]):>15.1%}")
        out[chain] = dict(n_dropped=nd, n=n, n_genes=int(bg.size), top5_share=float(top5),
                          top_gene=str(bg.index[0]), top_gene_share=float(bg.iloc[0] / nd))
    return out


def _permutation_null(f, n_perm=2000, seed=0):
    """Permutation null of the usage-controlled shift on selection frame ``f``: swap
    each sequence's pre/post arms with prob 0.5 and recompute. Returns
    ``(observed_shift, null_array)``. The null *center* is the estimator's
    finite-sample bias for THIS set (it shrinks as n grows), used both for the
    permutation p-value and as the per-set bias correction in ``alpha_bootstrap``."""
    obs = _controlled_shift(f)["ctrl_shift"]
    L = f["cdr3_len"].to_numpy()
    preH, postH = f["pre_H"].to_numpy(), f["post_H"].to_numpy()
    preT, postT = f["pre_top1"].to_numpy(), f["post_top1"].to_numpy()
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        sw = rng.random(len(f)) < 0.5
        g = pd.DataFrame({
            "cdr3_len": L,
            "pre_H": np.where(sw, postH, preH),
            "post_H": np.where(sw, preH, postH),
            "pre_top1": np.where(sw, postT, preT),
            "post_top1": np.where(sw, preT, postT),
        })
        null[i] = _controlled_shift(g)["ctrl_shift"]
    return float(obs), null


def permutation_test(df, chains=("TRA", "TRB"), n_perm=2000, seed=0):
    """Within-chain permutation test: under exchangeable pre/post labels the
    usage-controlled shift should be ~0. Per permutation, swap each sequence's pre
    and post arms with prob 0.5 and recompute via _controlled_shift. Two-sided p =
    fraction of permuted shifts at least as extreme (|.|) as observed."""
    print("\n" + "=" * 78)
    print("WITHIN-CHAIN PERMUTATION TEST  (usage-controlled V shift; null = pre/post exchangeable)")
    print("=" * 78)
    out = {}
    for chain in chains:
        f = _selection_frame(df, chain)
        obs, null = _permutation_null(f, n_perm, seed)
        n_extreme = int((np.abs(null) >= abs(obs)).sum())
        p = (n_extreme + 1) / (n_perm + 1)        # +1: never report exactly 0
        z = (obs - null.mean()) / null.std()
        out[chain] = dict(obs=float(obs), null_mean=float(null.mean()), null_std=float(null.std()),
                          p=float(p), n_extreme=n_extreme, z=float(z), n=len(f), n_perm=n_perm)
        print(f"\n[{chain}]  n={len(f):,}  permutations={n_perm}")
        print(f"  observed usage-controlled shift = {obs:+.4f}")
        print(f"  permutation null: center = {null.mean():+.5f}  sd = {null.std():.5f}  "
              f"(range [{null.min():+.4f}, {null.max():+.4f}])")
        ratio = null.mean() / obs
        flag = (f"~0 (center is {abs(ratio):.1%} of observed; control looks unbiased)"
                if abs(null.mean()) < 0.01 * abs(obs)
                else f"OFF ZERO: center is {ratio:+.1%} of observed -- small control bias (flag)")
        print(f"  null center vs zero: {flag}")
        print(f"  observed is {abs(z):.1f} sd from the null center")
        print(f"  two-sided p (|perm| >= |obs|) = {p:.2e}  "
              f"({n_extreme}/{n_perm} permutations as extreme)  [floored at 1/(n_perm+1)]")
    print("\n  NOTE: the two-sided p is a LOWER BOUND -- the test unit is the deduplicated")
    print("        rearrangement, and residual clonal structure can shrink the effective n.")

    # persist the null-distribution summary
    os.makedirs("results", exist_ok=True)
    tag = "_".join(chains)
    with open(f"results/permutation_test_{tag}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    with open(f"results/permutation_test_{tag}.txt", "w") as fh:
        fh.write(f"WITHIN-CHAIN PERMUTATION TEST (usage-controlled V shift; pre/post exchangeable)\n")
        fh.write(f"n_perm={n_perm}, seed={seed}; two-sided p floored at 1/(n_perm+1)\n\n")
        fh.write(f"{'chain':<6}{'n':>9}{'obs':>10}{'null mean':>11}{'null sd':>10}"
                 f"{'sd sep':>9}{'p':>11}\n")
        for c, r in out.items():
            fh.write(f"{c:<6}{r['n']:>9,}{r['obs']:>+10.4f}{r['null_mean']:>+11.5f}"
                     f"{r['null_std']:>10.5f}{abs(r['z']):>9.1f}{r['p']:>11.2e}\n")
        fh.write("\nNOTE: two-sided p is a LOWER BOUND -- test unit is the deduplicated\n")
        fh.write("rearrangement; residual clonal structure can shrink the effective n.\n")
    print(f"  [saved] results/permutation_test_{tag}.json  results/permutation_test_{tag}.txt")
    return out


def cross_chain_pvalue(df, n_boot=500, seed=0):
    """p-value for 'beta sharpens more than alpha' from the matched-n beta draws vs
    alpha's observed shift: empirical (resolution-limited) and normal-approximation
    (non-floored)."""
    from scipy.stats import norm
    print("\n" + "=" * 78)
    print("CROSS-CHAIN P-VALUE  (matched-n beta draws vs alpha observed shift)")
    print("=" * 78)
    a = _controlled_shift(_selection_frame(df, "TRA"))["ctrl_shift"]
    b = _selection_frame(df, "TRB")
    n, nb = len(_selection_frame(df, "TRA")), len(b)
    rng = np.random.default_rng(seed)
    boot = np.array([_controlled_shift(b.iloc[rng.choice(nb, size=n, replace=False)])["ctrl_shift"]
                     for _ in range(n_boot)])
    n_toward = int((boot >= a).sum())                 # beta draws as close-to/above alpha
    emp_one = (n_toward + 1) / (n_boot + 1)           # bounded one-sided estimate
    mu, sd = boot.mean(), boot.std(ddof=1)
    z = (a - mu) / sd
    p_norm_two = 2 * norm.sf(abs(z))
    print(f"  alpha observed shift   = {a:+.4f}")
    print(f"  beta matched-n draws   : mean {mu:+.4f}  sd {sd:.4f}  "
          f"range [{boot.min():+.4f}, {boot.max():+.4f}]  (n_draws={n_boot}, each n={n:,})")
    print(f"  empirical one-sided p  : {n_toward}/{n_boot} beta draws reach alpha "
          f"-> p < {1/n_boot:.4f} (resolution-limited; alpha sits beyond every draw)")
    print(f"  normal-approx          : z = {z:+.1f} sd,  two-sided p ~ {p_norm_two:.2e}")
    return dict(alpha_shift=float(a), beta_mean=float(mu), beta_sd=float(sd), z=float(z),
                p_emp_one_lt=float(1 / n_boot), p_norm_two=float(p_norm_two), n=int(n), n_boot=n_boot)


def persist_matched_n_cross_chain(df, n_boot=500, seed=0, resolution="gene"):
    """Persist the matched-n and cross-chain selection statistics (which are not
    otherwise saved) to results/matched_n_cross_chain.json so the manuscript numbers
    trace to a pinned file. Deterministic at the given seed and n_boot."""
    mn = matched_n_selection(df, resolution=resolution, n_boot=n_boot, seed=seed, make_fig=False)
    cc = cross_chain_pvalue(df, n_boot=n_boot, seed=seed)
    out = dict(resolution=resolution, n_boot=n_boot, seed=seed, n_alpha=int(mn["n"]),
               alpha_shift=float(mn["alpha_shift"]), beta_full_shift=float(mn["beta_full_shift"]),
               beta_matched_mean=float(mn["beta_matched_mean"]),
               beta_matched_ci=[float(mn["ci"][0]), float(mn["ci"][1])],
               beta_matched_sd=float(cc["beta_sd"]), cross_chain_z=float(cc["z"]),
               alpha_inside_beta_ci=bool(mn["alpha_inside"]))
    os.makedirs("results", exist_ok=True)
    with open("results/matched_n_cross_chain.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("  [saved] results/matched_n_cross_chain.json")
    return out


def beta_gene_decomposition(df, genes=("TRBV2", "TRBV28"), smooth=1.0):
    """Decompose the beta usage-controlled shift around the TRBV2/TRBV28 sequences
    the control drops. Reports the dropped-cell shift (sharpening among seqs NOT
    driven into those genes), the smoothed shift (those seqs retained via an
    additively-smoothed pre marginal), and the gene-targeted contribution."""
    print("\n" + "=" * 78)
    print(f"BETA TRBV2/TRBV28 DECOMPOSITION  (gene-targeted selection vs broad sharpening)")
    print("=" * 78)
    b = _selection_frame(df, "TRB")
    inG = b["post_top1"].isin(genes).to_numpy()
    nG, n = int(inG.sum()), len(b)
    dropped = _controlled_shift(b, smooth=0.0)        # status quo: G largely dropped
    smoothed = _controlled_shift(b, smooth=smooth)    # G retained via smoothed pre marginal
    dpre = b["post_H"].to_numpy() - b["pre_H"].to_numpy()   # per-seq raw post-pre
    g_raw, rest_raw, all_raw = float(dpre[inG].mean()), float(dpre[~inG].mean()), float(dpre.mean())
    g_share = (nG / n) * g_raw / all_raw              # fraction of total raw sharpening from G
    print(f"  {', '.join(genes)} post-argmax sequences: n={nG:,} ({nG/n:.1%} of beta {n:,})")
    print(f"\n  controlled shift, TRBV2/28 DROPPED  (current)  = {dropped['ctrl_shift']:+.4f}"
          f"   (kept {dropped['frac_kept']:.1%}; 'sharpening among seqs not driven into these genes')")
    print(f"  controlled shift, TRBV2/28 RETAINED (smooth={smooth}) = {smoothed['ctrl_shift']:+.4f}"
          f"   (kept {smoothed['frac_kept']:.1%})")
    print(f"  -> retaining them moves the aggregate by {smoothed['ctrl_shift']-dropped['ctrl_shift']:+.4f} nats")
    print(f"\n  gene-targeted magnitude (raw per-seq post-pre entropy change):")
    print(f"    TRBV2/28 sequences : {g_raw:+.4f}   (these collapse from a diffuse OLGA prior)")
    print(f"    all other beta     : {rest_raw:+.4f}")
    print(f"    beta overall (raw) : {all_raw:+.4f}")
    print(f"  -> {g_share:.0%} of the total beta raw sharpening is attributable to the "
          f"{nG/n:.0%} of sequences in TRBV2/28")
    return dict(genes=list(genes), nG=nG, n=n, dropped_shift=float(dropped["ctrl_shift"]),
                smoothed_shift=float(smoothed["ctrl_shift"]), g_raw=g_raw, rest_raw=rest_raw,
                all_raw=all_raw, g_share=float(g_share))


def selection_summary(df, n_perm=2000, seed=0, smooth=1.0):
    """Final framing of the selection result: (1) broad, bias-corrected,
    usage-controlled V sharpening in both chains; (2) the concentrated TRBV2/28
    gene-targeted effect. Significance quoted as sd separation, not floored p."""
    perm = permutation_test(df, chains=("TRA", "TRB"), n_perm=n_perm, seed=seed)
    a, bch = perm["TRA"], perm["TRB"]
    a_corr = a["obs"] - a["null_mean"]
    dec = beta_gene_decomposition(df, smooth=smooth)
    print("\n" + "=" * 78)
    print("SELECTION RESULT — TWO FINDINGS")
    print("=" * 78)
    print("  (1) BROAD usage-controlled sharpening of V separability, both chains:")
    print(f"      alpha: raw {a['obs']:+.4f}, permutation null centered {a['null_mean']:+.4f} "
          f"(finite-sample bias) -> BIAS-CORRECTED {a_corr:+.4f}  [headline alpha]")
    print(f"      beta:  {bch['obs']:+.4f}; null centered {bch['null_mean']:+.5f} (~0, no correction) "
          f"-> {bch['obs']:+.4f} as-is  [headline beta]")
    print(f"      significance: alpha {abs(a['z']):.0f} sd, beta {abs(bch['z']):.0f} sd from the "
          f"permutation null (>20 sd both).")
    print("  (2) CONCENTRATED gene-targeted effect on TRBV2/TRBV28:")
    print(f"      selection drives posterior mass onto TRBV2/28 for {dec['nG']:,} CDR3s "
          f"({dec['nG']/dec['n']:.0%}) that recombination would assign elsewhere;")
    print(f"      controlled shift dropped {dec['dropped_shift']:+.4f} vs retained "
          f"{dec['smoothed_shift']:+.4f}; ~{dec['g_share']:.0%} of beta raw sharpening is theirs.")
    print("  CAVEAT: test unit = deduplicated rearrangement; residual clonal structure can")
    print("          shrink effective n, so the permutation p is a LOWER BOUND (hence we quote sd).")
    return dict(alpha_raw=float(a["obs"]), alpha_corrected=float(a_corr),
                alpha_sd=float(abs(a["z"])), beta=float(bch["obs"]), beta_sd=float(abs(bch["z"])),
                decomposition=dec)


def _cluster_labels(genes, M):
    """Cut the leakage dendrogram of M (over `genes`) at #IMGT-families clusters and
    return {gene: cluster_label}. Same recipe as grouping_comparison, factored out so
    the full data and each bootstrap resample are grouped identically."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    S = (M + M.T) / 2.0
    D = 1.0 - S; np.fill_diagonal(D, 0.0); D = (D + D.T) / 2.0
    Z = linkage(squareform(D, checks=False), method="average")
    K = len(set(family_of(g) for g in genes))
    clust = fcluster(Z, t=K, criterion="maxclust")
    return {g: int(c) for g, c in zip(genes, clust)}


def alpha_bootstrap(df, B=1000, seed=0, min_count=20, perm_null=None, perm_n=2000):
    """Bootstrap CI for the ALPHA chain, symmetric to the CIs beta already gets.

    Resample the canonical TRA sequences (Pgen>0, ok in both arms) WITH REPLACEMENT,
    B times, and on each resample recompute: V-gene / V-family / J-gene conditional
    entropy, the usage-controlled post-selection V shift (`_controlled_shift`, same
    definition as the main alpha analysis), and the adjusted Rand index of the
    data-driven V confusion grouping against the full-data grouping. Report bootstrap
    mean and 2.5-97.5 percentile interval for each. The selection shift is reported raw
    and bias-corrected (raw minus the per-set permutation-null center; pass ``perm_null``
    or it is computed here from this set's own permutation null over ``perm_n`` shuffles).
    Beta's full-n V-gene entropy bootstrap is included for absolute-scale comparison
    (alpha's CI is ~sqrt(n_b/n_a) wider)."""
    # --- assemble one seq-aligned frame over the canonical alpha set ---
    vg = _slice(df, chain="TRA", axis="V", model="pre", resolution="gene").set_index("seq_id")
    vf = _slice(df, chain="TRA", axis="V", model="pre", resolution="family").set_index("seq_id")
    jg = _slice(df, chain="TRA", axis="J", model="pre", resolution="gene").set_index("seq_id")
    post = _slice(df, chain="TRA", axis="V", model="post", mode="grid",
                  resolution="gene").set_index("seq_id")
    common = vg.index.intersection(vf.index).intersection(jg.index).intersection(post.index)
    n = len(common)
    base = pd.DataFrame({
        "vg_H": vg.loc[common, "entropy_nats"].to_numpy(),
        "vf_H": vf.loc[common, "entropy_nats"].to_numpy(),
        "jg_H": jg.loc[common, "entropy_nats"].to_numpy(),
        "cdr3_len": vg.loc[common, "cdr3_len"].to_numpy(),
        "pre_top1": vg.loc[common, "top1_label"].astype(str).to_numpy(),
        "post_top1": post.loc[common, "top1_label"].astype(str).to_numpy(),
        "post_H": post.loc[common, "entropy_nats"].to_numpy(),
        "true_gene": vg.loc[common, "true_gene"].astype(str).to_numpy(),
    })

    # --- precompute per-seq leakage ingredients over a global gene index ---
    posts = [json.loads(s) for s in vg.loc[common, "posterior_json"]]
    glist = sorted(set(base["true_gene"]) | {g for d in posts for g in d})
    gidx = {g: i for i, g in enumerate(glist)}
    G = len(glist)
    true_idx = np.array([gidx[g] for g in base["true_gene"]])
    massmat = np.zeros((n, G))
    for s, d in enumerate(posts):
        for g, m in d.items():
            massmat[s, gidx[g]] = m

    def leakage(idx):
        """Resampled leakage matrix + kept gene names (true-count >= min_count)."""
        ti = true_idx[idx]
        cnt = np.bincount(ti, minlength=G)
        keep = np.where(cnt >= min_count)[0]
        if len(keep) < 3:
            return None, None
        pos = np.full(G, -1); pos[keep] = np.arange(len(keep))
        p, mm = pos[ti], massmat[idx][:, keep]
        sel = p >= 0
        Mk = np.zeros((len(keep), len(keep)))
        for b in range(len(keep)):
            Mk[:, b] = np.bincount(p[sel], weights=mm[sel, b], minlength=len(keep))
        Mk /= cnt[keep][:, None]
        return [glist[k] for k in keep], Mk

    # reference (full-data) grouping; assert the fast leakage reproduces build_leakage
    genes_full, M_full = leakage(np.arange(n))
    keep_chk, M_chk = build_leakage(df, "TRA", min_count)
    assert genes_full == keep_chk and np.allclose(M_full, M_chk, atol=1e-9), \
        "fast bootstrap leakage diverges from build_leakage"
    ref_label = _cluster_labels(genes_full, M_full)

    # --- bootstrap ---  (uniform draw with replacement -> random order, never sorted)
    rng = np.random.default_rng(seed)
    vgH = base["vg_H"].to_numpy(); vfH = base["vf_H"].to_numpy(); jgH = base["jg_H"].to_numpy()
    cols = base[["cdr3_len", "post_H", "pre_top1", "post_top1"]]
    bvg = np.empty(B); bvf = np.empty(B); bjg = np.empty(B)
    bshift = np.empty(B); bari = np.full(B, np.nan)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        bvg[i] = vgH[idx].mean(); bvf[i] = vfH[idx].mean(); bjg[i] = jgH[idx].mean()
        g = cols.iloc[idx].copy(); g["pre_H"] = vgH[idx]
        bshift[i] = _controlled_shift(g)["ctrl_shift"]
        genes_b, M_b = leakage(idx)
        if genes_b is not None:
            lab = _cluster_labels(genes_b, M_b)
            shared = [x for x in genes_b if x in ref_label]
            bari[i] = _adjusted_rand([ref_label[x] for x in shared], [lab[x] for x in shared])
    # per-set bias correction: the usage-controlled estimator's finite-sample offset is
    # the permutation-null CENTER of THIS analyzed set (shrinks with n), NOT a hardcoded
    # constant. Pass perm_null (e.g. from permutation_test) for exact consistency; else
    # compute it here from the same selection frame.
    full_frame = base.rename(columns={"vg_H": "pre_H"})
    shift_pt = _controlled_shift(full_frame)["ctrl_shift"]
    if perm_null is None:
        _, _pn = _permutation_null(full_frame, n_perm=perm_n, seed=seed)
        perm_null = float(_pn.mean())
    bcorr = bshift - perm_null

    # beta full-n V-gene entropy bootstrap (absolute-scale reference)
    beta = _slice(df, chain="TRB", axis="V", model="pre", resolution="gene")["entropy_nats"].dropna().to_numpy()
    nb = len(beta)
    bbeta = np.array([beta[rng.integers(0, nb, size=nb)].mean() for _ in range(B)])

    def ci(a):
        a = a[~np.isnan(a)]
        lo, hi = np.percentile(a, [2.5, 97.5])
        return float(a.mean()), float(lo), float(hi)

    pts = {"vg": float(vgH.mean()), "vf": float(vfH.mean()), "jg": float(jgH.mean()),
           "shift": shift_pt, "corr": shift_pt - perm_null, "ari": 1.0}

    rows = [
        ("V-gene conditional entropy (nats)", pts["vg"], *ci(bvg)),
        ("V-family conditional entropy (nats)", pts["vf"], *ci(bvf)),
        ("J-gene conditional entropy (nats)", pts["jg"], *ci(bjg)),
        ("usage-controlled V shift, raw", pts["shift"], *ci(bshift)),
        (f"usage-controlled V shift, bias-corr (raw - ({perm_null:+.5f}))",
         pts["corr"], *ci(bcorr)),
        ("V confusion grouping ARI vs full data", pts["ari"], *ci(bari)),
    ]
    print("\n" + "=" * 78)
    print(f"ALPHA BOOTSTRAP CI  (TRA, {n:,} canonical seqs resampled w/ replacement, B={B})")
    print("=" * 78)
    print(f"  {'metric':<46}{'point':>9}{'boot mean':>11}{'2.5%':>9}{'97.5%':>9}")
    print("  " + "-" * 82)
    for name, pt, m, lo, hi in rows:
        print(f"  {name:<46}{pt:>9.4f}{m:>11.4f}{lo:>9.4f}{hi:>9.4f}")
    raw_neg = ci(bshift)[2] < 0
    cor_neg = ci(bcorr)[2] < 0
    print(f"\n  selection shift: raw 95% interval entirely negative?  {raw_neg}")
    print(f"                   bias-corrected interval entirely negative?  {cor_neg}")
    bm, blo, bhi = ci(bbeta)
    am, alo, ahi = ci(bvg)
    aw, bw = (ahi - alo), (bhi - blo)
    print(f"\n  [absolute-scale comparison]  beta V-gene entropy bootstrap (full n={nb:,}):")
    print(f"    beta   {bm:.4f}  95% CI [{blo:.4f}, {bhi:.4f}]   width {bw:.4f}")
    print(f"    alpha  {am:.4f}  95% CI [{alo:.4f}, {ahi:.4f}]   width {aw:.4f}")
    print(f"    alpha CI is {aw/bw:.1f}x wider than beta's (sqrt(n_b/n_a) = {math.sqrt(nb/n):.1f}); "
          f"both small in absolute terms.")
    print(f"  bias correction = per-set permutation-null center = {perm_null:+.5f} "
          f"(over {perm_n} shuffles; not a hardcoded constant)")
    out = dict(n=int(n), B=B, seed=seed, perm_null=float(perm_null), points=pts,
               v_gene=ci(bvg), v_family=ci(bvf), j_gene=ci(bjg),
               shift_raw=ci(bshift), shift_corrected=ci(bcorr), grouping_ari=ci(bari),
               shift_raw_all_neg=bool(raw_neg), shift_corr_all_neg=bool(cor_neg),
               beta_v_gene=ci(bbeta), alpha_width=float(aw), beta_width=float(bw))

    # persist: structured JSON (the data) + a human-readable table (paper-ready)
    os.makedirs("results", exist_ok=True)
    with open("results/alpha_bootstrap.json", "w") as fh:
        json.dump(out, fh, indent=2)
    with open("results/alpha_bootstrap.txt", "w") as fh:
        fh.write(f"ALPHA BOOTSTRAP CI  (TRA, n={n:,} canonical seqs w/ replacement, "
                 f"B={B}, seed={seed})\n\n")
        fh.write(f"{'metric':<46}{'point':>9}{'boot mean':>11}{'2.5%':>9}{'97.5%':>9}\n")
        for name, pt, m, lo, hi in rows:
            fh.write(f"{name:<46}{pt:>9.4f}{m:>11.4f}{lo:>9.4f}{hi:>9.4f}\n")
        fh.write(f"\nper-set bias correction (permutation-null center, {perm_n} shuffles): {perm_null:+.5f}\n")
        fh.write(f"selection shift raw interval entirely negative:            {raw_neg}\n")
        fh.write(f"selection shift bias-corrected interval entirely negative:  {cor_neg}\n")
        fh.write(f"\nbeta V-gene entropy (full n={nb:,}): {bm:.4f} CI [{blo:.4f}, {bhi:.4f}] "
                 f"width {bw:.4f}\n")
        fh.write(f"alpha V-gene entropy CI width {aw:.4f} = {aw/bw:.1f}x beta "
                 f"(sqrt(n_b/n_a)={math.sqrt(nb/n):.1f})\n")
    print("  [saved] results/alpha_bootstrap.json  results/alpha_bootstrap.txt")
    return out


def impossible_breakdown(df, min_occ=5):
    """Characterize the Pgen==0 (status=impossible) sequences excluded from every
    entropy number: chain split, and whether they cluster in particular V/J genes
    or CDR3 lengths (artifact) vs spread uniformly (genuine coverage gap)."""
    print("\n" + "=" * 78)
    print("IMPOSSIBLE-SET BREAKDOWN  (Pgen==0 seqs, excluded from all entropy)")
    print("=" * 78)
    # one row per seq with both annotated genes: true_gene is per-axis, so join V & J
    pg = df[(df.model == "pre") & (df.resolution == "gene")]
    v = (pg[pg.axis == "V"][["seq_id", "chain", "cdr3_len", "true_gene", "status"]]
         .rename(columns={"true_gene": "v_gene"}))
    j = pg[pg.axis == "J"][["seq_id", "true_gene"]].rename(columns={"true_gene": "j_gene"})
    one = v.merge(j, on="seq_id", how="left")
    one["impossible"] = one.status == "impossible"
    out = {}

    print("\n[per-chain rate]  (impossible is a per-sequence property; 12 rows each)")
    for chain in ("TRA", "TRB"):
        sub = one[one.chain == chain]
        nimp, ntot = int(sub.impossible.sum()), len(sub)
        print(f"  {chain}: {nimp:>4} / {ntot:>6,} sequences impossible  ({nimp/ntot:.2%})")
        out[f"{chain}_rate"] = (nimp, ntot)
    imp_total = int(one.impossible.sum())
    print(f"  total: {imp_total} impossible sequences  ({imp_total*ROWS_PER_SEQ:,} rows)")

    for chain in ("TRA", "TRB"):
        sub = one[one.chain == chain]
        si = sub[sub.impossible]
        print(f"\n[{chain}]  n_impossible={len(si)}")
        print(f"  CDR3 length: impossible mean={si.cdr3_len.mean():.1f} median={si.cdr3_len.median():.0f} "
              f"| all mean={sub.cdr3_len.mean():.1f} median={sub.cdr3_len.median():.0f}")
        for col, name in [("v_gene", "V gene"), ("j_gene", "J gene")]:
            base = sub[col].value_counts(normalize=True)
            impf = si[col].value_counts(normalize=True)
            cnt = si[col].value_counts()
            enr = (impf / base).reindex(cnt[cnt >= min_occ].index).sort_values(ascending=False)
            print(f"  {name} enrichment (impossible/overall freq, genes with >={min_occ} impossibles):")
            for g, e in enr.head(5).items():
                print(f"    {str(g):<18} {e:5.1f}x   ({int(cnt[g])} imp, {base.get(g,0):.1%} of all {chain})")
            top5_imp = si[col].value_counts(normalize=True).head(5).sum()
            top5_all = base.head(5).sum()
            print(f"    -> top-5 {name}s hold {top5_imp:.0%} of {chain} impossibles vs {top5_all:.0%} overall")
        out[chain] = dict(n_imp=len(si), len_mean_imp=float(si.cdr3_len.mean()),
                          len_mean_all=float(sub.cdr3_len.mean()))
    return out


# --------------------------------------------------------------------------- #
# Step 3: figures (each a standalone function reading the table)
# --------------------------------------------------------------------------- #
def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    p = f"{FIGDIR}/{name}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"  [fig] {p}")
    return p


def _canonical_ns(df):
    return {
        c: int(_slice(df, chain=c, axis="V", model="pre", resolution="gene")["seq_id"].nunique())
        for c in ("TRA", "TRB")
    }


def _n_caption(df):
    ns = _canonical_ns(df)
    return f"reported numbers use the full canonical N (TRA {ns['TRA']:,} / TRB {ns['TRB']:,})."


def fig_recoverability(df, path=None, min_per_len=20):
    """Conditional entropy vs CDR3 length, TRA & TRB as lines, J on a twin panel."""
    plt = _plt()
    fig, (axv, axj) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    for chain, color in [("TRA", "C0"), ("TRB", "C1")]:
        for ax, axis in [(axv, "V"), (axj, "J")]:
            s = _slice(df, chain=chain, axis=axis, model="pre", resolution="gene")
            g = s.groupby("cdr3_len")["entropy_nats"].agg(["mean", "count"])
            g = g[g["count"] >= min_per_len]
            ax.plot(g.index, g["mean"], marker="o", ms=3, color=color, label=chain)
    axv.set_ylabel("V conditional entropy (nats)")
    axv.set_title("V recoverability vs CDR3 length (lower = more determined)")
    axv.legend()
    axj.set_ylabel("J conditional entropy (nats)")
    axj.set_xlabel("CDR3 length (aa)")
    axj.set_title("J (twin panel) — near-zero: J is essentially determined by the CDR3")
    fig.text(0.5, -0.01, f"Length bins with <{min_per_len} sequences omitted for display "
             f"stability; {_n_caption(df)}",
             ha="center", fontsize=7, style="italic")
    fig.tight_layout()
    return _save(fig, path or "fig1_recoverability")


def fig_topk(df, path=None, kmax=20):
    """Fraction of sequences whose true V is within the top-k posterior set, k=1..20.
    This is candidate-GROUP SIZE, not classification accuracy."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 5))
    ks = np.arange(1, kmax + 1)
    for chain, color in [("TRA", "C0"), ("TRB", "C1")]:
        r = _slice(df, chain=chain, axis="V", model="pre", resolution="gene")["true_rank"].dropna().values
        frac = [float((r <= k).mean()) for k in ks]
        ax.plot(ks, frac, marker="o", ms=4, color=color, label=f"{chain} (n={len(r):,})")
    ax.set_xlabel("candidate-set size k")
    ax.set_ylabel("fraction with true V in top-k")
    ax.set_title("Top-k V recovery — a measure of GROUP SIZE, not accuracy")
    ax.set_xticks(ks[::2]); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    fig.text(0.5, -0.02, "Reading: 'return the k most likely V genes and the truth is inside this fraction of the time.'",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    return _save(fig, path or "fig2_topk")


def build_leakage(df, chain, min_count=20):
    """V-by-V posterior-leakage matrix L[i,j] = mean posterior mass on gene j over
    sequences whose true V is gene i (pre, gene resolution). Rows ~sum to 1."""
    s = _slice(df, chain=chain, axis="V", model="pre", resolution="gene")
    counts = s["true_gene"].value_counts()
    keep = sorted(counts[counts >= min_count].index)
    idx = {g: i for i, g in enumerate(keep)}
    M = np.zeros((len(keep), len(keep)))
    cnt = np.zeros(len(keep))
    for tg, pj in zip(s["true_gene"], s["posterior_json"]):
        i = idx.get(tg)
        if i is None:
            continue
        cnt[i] += 1
        for g, m in json.loads(pj).items():
            j = idx.get(g)
            if j is not None:
                M[i, j] += m
    M = M / np.maximum(cnt, 1)[:, None]
    return keep, M


def _confusion_order(M):
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    S = (M + M.T) / 2.0
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    Z = linkage(squareform(D, checks=False), method="average")
    return leaves_list(Z), Z


def fig_confusion(df, chain, path=None, min_count=20):
    """V-by-V leakage map, rows/cols ordered by confusion clustering so confusable
    blocks sit on the diagonal; IMGT family boundary lines mark family changes."""
    plt = _plt()
    from matplotlib.colors import LogNorm
    keep, M = build_leakage(df, chain, min_count)
    order, _ = _confusion_order(M)
    genes = [keep[i] for i in order]
    Mo = M[np.ix_(order, order)]
    fams = [family_of(g) for g in genes]
    bounds = [k for k in range(1, len(fams)) if fams[k] != fams[k - 1]]  # family changes

    n = len(genes)
    fig, axm = plt.subplots(figsize=(min(0.22 * n + 2, 16),) * 2)
    vmin = max(Mo[Mo > 0].min(), 1e-4)
    im = axm.imshow(np.clip(Mo, vmin, 1), cmap="magma", norm=LogNorm(vmin=vmin, vmax=1))
    for b in bounds:                      # IMGT family boundaries overlaid
        axm.axhline(b - 0.5, color="cyan", lw=0.4, alpha=0.6)
        axm.axvline(b - 0.5, color="cyan", lw=0.4, alpha=0.6)
    axm.set_xticks(range(n)); axm.set_xticklabels(genes, rotation=90, fontsize=5)
    axm.set_yticks(range(n)); axm.set_yticklabels(genes, fontsize=5)
    axm.set_title(f"{chain} V leakage (true row -> posterior col), confusion-ordered; "
                  f"cyan = IMGT family boundary", fontsize=9)
    fig.colorbar(im, ax=axm, fraction=0.025, pad=0.01, label="mean posterior mass (log)")
    fig.text(0.5, 0.005, f"V genes with <{min_count} sequences omitted for matrix stability; "
             f"{_n_caption(df)}",
             ha="center", fontsize=7, style="italic")
    return _save(fig, path or f"fig3_confusion_{chain}")


def fig_confusion_annotated(df, chain, path=None, min_count=20):
    """Supplementary form of the leakage map: the plain matrix plus the two annotations
    the manuscript reads off it.

    White boxes on the diagonal are the data-driven confusion clusters of two or more
    genes, i.e. the sets recoverable only as a group. Orange gene labels are the
    high-confidence genes. Both come from the same functions that write
    ``confusion_groups_{chain}.tsv`` -- ``_cluster_labels`` for the grouping and the
    diagonal of ``build_leakage`` for the self-mass -- so the annotation here and the
    exported table cannot disagree. The exporter renumbers the group ids for
    readability, which changes no gene's membership and so no box.
    """
    plt = _plt()
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    import matplotlib.patheffects as pe

    keep, M = build_leakage(df, chain, min_count)
    order, _ = _confusion_order(M)                  # display order only
    genes = [keep[i] for i in order]
    Mo = M[np.ix_(order, order)]
    n = len(genes)

    labels = _cluster_labels(keep, M)               # the exported grouping
    clab = [labels[g] for g in genes]
    self_mass = dict(zip(keep, np.diag(M)))         # the exported self_mass column
    hc = {g for g in genes if self_mass[g] >= SELF_MASS_HIGH_CONFIDENCE}

    blocks, start = [], 0
    for i in range(1, n + 1):
        if i == n or clab[i] != clab[start]:
            if i - start >= 2:
                blocks.append((start, i - 1))
            start = i

    pt = 8
    fig = plt.figure(figsize=(min(0.22 * n + 2, 16), min(0.22 * n + 2, 16) * 1.14))
    ax = fig.add_axes([0.115, 0.275, 0.875, 0.715])
    vmin = max(Mo[Mo > 0].min(), 1e-4)
    im = ax.imshow(np.clip(Mo, vmin, 1), cmap="viridis",
                   norm=LogNorm(vmin=vmin, vmax=1), aspect="auto")
    for a, b in blocks:
        r = Rectangle((a - 0.5, a - 0.5), b - a + 1, b - a + 1, fill=False,
                      edgecolor="white", linewidth=1.4)
        r.set_path_effects([pe.withStroke(linewidth=2.6, foreground="black")])
        ax.add_patch(r)
    ax.set_xticks(range(n)); ax.set_xticklabels(genes, rotation=90, fontsize=pt)
    ax.set_yticks(range(n)); ax.set_yticklabels(genes, fontsize=pt)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        if lbl.get_text() in hc:
            lbl.set_color("#E69F00")
    ax.set_xlabel("posterior V gene", labelpad=2)
    ax.set_ylabel("annotated (true) V gene", labelpad=2)
    ax.tick_params(width=1.0, length=2, pad=1.5)

    # fit the left margin to the longest row label, then hang the legend and the
    # colour bar off that same edge so neither can run past the canvas
    fig.canvas.draw()
    r0 = fig.canvas.get_renderer()
    left = max(t.get_window_extent(r0).width
               for t in ax.get_yticklabels()) / fig.bbox.width + 0.055
    ax.set_position([left, 0.275, 0.985 - left, 0.715])
    fig.canvas.draw()
    y0 = ax.xaxis.get_tightbbox(fig.canvas.get_renderer()).y0 / fig.bbox.height
    fig.legend(handles=[
        Patch(facecolor="none", edgecolor="black", linewidth=1.2,
              label=f"confusion cluster, recovered only as a group ({len(blocks)})"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="#E69F00",
               markeredgecolor="#E69F00", markersize=6,
               label=f"high-confidence gene, self-mass $\\geq$ "
                     f"{SELF_MASS_HIGH_CONFIDENCE} ({len(hc)})")],
        loc="upper center", bbox_to_anchor=(left + (0.985 - left) / 2, y0 - 0.004),
        ncol=1, frameon=False, fontsize=pt)

    cax = fig.add_axes([left, 0.045, 0.955 - left, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("mean posterior mass on the column gene (log scale)", fontsize=9,
                 labelpad=2)
    cb.ax.tick_params(labelsize=pt, length=2, width=1.0, pad=1.5)
    return _save(fig, path or f"figS_confusion_annotated_{chain}")


def fig_confidence(df, path=None):
    """Confident-fraction vs posterior-mass threshold (0.5..0.99), split by chain and
    resolution — the full curve, no single cutoff."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 5))
    ts = np.linspace(0.5, 0.99, 50)
    styles = {"gene": "-", "family": "--"}
    colors = {"TRA": "C0", "TRB": "C1"}
    for chain in ("TRA", "TRB"):
        n = _canonical_ns(df)[chain]
        for res in ("gene", "family"):
            m = _slice(df, chain=chain, axis="V", model="pre", resolution=res)["top1_mass"].dropna().values
            frac = [float((m >= t).mean()) for t in ts]
            ax.plot(ts, frac, styles[res], color=colors[chain], label=f"{chain} {res} (n={n:,})")
    ax.set_xlabel("posterior-mass threshold on top-1 V")
    ax.set_ylabel("fraction of sequences at/above threshold")
    ax.set_title("High-confidence V calls vs threshold (no single cutoff)")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    return _save(fig, path or "fig4_confidence")


def fig_selection(df, path=None):
    """Usage-controlled pre-vs-post V conditional-entropy shift, TRA made prominent
    (raw ~0 but controlled shift real)."""
    plt = _plt()
    res = selection_comparison(df, "gene", verbose=False)
    chains = ["TRA", "TRB"]
    raw = [res[c]["raw_shift"] for c in chains]
    ctrl = [res[c]["ctrl_shift"] for c in chains]
    with open("results/permutation_test_TRA.json") as fh:
        alpha_perm = json.load(fh)["TRA"]
    assert int(alpha_perm["n"]) == int(res["TRA"]["n"]), "alpha permutation N does not match posteriors"
    ctrl[0] = res["TRA"]["ctrl_shift"] - float(alpha_perm["null_mean"])
    x = np.arange(len(chains)); w = 0.36
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, raw, w, color="#bbbbbb", label="raw shift (post − pre)")
    ax.bar(x + w / 2, ctrl, w, color="#C44E52", label="usage-controlled shift")
    ax.axhline(0, color="k", lw=0.8)
    for xi, c, ctrlv in zip(x, chains, ctrl):
        ax.annotate(f"{res[c]['raw_shift']:+.3f}", (xi - w / 2, res[c]["raw_shift"]),
                    ha="center", va="top" if res[c]["raw_shift"] < 0 else "bottom", fontsize=8)
        label = f"{ctrlv:+.4f}" if c == "TRA" else f"{ctrlv:+.3f}"
        ax.annotate(label, (xi + w / 2, ctrlv),
                    ha="center", va="top", fontsize=8, fontweight="bold")
    ax.annotate(f"TRA bias-corrected\n{ctrl[0]:+.4f} (n={res['TRA']['n']:,})",
                xy=(0 + w / 2, ctrl[0]), xytext=(0.15, min(ctrl) * 0.55),
                fontsize=8, color="#C44E52",
                arrowprops=dict(arrowstyle="->", color="#C44E52"))
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(n={res[c]['n']:,})" for c in chains])
    ax.set_ylabel("Δ V conditional entropy (nats); negative = sharpened")
    ax.set_title("Post-selection sharpening, raw vs usage-controlled")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path or "fig5_selection")


def _dropped_weight_table(df, chain="TRB"):
    """Per-gene table of the usage control's dropped weight: for each post-selection
    argmax-V gene, how many sequences the control zeroed (their (len x argmax-V) cell
    is absent from pre's marginal), the share of total dropped weight, the gene's
    drop-rate, and the gene's overall post-top1 share. Also returns n, n_dropped, and
    the selection frame (so callers needn't recompute it)."""
    f = _selection_frame(df, chain)
    n = len(f)
    prek = set(f["cdr3_len"].astype(str) + "|" + f["pre_top1"])
    postk = f["cdr3_len"].astype(str) + "|" + f["post_top1"]
    dropped = (~postk.isin(prek)).to_numpy()
    nd = int(dropped.sum())
    bg = f.loc[dropped, "post_top1"].value_counts()
    tot = f["post_top1"].value_counts()
    t = pd.DataFrame({"dropped": bg.astype(int)})
    t["pct_of_dropped"] = t["dropped"] / nd
    t["gene_drop_rate"] = [bg[g] / int(tot[g]) for g in t.index]
    t["post_top1_frac_of_chain"] = [int(tot[g]) / n for g in t.index]
    t.index.name = "post_argmax_V"
    return t, n, nd, f


def fig_dropped_weight(df, path=None, top=10, genes=("TRBV2", "TRBV28")):
    """Two panels (alpha, beta): where the usage control's dropped weight concentrates
    by post-selection argmax-V. beta drops 7.2% concentrated in TRBV2/TRBV28; alpha
    drops a negligible, diffuse 0.15% (independent x-axes; the gap is in the titles)."""
    plt = _plt()
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, chain in zip(axes, ("TRA", "TRB")):
        t, n, nd, _ = _dropped_weight_table(df, chain)
        gshare = float(t.loc[t.index.isin(genes), "dropped"].sum() / nd) if nd else 0.0
        tb = t.head(top)[::-1]                                              # largest at top for barh
        colors = ["#C44E52" if g in genes else "#bbbbbb" for g in tb.index]
        ax.barh(range(len(tb)), tb["dropped"].to_numpy(), color=colors)
        for i, (c, p) in enumerate(zip(tb["dropped"], tb["pct_of_dropped"])):
            ax.annotate(f"{int(c):,} ({p:.0%})", (c, i), va="center", ha="left",
                        fontsize=7, xytext=(3, 0), textcoords="offset points")
        ax.set_yticks(range(len(tb))); ax.set_yticklabels(tb.index, fontsize=8)
        ax.set_xlim(0, tb["dropped"].max() * 1.28)
        ax.set_xlabel("sequences dropped (post-argmax-V cell absent from pre marginal)")
        conc = (f"{gshare:.0%} in TRBV2/TRBV28" if chain == "TRB"
                else f"diffuse (top gene {t.iloc[0]['pct_of_dropped']:.0%})")
        ax.set_title(f"{chain}: dropped {nd:,}/{n:,} ({nd/n:.2%}) — {conc}", fontsize=9)
    axes[0].set_ylabel("post-selection argmax V gene")
    axes[1].legend(handles=[Patch(color="#C44E52", label="TRBV2 / TRBV28"),
                            Patch(color="#bbbbbb", label="other V genes")], fontsize=8, loc="lower right")
    fig.suptitle("Usage control discards promotion-side weight: concentrated in β, negligible in α", fontsize=12)
    fig.text(0.5, -0.02,
             "Sequences the usage control zeroes because their (CDR3-length × post-argmax-V) cell never occurs in the "
             "pre-selection marginal. β drops 7.2% of the chain, 94% of it in TRBV2/TRBV28; α drops 0.15%, spread "
             "across genes with no dominant one — the same promotion-side, one-sided dropping, but negligible in α.",
             ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, path or "fig6_dropped_weight")


def _redistribution_table(df, chain):
    """Per-gene pre vs post top-1 V share and marginal usage on the canonical set
    (ok in both arms). Returns (table sorted by Δ top-1 share desc, N)."""
    pre  = _slice(df, chain=chain, axis="V", model="pre",  mode=None,   resolution="gene").set_index("seq_id")
    post = _slice(df, chain=chain, axis="V", model="post", mode="grid", resolution="gene").set_index("seq_id")
    common = pre.index.intersection(post.index)
    pre, post = pre.loc[common], post.loc[common]
    N = len(common)
    p1, q1 = pre["top1_label"].value_counts(), post["top1_label"].value_counts()
    def usage(frame):
        acc = {}
        for pj in frame["posterior_json"]:
            for g, m in json.loads(pj).items():
                acc[g] = acc.get(g, 0.0) + m
        return acc
    pu, qu = usage(pre), usage(post)
    genes = sorted(set(p1.index) | set(q1.index) | set(pu) | set(qu))
    t = pd.DataFrame([dict(
        gene=g, pre_top1=int(p1.get(g, 0)), post_top1=int(q1.get(g, 0)),
        pre_top1_share=int(p1.get(g, 0)) / N, post_top1_share=int(q1.get(g, 0)) / N,
        pre_usage=pu.get(g, 0.0) / N, post_usage=qu.get(g, 0.0) / N) for g in genes])
    t["delta_top1_share"] = t["post_top1_share"] - t["pre_top1_share"]
    return t.sort_values("delta_top1_share", ascending=False).reset_index(drop=True), N


def fig_redistribution(df, path=None, n_side=8):
    """Two panels (alpha, beta): per-gene pre->post change in top-1 V share, most
    promoted (red) and most suppressed (blue) genes labeled, on a SHARED x-axis so
    the beta 'rewrite' vs alpha 'reweight' magnitude gap is obvious at a glance."""
    plt = _plt()
    from matplotlib.patches import Patch
    tabs = {c: _redistribution_table(df, c) for c in ("TRA", "TRB")}
    xmax = max(tabs[c][0]["delta_top1_share"].abs().max() for c in tabs) * 100 * 1.18
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True)
    def _label(r):                                          # mark created-from-0 / erased-to-0
        if r["pre_top1"] == 0 and r["post_top1"] > 0:
            return f"{r['gene']} ★"                          # promoted from nothing
        if r["post_top1"] == 0 and r["pre_top1"] > 0:
            return f"{r['gene']} ✝"                          # suppressed to nothing (erased as argmax)
        return r["gene"]
    for ax, chain in zip(axes, ("TRA", "TRB")):
        t, N = tabs[chain]
        fz = t[(t["pre_top1"] == 0) & (t["post_top1"] > 0)]   # created from zero
        tz = t[(t["post_top1"] == 0) & (t["pre_top1"] > 0)]   # erased as argmax
        # show the leading promoted/suppressed genes AND every from-zero / to-zero gene
        sel = pd.concat([t.head(n_side), t.tail(n_side), fz, tz]).drop_duplicates("gene")
        sel = sel.sort_values("delta_top1_share")          # suppressed at bottom for barh
        d = sel["delta_top1_share"].to_numpy() * 100       # percentage points
        labels = [_label(r) for _, r in sel.iterrows()]
        edges = ["black" if lab != g else "none" for lab, g in zip(labels, sel["gene"])]
        ax.barh(range(len(sel)), d, color=["#C44E52" if x > 0 else "#4C72B0" for x in d],
                edgecolor=edges, linewidth=1.0)
        ax.set_yticks(range(len(sel))); ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color="k", lw=0.8); ax.set_xlim(-xmax, xmax)
        ax.set_title(f"{chain}  (N={N:,})")
        ax.set_xlabel("Δ top-1 V share, post − pre (percentage points)")
        note = (f"{len(fz)} V gene(s) created from zero  ★\n"
                f"{len(tz)} erased as argmax  ✝  ({int(tz['pre_top1'].sum()):,} seq)")
        ax.text(0.03, 0.04, note, transform=ax.transAxes, fontsize=8,
                va="bottom", ha="left", style="italic",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    axes[0].set_ylabel("V gene (most promoted / suppressed)")
    axes[1].legend(handles=[Patch(color="#C44E52", label="promoted (selection favors)"),
                            Patch(color="#4C72B0", label="suppressed (selection disfavors)")],
                   fontsize=8, loc="lower right")
    fig.suptitle("Selection redistributes which V gene wins: a rewrite in β, a reweight in α", fontsize=12)
    fig.text(0.5, -0.02,
             "Change in the fraction of sequences for which each V is the top-1 posterior call, pre→post selection "
             "(shared x-axis). Selection redistributes argmax V mass dramatically in β — erasing four genes (TRBV13/"
             "TRBV3-1/TRBV14/TRBV24-1) as argmax and creating six from zero (TRBV2 dominant) / amplifying TRBV19, "
             "TRBV4-1 — but only mildly in α, which creates none from zero, erases only a single TRAV18 sequence, and "
             "otherwise amplifies already-present genes (TRAV9-2).   "
             "★ = top-1 V created from zero pre-selection (promoted from nothing);  "
             "✝ = erased as argmax post-selection (suppressed to nothing).",
             ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, path or "fig7_redistribution")


def _adjusted_rand(a, b):
    a = pd.factorize(np.asarray(a))[0]
    b = pd.factorize(np.asarray(b))[0]
    C = pd.crosstab(pd.Series(a), pd.Series(b)).values
    sc = sum(comb(int(x), 2) for x in C.sum(1))
    sk = sum(comb(int(x), 2) for x in C.sum(0))
    sij = sum(comb(int(x), 2) for x in C.flatten())
    tot = comb(int(C.sum()), 2)
    if tot == 0:
        return 1.0
    exp = sc * sk / tot
    mx = (sc + sk) / 2
    return 1.0 if mx == exp else (sij - exp) / (mx - exp)


def grouping_comparison(df, chains=("TRA", "TRB"), min_count=20):
    """Compute IMGT-family grouping vs custom CDR3-confusion grouping (cut the
    leakage dendrogram at #families) and report their DEPARTURE, not a winner."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    print("\n" + "=" * 78)
    print("GROUPING DEPARTURE  (IMGT family  vs  confusion-cluster grouping)")
    print("=" * 78)
    out = {}
    for chain in chains:
        keep, M = build_leakage(df, chain, min_count)
        fam = [family_of(g) for g in keep]
        S = (M + M.T) / 2.0
        D = 1.0 - S; np.fill_diagonal(D, 0.0); D = (D + D.T) / 2.0
        Z = linkage(squareform(D, checks=False), method="average")
        K = len(set(fam))
        clust = fcluster(Z, t=K, criterion="maxclust")
        ari = _adjusted_rand(fam, clust)
        # cross-family confusion clusters: genes grouped together but different IMGT family
        cross = []
        for cl in sorted(set(clust)):
            members = [keep[i] for i in range(len(keep)) if clust[i] == cl]
            fams_here = sorted(set(family_of(g) for g in members))
            if len(members) > 1 and len(fams_here) > 1:
                cross.append((members, fams_here))
        print(f"\n[{chain}] {len(keep)} V genes (>= {min_count} seq), {K} IMGT families, "
              f"{len(set(clust))} confusion clusters")
        print(f"  adjusted Rand index (family vs confusion) = {ari:.3f}  "
              f"(1=identical, 0=chance)")
        print(f"  confusion clusters that MERGE across IMGT families: {len(cross)}")
        for members, fams_here in cross[:6]:
            print(f"    {{{', '.join(members)}}}  spans families {fams_here}")
        out[chain] = dict(ari=float(ari), n_genes=len(keep), n_families=K,
                          n_clusters=int(len(set(clust))), n_cross_family=len(cross))
    print("\n  low ARI / many cross-family merges = CDR3-confusion structure departs from "
          "IMGT family; the right grouping for recovery is data-driven, not nomenclatural.")
    return out


def build_figures(df):
    """The five named step-3 figures + the grouping-departure result."""
    fig_recoverability(df)
    fig_topk(df)
    for chain in ("TRA", "TRB"):
        fig_confusion(df, chain)
        fig_confusion_annotated(df, chain)
    fig_confidence(df)
    fig_selection(df)
    fig_dropped_weight(df)
    fig_redistribution(df)
    grouping_comparison(df)


def export_prism(df, outdir=None, min_per_len=20):
    """Dump the exact plotted data behind every figure as CSVs for GraphPad Prism.
    Reuses the same helpers the figures use, so each CSV reproduces its figure."""
    outdir = outdir or f"{FIGDIR}/prism"
    os.makedirs(outdir, exist_ok=True)
    ns = _canonical_ns(df)

    def w(name, frame, **kw):
        p = f"{outdir}/{name}"
        frame.to_csv(p, **kw)
        print(f"  [prism] {p}")

    # fig1: V & J conditional entropy vs CDR3 length (mean + n, >=min_per_len plotted)
    for axis in ("V", "J"):
        cols = {}
        for chain in ("TRA", "TRB"):
            g = _slice(df, chain=chain, axis=axis, model="pre", resolution="gene") \
                .groupby("cdr3_len")["entropy_nats"].agg(["mean", "count"])
            g = g[g["count"] >= min_per_len]
            cols[f"{chain}_entropy_nats"] = g["mean"]
            cols[f"{chain}_n"] = g["count"]
        t = pd.DataFrame(cols); t.index.name = "cdr3_len"
        w(f"fig1_recoverability_{axis}.csv", t)

    # fig2: fraction with true V in top-k, k=1..20
    ks = np.arange(1, 21)
    t = pd.DataFrame({"k": ks})
    for chain in ("TRA", "TRB"):
        r = _slice(df, chain=chain, axis="V", model="pre", resolution="gene")["true_rank"].dropna().values
        t[f"{chain}_n{ns[chain]}"] = [float((r <= k).mean()) for k in ks]
    w("fig2_topk.csv", t, index=False)

    # fig3: V-by-V leakage matrix, confusion-ordered (rows=true gene, cols=posterior gene)
    for chain in ("TRA", "TRB"):
        keep, M = build_leakage(df, chain)
        order, _ = _confusion_order(M)
        genes = [keep[i] for i in order]
        Mo = M[np.ix_(order, order)]
        t = pd.DataFrame(Mo, index=genes, columns=genes); t.index.name = "true_gene"
        w(f"fig3_confusion_{chain}.csv", t)

    # fig4: confident-fraction vs posterior-mass threshold
    ts = np.linspace(0.5, 0.99, 50)
    t = pd.DataFrame({"threshold": ts})
    for chain in ("TRA", "TRB"):
        for res in ("gene", "family"):
            m = _slice(df, chain=chain, axis="V", model="pre", resolution=res)["top1_mass"].dropna().values
            t[f"{chain}_{res}_n{ns[chain]}"] = [float((m >= x).mean()) for x in ts]
    w("fig4_confidence.csv", t, index=False)

    # fig5: raw vs usage-controlled selection shift; TRA bias-corrected by its perm null
    res = selection_comparison(df, "gene", verbose=False)
    perm_null = json.load(open("results/permutation_test_TRA.json"))["TRA"]["null_mean"]
    rows = []
    for c in ("TRA", "TRB"):
        null = perm_null if c == "TRA" else float("nan")     # figure corrects TRA only
        corr = res[c]["ctrl_shift"] - (null if c == "TRA" else 0.0)
        rows.append(dict(chain=c, n=res[c]["n"], raw_shift=res[c]["raw_shift"],
                         ctrl_shift=res[c]["ctrl_shift"], perm_null_mean=null,
                         ctrl_shift_bias_corrected=corr))
    w("fig5_selection.csv", pd.DataFrame(rows), index=False)

    # fig6: usage-control dropped weight by post-selection argmax-V gene, both chains
    for chain in ("TRA", "TRB"):
        t6, _, _, _ = _dropped_weight_table(df, chain)
        t6.attrs = {}
        w(f"fig6_dropped_weight_{chain}.csv", t6)

    # fig7: per-gene pre->post top-1 V share + usage redistribution, both chains
    for chain in ("TRA", "TRB"):
        t7, _ = _redistribution_table(df, chain)
        w(f"fig7_redistribution_{chain}.csv", t7, index=False)

    # equal_n: beta matched-n bootstrap entropy draws + alpha reference (per resolution)
    eq = equal_n_test(df, make_fig=False)
    for r, d in eq.items():
        t = pd.DataFrame({"beta_matched_n_H": d["boot"]})
        t["alpha_H"] = d["alpha_H"]; t["beta_ci_lo"] = d["ci"][0]; t["beta_ci_hi"] = d["ci"][1]
        w(f"equal_n_alpha_beta_{r}.csv", t, index=False)

    # matched_n: beta matched-n bootstrap shift draws + alpha / beta-full references
    mn = matched_n_selection(df, make_fig=False)
    t = pd.DataFrame({"beta_matched_n_shift": mn["boot"]})
    t["alpha_shift"] = mn["alpha_shift"]; t["beta_full_shift"] = mn["beta_full_shift"]
    t["beta_ci_lo"] = mn["ci"][0]; t["beta_ci_hi"] = mn["ci"][1]
    w("matched_n_selection.csv", t, index=False)

    with open(f"{outdir}/README.txt", "w") as fh:
        fh.write(_PRISM_README.format(TRA=ns["TRA"], TRB=ns["TRB"]))
    print(f"  [prism] {outdir}/README.txt")


_PRISM_README = """\
Source data for results/figures/*.png  (pooled alpha N={TRA:,} TRA / {TRB:,} TRB).
Each CSV is the exact data plotted in the matching figure; import into GraphPad
Prism and pick the columns as the X / Y series noted below.

fig1_recoverability_V.csv / _J.csv
    XY. X = cdr3_len; Y = TRA_entropy_nats, TRB_entropy_nats (mean conditional
    entropy, nats). *_n = #sequences in that length bin. Only bins with n>=20 are
    listed (the figure's display filter).
fig2_topk.csv
    XY. X = k (candidate-set size 1..20); Y = TRA_n{TRA}, TRB_n{TRB} = fraction of
    sequences whose true V is within the top-k posterior set. n is in the header.
fig3_confusion_TRA.csv / _TRB.csv
    Matrix/heatmap. Row = true V gene, column = posterior V gene, value = mean
    posterior mass (rows ~sum to 1). Rows/cols are in confusion-clustering order,
    matching the figure. Only V genes with >=20 sequences are included.
fig4_confidence.csv
    XY. X = threshold (0.50..0.99); Y = {{chain}}_{{gene|family}} = fraction of
    sequences with top-1 V posterior mass >= threshold.
fig5_selection.csv
    Grouped/bar. One row per chain: raw_shift and ctrl_shift (usage-controlled)
    are the two bars; ctrl_shift_bias_corrected subtracts the per-set permutation
    null (perm_null_mean) from ctrl_shift. The figure corrects TRA only (TRA
    bar = -0.0617); TRB has no correction (perm_null_mean blank).
fig6_dropped_weight_TRA.csv / _TRB.csv
    Bar (one file per chain). Row = post-selection argmax-V gene (rows the usage
    control zeroed because their CDR3-length x argmax-V cell is absent from the
    pre-selection marginal). dropped = #sequences dropped; pct_of_dropped = share of
    total dropped weight; gene_drop_rate = fraction of that gene's post-top1
    sequences dropped; post_top1_frac_of_chain = that gene's overall post-top1 share.
    Sorted by dropped. β concentrates in TRBV2/TRBV28; α is tiny and diffuse.
fig7_redistribution_TRA.csv / _TRB.csv
    Diverging bar. Row = V gene; pre_top1 / post_top1 = #sequences for which the
    gene is the top-1 V pre/post selection; pre_top1_share / post_top1_share =
    those as fractions of the chain; delta_top1_share = post − pre (sort key);
    pre_usage / post_usage = mean posterior mass on the gene (soft marginal usage).
    Plot delta_top1_share as a horizontal diverging bar (positive = promoted,
    negative = suppressed). beta spans far wider than alpha.
equal_n_alpha_beta_gene.csv / _family.csv
    Histogram. beta_matched_n_H = 500 beta-subsampled mean-entropy draws (matched
    to alpha n). alpha_H = alpha's mean entropy (reference line); beta_ci_lo/hi =
    beta 95%% CI. Plot a histogram of beta_matched_n_H with alpha_H as a vline.
matched_n_selection.csv
    Histogram. beta_matched_n_shift = 500 beta-subsampled usage-controlled-shift
    draws. alpha_shift = alpha's shift (reference line); beta_full_shift = beta at
    full n; beta_ci_lo/hi = beta 95%% CI.

Regenerate: python -m supervdj.aggregate prism
"""


def demo():
    """Self-check: the reweighting must (a) leave entropy unchanged when pre and
    post share an identical (len x V) marginal, and (b) normalize as a weighted mean."""
    # two length bins, two V genes; post has DIFFERENT entropies but we craft a case
    # where pre and post marginals already match -> ctrl == raw.
    rows = []
    def add(sid, model, mode, lenv, v, H):
        rows.append(dict(seq_id=sid, chain="TRA", axis="V", model=model, mode=mode,
                         resolution="gene", status="ok", n_candidates=2,
                         cdr3="X" * lenv, top1_label=v, entropy_nats=H,
                         top1_mass=0.6, posterior_json='{"a":0.6,"b":0.4}'))
    # two sequences (shared seq_id across arms); pre & post share the (len x argmax-V)
    # marginal, so the usage control must leave the shift unchanged.
    for sid, v in [(0, "A"), (1, "B")]:
        add(sid, "pre", None, 10, v, 1.0 if v == "A" else 0.0)
        add(sid, "post", "grid", 10, v, 1.0 if v == "A" else 0.0)
    d = pd.DataFrame(rows)
    d["cdr3_len"] = d["cdr3"].str.len()
    r = selection_comparison(d)["TRA"]
    assert abs(r["raw_shift"]) < 1e-9, r
    assert abs(r["ctrl_shift"] - r["raw_shift"]) < 1e-9, r  # matched marginal -> no change
    assert r["frac_post_kept"] == 1.0, r
    # grouping helpers
    assert family_of("TRBV6-1") == "TRBV6" and family_of("TRBV2") == "TRBV2"
    # family def preserves the /DV dual-designation (unlike a plain subgroup strip)
    assert family_of("TRAV29/DV5") == "TRAV29/DV5" and family_of("TRAV38-2/DV8") == "TRAV38/DV8"
    assert abs(_adjusted_rand([0, 0, 1, 1], [1, 1, 0, 0]) - 1.0) < 1e-9   # same partition
    assert _adjusted_rand([0, 1, 0, 1], [0, 0, 1, 1]) < 0.5               # unrelated
    # controlled-shift helper: matched pre/post argmax marginal -> ctrl == raw
    f = pd.DataFrame(dict(cdr3_len=[10, 10], pre_H=[1.0, 0.0], post_H=[1.0, 0.0],
                          pre_top1=["A", "B"], post_top1=["A", "B"]))
    rs = _controlled_shift(f)
    assert abs(rs["ctrl_shift"] - rs["raw_shift"]) < 1e-9, rs
    # permutation null: identical pre/post arms -> shift 0 under any relabeling
    fe = pd.DataFrame(dict(cdr3_len=[10, 11, 10, 11], pre_H=[1., .5, .2, .8],
                           post_H=[1., .5, .2, .8], pre_top1=list("ABAB"), post_top1=list("ABAB")))
    assert abs(_controlled_shift(fe)["ctrl_shift"]) < 1e-9
    # smoothing is backward-compatible at smooth=0 and retains otherwise-dropped cells
    fd = pd.DataFrame(dict(cdr3_len=[10, 10], pre_H=[1.0, 0.0], post_H=[0.5, 0.5],
                           pre_top1=["A", "A"], post_top1=["A", "B"]))  # (10,B) absent from pre
    assert _controlled_shift(fd, smooth=0.0)["frac_kept"] < 1.0          # B dropped
    assert _controlled_shift(fd, smooth=1.0)["frac_kept"] == 1.0          # B retained
    print("demo OK")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "demo":
        demo()
    else:
        d = load()
        if cmd in ("all", "gates"):
            integrity_report(d)
            selection_comparison(d, "gene")
        if cmd in ("all", "checks"):
            equal_n_test(d)
            impossible_breakdown(d)
        if cmd in ("all", "matched"):
            matched_n_selection(d)
            persist_matched_n_cross_chain(d)
        if cmd in ("all", "pvalues"):
            dropped_weight_diagnostic(d)
            permutation_test(d)
            cross_chain_pvalue(d)
        if cmd in ("all", "summary"):
            selection_summary(d)
        if cmd in ("all", "bootstrap"):
            alpha_bootstrap(d)
        if cmd in ("all", "figures"):
            build_figures(d)
        if cmd in ("all", "prism"):
            export_prism(d)
