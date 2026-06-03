"""Benchmark whether a supervised V model improves OLGA/SONIA recovery.

The benchmark uses VDJdb only as held-out evaluation rows. It compares:

* ``olga_sonia``: candidate posteriors from OLGA + SONIA + motif/usage terms.
* ``v_model``: the supervised V model by itself, for V-gene recovery only.
* ``olga_sonia_plus_v_model_wX``: full V/J candidate scores with
  ``X * log P_model(V | CDR3)`` added to each pair before posterior
  marginalization.
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_vdjdb_recovery import (
    exact_gene_match,
    family_match,
    gene_family,
    load_eval_references,
    strip_allele,
    topk_recovery,
)
from supervdj.filters import (
    chain_compatible,
    filter_j_genes,
    filter_v_genes,
    normalize_cdr3,
)
from supervdj.models import CandidateScore, PredictionConfig
from supervdj.scoring import _stable_softmax, score_candidates
from supervdj.vdjdb import VdjdbRow, load_vdjdb
from supervdj.v_multilabel import LinearVGeneModel


SUMMARY_KS = (1, 3, 5, 10, 20, 50)
CATEGORIES = ("V exact", "V family", "J exact", "J family", "V+J exact", "V+J family")

_WORKER_CHAIN = ""
_WORKER_WEIGHTS: list[float] = []
_WORKER_TOP_K_MAX = 50
_WORKER_V_REFS = None
_WORKER_J_REFS = None
_WORKER_V_MODEL = None


def _select_rows(rows: list[VdjdbRow], n: Optional[int]) -> list[VdjdbRow]:
    if n is None or n >= len(rows):
        return rows
    return rows[:n]


def _attach_posteriors(scores: Sequence[CandidateScore], values: Sequence[float]) -> None:
    for score, posterior in zip(scores, _stable_softmax(values)):
        score.posterior_probability = posterior


def _marginalize(scores: Sequence[CandidateScore]) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    v_probs: Dict[str, float] = {}
    j_probs: Dict[str, float] = {}
    for score in scores:
        v_probs[score.v_gene] = v_probs.get(score.v_gene, 0.0) + score.posterior_probability
        j_probs[score.j_gene] = j_probs.get(score.j_gene, 0.0) + score.posterior_probability
    return (
        sorted(v_probs.items(), key=lambda kv: kv[1], reverse=True),
        sorted(j_probs.items(), key=lambda kv: kv[1], reverse=True),
    )


def _first_rank(ranking: Iterable[Tuple[str, float]], truth: str, matcher) -> Optional[int]:
    for rank, (gene, _) in enumerate(ranking, start=1):
        if matcher(gene, truth):
            return rank
    return None


def _pair_rank(scores: Sequence[CandidateScore], true_v: str, true_j: str, matcher) -> Optional[int]:
    for rank, score in enumerate(scores, start=1):
        if matcher(score.v_gene, true_v) and matcher(score.j_gene, true_j):
            return rank
    return None


def _median_rank(ranks: Sequence[Optional[int]], missing_rank: int) -> float:
    if not ranks:
        return 0.0
    return float(median([rank if rank is not None else missing_rank for rank in ranks]))


def _mrr(ranks: Sequence[Optional[int]]) -> float:
    if not ranks:
        return 0.0
    return sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks)


def _rank_columns(category: str) -> str:
    return {
        "V exact": "v_exact_rank",
        "V family": "v_family_rank",
        "J exact": "j_exact_rank",
        "J family": "j_family_rank",
        "V+J exact": "pair_exact_rank",
        "V+J family": "pair_family_rank",
    }[category]


def _summary_rows(df: pd.DataFrame, top_k_max: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    missing_rank = top_k_max + 1
    for (chain, arm), arm_df in df.groupby(["chain", "arm"]):
        for category in CATEGORIES:
            ranks = [
                int(x) if not pd.isna(x) and str(x) != "" else None
                for x in arm_df[_rank_columns(category)].tolist()
            ]
            row: dict[str, object] = {
                "chain": chain,
                "arm": arm,
                "category": category,
                "n": len(ranks),
                "mrr": _mrr(ranks),
                "median_rank": _median_rank(ranks, missing_rank),
                "recovered_at_top_k_max": topk_recovery(ranks, top_k_max),
            }
            for k in SUMMARY_KS:
                if k <= top_k_max:
                    row[f"top{k}"] = topk_recovery(ranks, k)
            rows.append(row)
    return rows


def _curve_rows(df: pd.DataFrame, top_k_max: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (chain, arm), arm_df in df.groupby(["chain", "arm"]):
        for category in CATEGORIES:
            ranks = [
                int(x) if not pd.isna(x) and str(x) != "" else None
                for x in arm_df[_rank_columns(category)].tolist()
            ]
            for k in range(1, top_k_max + 1):
                rows.append(
                    {
                        "chain": chain,
                        "arm": arm,
                        "category": category,
                        "k": k,
                        "fraction_recovered": topk_recovery(ranks, k),
                        "n": len(ranks),
                    }
                )
    return rows


def _candidate_scores(cdr3: str, chain: str, v_refs, j_refs):
    cfg = PredictionConfig(chain=chain, use_olga=True, use_sonia=True)
    cdr3 = normalize_cdr3(cdr3)
    v_refs = chain_compatible(v_refs, chain)
    j_refs = chain_compatible(j_refs, chain)
    v_survivors = filter_v_genes(cdr3, v_refs, cfg)
    j_survivors = filter_j_genes(cdr3, j_refs, cfg)
    if (not v_survivors or not j_survivors) and cfg.relax_filters_if_empty:
        v_survivors = filter_v_genes(cdr3, v_refs, cfg, relaxed=True)
        j_survivors = filter_j_genes(cdr3, j_refs, cfg, relaxed=True)
    if not v_survivors or not j_survivors:
        return []
    scores, _, _, _, _ = score_candidates(cdr3, chain, v_survivors, j_survivors, cfg, progress=False)
    return scores


def _record_for_arm(
    row: VdjdbRow,
    arm: str,
    v_ranking: Sequence[Tuple[str, float]],
    j_ranking: Sequence[Tuple[str, float]],
    pair_ranking: Sequence[CandidateScore],
) -> dict[str, object]:
    true_v = strip_allele(row.v)
    true_j = strip_allele(row.j)
    return {
        "chain": row.chain,
        "cdr3": row.cdr3,
        "arm": arm,
        "true_v": true_v,
        "true_j": true_j,
        "true_v_family": gene_family(true_v),
        "true_j_family": gene_family(true_j),
        "top_v": strip_allele(v_ranking[0][0]) if v_ranking else "",
        "top_j": strip_allele(j_ranking[0][0]) if j_ranking else "",
        "top_v_family": gene_family(v_ranking[0][0]) if v_ranking else "",
        "top_j_family": gene_family(j_ranking[0][0]) if j_ranking else "",
        "v_exact_rank": _first_rank(v_ranking, true_v, exact_gene_match),
        "v_family_rank": _first_rank(v_ranking, true_v, family_match),
        "j_exact_rank": _first_rank(j_ranking, true_j, exact_gene_match),
        "j_family_rank": _first_rank(j_ranking, true_j, family_match),
        "pair_exact_rank": _pair_rank(pair_ranking, true_v, true_j, exact_gene_match),
        "pair_family_rank": _pair_rank(pair_ranking, true_v, true_j, family_match),
    }


def evaluate_row(
    row: VdjdbRow,
    chain: str,
    weights: Sequence[float],
    v_refs,
    j_refs,
    v_model,
) -> list[dict[str, object]]:
    scores = _candidate_scores(row.cdr3, chain, v_refs, j_refs)
    if not scores:
        empty: list[tuple[str, float]] = []
        return [_record_for_arm(row, "olga_sonia", empty, empty, [])]

    base_values = [score.final_log_score for score in scores]
    base_scores = list(scores)
    _attach_posteriors(base_scores, base_values)
    base_scores.sort(key=lambda s: s.posterior_probability, reverse=True)
    base_v, base_j = _marginalize(base_scores)
    records = [_record_for_arm(row, "olga_sonia", base_v, base_j, base_scores)]

    if v_model is None:
        return records

    model_genes = v_model.expand_genes(row.cdr3, [v.gene for v in v_refs])
    model_ranking = v_model.rank(row.cdr3, model_genes)
    records.append(_record_for_arm(row, "v_model", model_ranking, [], []))
    model_scores = v_model.log_scores(row.cdr3, model_genes)
    for weight in weights:
        fused_values = [
            score.final_log_score + float(weight) * model_scores.get(score.v_gene, math.log(1e-12))
            for score in scores
        ]
        fused_scores = list(scores)
        _attach_posteriors(fused_scores, fused_values)
        fused_scores.sort(key=lambda s: s.posterior_probability, reverse=True)
        fused_v, fused_j = _marginalize(fused_scores)
        arm = f"olga_sonia_plus_v_model_w{weight:g}"
        records.append(_record_for_arm(row, arm, fused_v, fused_j, fused_scores))
    return records


def _load_linear_model(model_dir: Optional[Path]):
    if model_dir is None:
        return None
    return LinearVGeneModel.load(model_dir)


def _init_worker(chain: str, weights: Sequence[float], top_k_max: int, model_dir: Optional[Path]) -> None:
    global _WORKER_CHAIN
    global _WORKER_WEIGHTS
    global _WORKER_TOP_K_MAX
    global _WORKER_V_REFS
    global _WORKER_J_REFS
    global _WORKER_V_MODEL
    _WORKER_CHAIN = chain
    _WORKER_WEIGHTS = list(weights)
    _WORKER_TOP_K_MAX = top_k_max
    _WORKER_V_REFS, _WORKER_J_REFS = load_eval_references(chain)
    _WORKER_V_MODEL = _load_linear_model(model_dir)


def _worker(row: VdjdbRow) -> list[dict[str, object]]:
    return evaluate_row(
        row,
        _WORKER_CHAIN,
        _WORKER_WEIGHTS,
        _WORKER_V_REFS,
        _WORKER_J_REFS,
        _WORKER_V_MODEL,
    )


def evaluate_chain(
    vdjdb: Path,
    chain: str,
    n: Optional[int],
    workers: int,
    weights: Sequence[float],
    top_k_max: int,
    model_dir: Optional[Path],
) -> list[dict[str, object]]:
    rows = _select_rows(load_vdjdb(vdjdb, chain=chain), n=n)
    if workers <= 1:
        v_refs, j_refs = load_eval_references(chain)
        v_model = _load_linear_model(model_dir)
        out: list[dict[str, object]] = []
        for row in tqdm(rows, desc=f"Benchmarking {chain}", unit="row"):
            out.extend(evaluate_row(row, chain, weights, v_refs, j_refs, v_model))
        return out

    out = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(chain, weights, top_k_max, model_dir),
    ) as pool:
        for records in tqdm(
            pool.map(_worker, rows, chunksize=1),
            total=len(rows),
            desc=f"Benchmarking {chain}",
            unit="row",
        ):
            out.extend(records)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare OLGA/SONIA alone against OLGA/SONIA + V model.")
    parser.add_argument("--vdjdb", type=Path, required=True)
    parser.add_argument("--chain", choices=["TRA", "TRB"], default="TRB")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--top-k-max", type=int, default=50)
    parser.add_argument("--weights", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--linear-v-model-dir",
        type=Path,
        default=None,
        help="Optional LinearVGeneModel artifact directory.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/v_model_ablation"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows = evaluate_chain(
        vdjdb=args.vdjdb,
        chain=args.chain,
        n=args.n,
        workers=max(1, args.workers),
        weights=args.weights,
        top_k_max=args.top_k_max,
        model_dir=args.linear_v_model_dir,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    per_df = pd.DataFrame(rows)
    per_df.to_csv(args.outdir / "per_sequence_ablation.tsv", sep="\t", index=False)
    pd.DataFrame(_summary_rows(per_df, args.top_k_max)).to_csv(
        args.outdir / "ablation_summary.tsv", sep="\t", index=False
    )
    pd.DataFrame(_curve_rows(per_df, args.top_k_max)).to_csv(
        args.outdir / f"topk_curve_{args.chain}.tsv", sep="\t", index=False
    )
    print(f"wrote V-model ablation outputs to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
