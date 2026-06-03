"""Weight ablation benchmark for VDJdb V/J marginal recovery.

All arms use the same candidate set per row, the same softmax over joint
candidate scores, and the same posterior marginalization over V and J. Only the
weight vector differs across arms.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_vdjdb_recovery import strip_allele  # noqa: E402
from supervdj.filters import (  # noqa: E402
    chain_compatible,
    filter_j_genes,
    filter_v_genes,
    normalize_cdr3,
)
from supervdj.models import GeneReference, PredictionConfig  # noqa: E402
from supervdj.motif import boundary_penalty, j_motif_score, usage_prior_log, v_motif_score  # noqa: E402
from supervdj.olga_wrapper import compute_log_pgen  # noqa: E402
from supervdj.reference import load_reference, split_v_j  # noqa: E402
from supervdj.scoring import _stable_softmax  # noqa: E402
from supervdj.sonia_wrapper import compute_log_selection  # noqa: E402
from supervdj.vdjdb import VdjdbRow, load_vdjdb  # noqa: E402


DEFAULT_WEIGHTS = {
    "olga_weight": 0.1,
    "sonia_weight": 0.1,
    "motif_weight": 1.0,
    "usage_weight": 1.0,
    "boundary_weight": 1.0,
}
SUMMARY_KS = (1, 5, 10, 20)


@dataclass(frozen=True)
class Arm:
    name: str
    olga_weight: float
    sonia_weight: float
    motif_weight: float
    usage_weight: float
    boundary_weight: float
    fixed_j: bool = False


@dataclass(frozen=True)
class ComponentScore:
    v_gene: str
    j_gene: str
    log_pgen: float
    log_selection: float
    motif_score: float
    usage_prior: float
    boundary_penalty: float


_WORKER_CHAIN = ""
_WORKER_ARMS: list[Arm] = []
_WORKER_V_REFS: list[GeneReference] = []
_WORKER_J_REFS: list[GeneReference] = []


def build_arms() -> list[Arm]:
    arms = [
        Arm("usage_prior_only", 0.0, 0.0, 0.0, 1.0, 0.0),
        Arm("raw_pgen_only", 1.0, 0.0, 0.0, 0.0, 0.0),
        Arm("canonical_ppost", 1.0, 1.0, 0.0, 0.0, 0.0),
        Arm("canonical_ppost_fixedJ", 1.0, 1.0, 0.0, 0.0, 0.0, fixed_j=True),
        Arm("heuristic_only", 0.0, 0.0, 1.0, 1.0, 1.0),
        Arm("damped_composite", **DEFAULT_WEIGHTS),
    ]
    for olga_weight in (0.1, 0.3, 1.0, 3.0):
        for sonia_weight in (0.1, 0.3, 1.0, 3.0):
            arms.append(
                Arm(
                    f"sweep_olga{olga_weight:g}_sonia{sonia_weight:g}",
                    olga_weight,
                    sonia_weight,
                    1.0,
                    1.0,
                    1.0,
                )
            )
    return arms


def cdr3_disjoint_split(
    rows: Sequence[VdjdbRow],
    n_test: Optional[int],
    seed: int,
) -> tuple[list[VdjdbRow], list[VdjdbRow]]:
    grouped: Dict[str, list[VdjdbRow]] = {}
    for row in rows:
        grouped.setdefault(row.cdr3, []).append(row)
    cdr3s = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(cdr3s)
    target = len(rows) if n_test is None else min(n_test, len(rows))
    test_keys: set[str] = set()
    test_rows: list[VdjdbRow] = []
    for cdr3 in cdr3s:
        if len(test_rows) >= target:
            break
        test_keys.add(cdr3)
        test_rows.extend(grouped[cdr3])
    train_rows = [row for row in rows if row.cdr3 not in test_keys]
    return train_rows, test_rows[:target]


def fit_usage_priors(
    v_refs: Sequence[GeneReference],
    j_refs: Sequence[GeneReference],
    train_rows: Sequence[VdjdbRow],
) -> tuple[list[GeneReference], list[GeneReference]]:
    v_counts = Counter(strip_allele(row.v) for row in train_rows if row.v)
    j_counts = Counter(strip_allele(row.j) for row in train_rows if row.j)
    v_total = sum(v_counts.values())
    j_total = sum(j_counts.values())

    def update(ref: GeneReference, counts: Counter, total: int) -> GeneReference:
        if total <= 0:
            return ref
        return GeneReference(
            gene=ref.gene,
            chain=ref.chain,
            gene_type=ref.gene_type,
            functional=ref.functional,
            anchor=ref.anchor,
            usage_prior=counts.get(ref.gene, 0) / total,
        )

    return (
        [update(ref, v_counts, v_total) for ref in v_refs],
        [update(ref, j_counts, j_total) for ref in j_refs],
    )


def _candidate_components(
    cdr3: str,
    chain: str,
    v_refs: Sequence[GeneReference],
    j_refs: Sequence[GeneReference],
) -> list[ComponentScore]:
    cdr3 = normalize_cdr3(cdr3)
    base = PredictionConfig(
        chain=chain,
        use_olga=True,
        use_sonia=True,
        olga_weight=1.0,
        sonia_weight=1.0,
        motif_weight=1.0,
        usage_weight=1.0,
        boundary_weight=1.0,
    )
    v_survivors = filter_v_genes(cdr3, chain_compatible(v_refs, chain), base)
    j_survivors = filter_j_genes(cdr3, chain_compatible(j_refs, chain), base)
    if (not v_survivors or not j_survivors) and base.relax_filters_if_empty:
        v_survivors = filter_v_genes(cdr3, chain_compatible(v_refs, chain), base, relaxed=True)
        j_survivors = filter_j_genes(cdr3, chain_compatible(j_refs, chain), base, relaxed=True)
    out: list[ComponentScore] = []
    for v in v_survivors:
        for j in j_survivors:
            olga = compute_log_pgen(cdr3, v, j, chain)
            if olga.impossible:
                continue
            sonia = compute_log_selection(cdr3, v, j, chain)
            if sonia.impossible:
                continue
            v_score, _ = v_motif_score(cdr3, v, base)
            j_score, _ = j_motif_score(cdr3, j, base)
            out.append(
                ComponentScore(
                    v_gene=v.gene,
                    j_gene=j.gene,
                    log_pgen=olga.log_pgen,
                    log_selection=sonia.log_selection,
                    motif_score=v_score + j_score,
                    usage_prior=usage_prior_log(v, j, base),
                    boundary_penalty=boundary_penalty(cdr3, v, j, base),
                )
            )
    return out


def _score_with_arm(score: ComponentScore, arm: Arm) -> float:
    return (
        arm.olga_weight * score.log_pgen
        + arm.sonia_weight * score.log_selection
        + arm.motif_weight * score.motif_score
        + arm.usage_weight * score.usage_prior
        - arm.boundary_weight * score.boundary_penalty
    )


def _rank_gene(
    ranking: Iterable[tuple[str, float]],
    truth: str,
) -> Optional[int]:
    truth = strip_allele(truth)
    for rank, (gene, _) in enumerate(ranking, start=1):
        if strip_allele(gene) == truth:
            return rank
    return None


def _marginal_ranks(components: Sequence[ComponentScore], arm: Arm, row: VdjdbRow) -> dict[str, object]:
    if arm.fixed_j:
        truth_j = strip_allele(row.j)
        components = [
            score for score in components if strip_allele(score.j_gene) == truth_j
        ]
    if not components:
        return {
            "arm": arm.name,
            "chain": row.chain,
            "cdr3": row.cdr3,
            "true_v": strip_allele(row.v),
            "true_j": strip_allele(row.j),
            "j_conditioning": "observed_j" if arm.fixed_j else "marginalized",
            "v_rank": None,
            "j_rank": None,
        }
    values = [_score_with_arm(score, arm) for score in components]
    posts = _stable_softmax(values)
    v_marg: Dict[str, float] = {}
    j_marg: Dict[str, float] = {}
    for score, posterior in zip(components, posts):
        v_marg[score.v_gene] = v_marg.get(score.v_gene, 0.0) + posterior
        j_marg[score.j_gene] = j_marg.get(score.j_gene, 0.0) + posterior
    v_ranking = sorted(v_marg.items(), key=lambda kv: kv[1], reverse=True)
    j_ranking = sorted(j_marg.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "arm": arm.name,
        "chain": row.chain,
        "cdr3": row.cdr3,
        "true_v": strip_allele(row.v),
        "true_j": strip_allele(row.j),
        "j_conditioning": "observed_j" if arm.fixed_j else "marginalized",
        "v_rank": _rank_gene(v_ranking, row.v),
        "j_rank": _rank_gene(j_ranking, row.j),
    }


def evaluate_row(row: VdjdbRow, chain: str, arms: Sequence[Arm], v_refs, j_refs) -> list[dict[str, object]]:
    components = _candidate_components(row.cdr3, chain, v_refs, j_refs)
    return [_marginal_ranks(components, arm, row) for arm in arms]


def _init_worker(chain: str, arms: Sequence[Arm], v_refs, j_refs) -> None:
    global _WORKER_CHAIN, _WORKER_ARMS, _WORKER_V_REFS, _WORKER_J_REFS
    _WORKER_CHAIN = chain
    _WORKER_ARMS = list(arms)
    _WORKER_V_REFS = list(v_refs)
    _WORKER_J_REFS = list(j_refs)


def _worker(row: VdjdbRow) -> list[dict[str, object]]:
    return evaluate_row(row, _WORKER_CHAIN, _WORKER_ARMS, _WORKER_V_REFS, _WORKER_J_REFS)


def topk(ranks: Sequence[Optional[int]], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mrr(ranks: Sequence[Optional[int]]) -> float:
    if not ranks:
        return 0.0
    return sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks)


def summarize(per_row: pd.DataFrame, arms: Sequence[Arm], train_rows, test_rows) -> pd.DataFrame:
    train_cdr3 = {row.cdr3 for row in train_rows}
    test_cdr3 = {row.cdr3 for row in test_rows}
    overlap = len(train_cdr3 & test_cdr3)
    rows = []
    for arm in arms:
        sub = per_row[per_row["arm"] == arm.name]
        v_ranks = [None if pd.isna(x) else int(x) for x in sub["v_rank"]]
        j_ranks = [None if pd.isna(x) else int(x) for x in sub["j_rank"]]
        record = {
            "arm": arm.name,
            "chain": sub["chain"].iloc[0] if not sub.empty else "",
            "n_test": len(test_rows),
            "n_prior_train": len(train_rows),
            "cdr3_overlap_train_test": overlap,
            "olga_weight": arm.olga_weight,
            "sonia_weight": arm.sonia_weight,
            "motif_weight": arm.motif_weight,
            "usage_weight": arm.usage_weight,
            "boundary_weight": arm.boundary_weight,
            "j_conditioning": "observed_j" if arm.fixed_j else "marginalized",
            "v_mrr": mrr(v_ranks),
            "j_mrr": mrr(j_ranks),
        }
        for k in SUMMARY_KS:
            record[f"v_top{k}"] = topk(v_ranks, k)
            record[f"j_top{k}"] = topk(j_ranks, k)
        rows.append(record)
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vdjdb", type=Path, required=True)
    parser.add_argument("--chain", choices=["TRA", "TRB"], default="TRB")
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/weight_ablation.csv"))
    parser.add_argument(
        "--per-row-out",
        type=Path,
        default=Path("results/weight_ablation_per_row.tsv"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    chain = args.chain.upper()
    all_rows = load_vdjdb(args.vdjdb, chain=chain)
    train_rows, test_rows = cdr3_disjoint_split(all_rows, args.n_test, args.seed)
    train_cdr3 = {row.cdr3 for row in train_rows}
    test_cdr3 = {row.cdr3 for row in test_rows}
    cdr3_overlap = len(train_cdr3 & test_cdr3)
    v_refs, j_refs = split_v_j(load_reference(chain))
    v_refs, j_refs = fit_usage_priors(v_refs, j_refs, train_rows)
    arms = build_arms()

    rows: list[dict[str, object]] = []
    if args.workers <= 1:
        for row in tqdm(test_rows, desc=f"Ablating {chain}", unit="row"):
            rows.extend(evaluate_row(row, chain, arms, v_refs, j_refs))
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(chain, arms, v_refs, j_refs),
        ) as pool:
            for result in tqdm(
                pool.map(_worker, test_rows, chunksize=1),
                total=len(test_rows),
                desc=f"Ablating {chain}",
                unit="row",
            ):
                rows.extend(result)

    per_row = pd.DataFrame(rows)
    summary = summarize(per_row, arms, train_rows, test_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.per_row_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    per_row.to_csv(args.per_row_out, sep="\t", index=False)

    print(f"validation cdr3_overlap_train_test: {cdr3_overlap}")
    print(
        "validation gene_usage_prior_source: training_partition_only "
        f"(n_prior_train={len(train_rows)}, n_test={len(test_rows)})"
    )
    print(
        "validation canonical_ppost_j_handling: canonical_ppost marginalizes over J; "
        "canonical_ppost_fixedJ conditions on the observed J per sequence."
    )
    print(summary.to_string(index=False))
    for metric in ("v_top1", "v_top5", "v_top10", "v_mrr"):
        best = summary.sort_values(metric, ascending=False).iloc[0]
        print(f"best {metric}: {best['arm']} = {best[metric]:.6f}")
    damped_v_mrr = float(summary.loc[summary["arm"] == "damped_composite", "v_mrr"].iloc[0])
    canonical_v_mrr = float(summary.loc[summary["arm"] == "canonical_ppost", "v_mrr"].iloc[0])
    fixed_v_mrr = float(summary.loc[summary["arm"] == "canonical_ppost_fixedJ", "v_mrr"].iloc[0])
    print(
        "canonical_ppost beats damped_composite on V MRR: "
        f"{canonical_v_mrr > damped_v_mrr} "
        f"(canonical_ppost={canonical_v_mrr:.6f}, damped_composite={damped_v_mrr:.6f})"
    )
    print(
        "canonical_ppost_fixedJ beats damped_composite on V MRR: "
        f"{fixed_v_mrr > damped_v_mrr} "
        f"(canonical_ppost_fixedJ={fixed_v_mrr:.6f}, damped_composite={damped_v_mrr:.6f})"
    )
    print(f"wrote summary CSV to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
