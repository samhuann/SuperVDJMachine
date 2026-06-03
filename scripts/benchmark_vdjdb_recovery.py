"""Benchmark VDJdb V/J recovery with exact and family-level metrics.

This script is intentionally analysis-only: it uses the packaged IMGT
references and does not estimate priors or train models from VDJdb.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm.auto import tqdm

from supervdj.models import PredictionConfig
from supervdj.reference import load_reference, split_v_j
from supervdj.scoring import predict
from supervdj.vdjdb import VdjdbRow, load_vdjdb

CATEGORIES = (
    "V exact",
    "J exact",
    "V+J exact",
    "V family",
    "J family",
    "V+J family",
)
SUMMARY_KS = (1, 3, 5, 10, 20, 50)

_WORKER_CHAIN = ""
_WORKER_TOP_K_MAX = 50
_WORKER_USE_OLGA = False
_WORKER_USE_SONIA = False
_WORKER_WEIGHTS = None
_WORKER_V_REFS = None
_WORKER_J_REFS = None


def strip_allele(gene: object) -> str:
    """Normalize a gene name and remove allele suffixes.

    Examples:
        ``TRBV7-2*01`` -> ``TRBV7-2``
        ``traj33*02`` -> ``TRAJ33``
    """
    if gene is None:
        return ""
    text = str(gene).strip().upper()
    if not text or text.lower() == "nan":
        return ""
    return text.split("*", 1)[0]


def gene_family(gene: object) -> str:
    """Return the V/J family used for ambiguity-aware matching.

    V examples:
        TRAV12-2 -> TRAV12
        TRBV5-1 -> TRBV5
        TRBV20-1 -> TRBV20

    J examples:
        TRBJ2-7 -> TRBJ2
        TRAJ33 -> TRAJ33
    """
    normalized = strip_allele(gene)
    if not normalized:
        return ""
    match = re.match(r"^(TR[AB][VJ]\d+)-\d+((?:/DV\d+)?)$", normalized)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return normalized


def exact_gene_match(predicted: object, truth: object) -> bool:
    """True when two genes match after allele stripping."""
    return bool(strip_allele(predicted)) and strip_allele(predicted) == strip_allele(truth)


def family_match(predicted: object, truth: object) -> bool:
    """True when two genes match after family collapsing."""
    return bool(gene_family(predicted)) and gene_family(predicted) == gene_family(truth)


def reciprocal_rank(rank: Optional[int]) -> float:
    return 0.0 if rank is None else 1.0 / float(rank)


def mrr(ranks: Sequence[Optional[int]]) -> float:
    """Mean reciprocal rank, treating missing ranks as zero."""
    if not ranks:
        return 0.0
    return sum(reciprocal_rank(rank) for rank in ranks) / len(ranks)


def topk_recovery(ranks: Sequence[Optional[int]], k: int) -> float:
    """Fraction of rows recovered at rank <= k."""
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def median_rank(ranks: Sequence[Optional[int]], missing_rank: int) -> float:
    """Median rank with missing targets counted as ``missing_rank``."""
    if not ranks:
        return 0.0
    return float(median([rank if rank is not None else missing_rank for rank in ranks]))


def _first_rank(
    ranking: Iterable[Tuple[str, float]],
    truth: str,
    matcher,
) -> Optional[int]:
    for rank, (gene, _) in enumerate(ranking, start=1):
        if matcher(gene, truth):
            return rank
    return None


def _first_pair_rank(candidates, true_v: str, true_j: str, matcher) -> Optional[int]:
    for rank, candidate in enumerate(candidates, start=1):
        if matcher(candidate.v_gene, true_v) and matcher(candidate.j_gene, true_j):
            return rank
    return None


def _first_candidate_gene_rank(candidates, truth: str, gene_type: str, matcher) -> Optional[int]:
    for rank, candidate in enumerate(candidates, start=1):
        gene = candidate.v_gene if gene_type == "V" else candidate.j_gene
        if matcher(gene, truth):
            return rank
    return None


def _candidate_payload(candidates) -> str:
    rows = [
        {
            "rank": rank,
            "v_gene": candidate.v_gene,
            "j_gene": candidate.j_gene,
            "posterior_probability": candidate.posterior_probability,
            "final_log_score": candidate.final_log_score,
            "motif_score": candidate.motif_score,
            "log_pgen": candidate.log_pgen,
            "log_selection": candidate.log_selection,
        }
        for rank, candidate in enumerate(candidates, start=1)
    ]
    return json.dumps(rows, separators=(",", ":"))


def load_eval_references(chain: str):
    """Load packaged references only; VDJdb rows are never used as priors."""
    refs = load_reference(chain)
    return split_v_j(refs)


def evaluate_row(
    row: VdjdbRow,
    chain: str,
    top_k_max: int,
    use_olga: bool,
    use_sonia: bool,
    weights: Dict[str, float],
    v_refs,
    j_refs,
) -> Dict[str, object]:
    cfg = PredictionConfig(
        chain=chain,
        top_k=top_k_max,
        use_olga=use_olga,
        use_sonia=use_sonia,
        olga_weight=weights["olga_weight"],
        sonia_weight=weights["sonia_weight"],
        motif_weight=weights["motif_weight"],
        usage_weight=weights["usage_weight"],
        boundary_weight=weights["boundary_weight"],
    )
    result = predict(
        row.cdr3,
        chain,
        config=cfg,
        v_refs=v_refs,
        j_refs=j_refs,
        progress=False,
    )

    true_v = strip_allele(row.v)
    true_j = strip_allele(row.j)
    top_v = result.v_ranking[0][0] if result.v_ranking else ""
    top_j = result.j_ranking[0][0] if result.j_ranking else ""
    top_pair = result.candidates[0] if result.candidates else None

    record: Dict[str, object] = {
        "chain": chain,
        "cdr3": row.cdr3,
        "true_v": true_v,
        "true_j": true_j,
        "true_v_family": gene_family(true_v),
        "true_j_family": gene_family(true_j),
        "top_v": strip_allele(top_v),
        "top_j": strip_allele(top_j),
        "top_v_family": gene_family(top_v),
        "top_j_family": gene_family(top_j),
        "top_pair_v": strip_allele(top_pair.v_gene) if top_pair else "",
        "top_pair_j": strip_allele(top_pair.j_gene) if top_pair else "",
        "v_exact_rank": _first_rank(result.v_ranking, true_v, exact_gene_match),
        "j_exact_rank": _first_rank(result.j_ranking, true_j, exact_gene_match),
        "pair_exact_rank": _first_pair_rank(
            result.candidates, true_v, true_j, exact_gene_match
        ),
        "v_family_rank": _first_rank(result.v_ranking, true_v, family_match),
        "j_family_rank": _first_rank(result.j_ranking, true_j, family_match),
        "pair_family_rank": _first_pair_rank(
            result.candidates, true_v, true_j, family_match
        ),
        "v_marginal_exact_rank": _first_rank(result.v_ranking, true_v, exact_gene_match),
        "j_marginal_exact_rank": _first_rank(result.j_ranking, true_j, exact_gene_match),
        "v_marginal_family_rank": _first_rank(result.v_ranking, true_v, family_match),
        "j_marginal_family_rank": _first_rank(result.j_ranking, true_j, family_match),
        "confidence": result.confidence,
        "relaxed": result.relaxed,
        "warnings": "; ".join(result.warnings),
        "ranked_candidates_json": _candidate_payload(result.candidates),
    }
    return record


def _select_rows(
    rows: List[VdjdbRow],
    n: Optional[int],
    seed: int,
    sample_mode: str,
) -> List[VdjdbRow]:
    if n is None or n >= len(rows):
        return rows
    if sample_mode == "first":
        return rows[:n]
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), n))
    return [rows[i] for i in indices]


def evaluate_chain(
    vdjdb_path: Path,
    chain: str,
    top_k_max: int,
    n: Optional[int],
    seed: int,
    sample_mode: str,
    workers: int,
    use_olga: bool,
    use_sonia: bool,
    weights: Dict[str, float],
) -> List[Dict[str, object]]:
    rows = load_vdjdb(vdjdb_path, chain=chain)
    rows = _select_rows(rows, n=n, seed=seed, sample_mode=sample_mode)

    worker_count = max(1, int(workers or 1))
    desc = f"Benchmarking {chain}"
    if worker_count == 1:
        v_refs, j_refs = load_eval_references(chain)
        return [
            evaluate_row(
                row,
                chain,
                top_k_max,
                use_olga,
                use_sonia,
                weights,
                v_refs,
                j_refs,
            )
            for row in tqdm(rows, desc=desc, unit="row")
        ]

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_worker,
        initargs=(chain, top_k_max, use_olga, use_sonia, weights),
    ) as pool:
        return list(
            tqdm(
                pool.map(_evaluate_row_in_worker, rows, chunksize=1),
                total=len(rows),
                desc=desc,
                unit="row",
            )
        )


def _init_worker(
    chain: str,
    top_k_max: int,
    use_olga: bool,
    use_sonia: bool,
    weights: Dict[str, float],
) -> None:
    global _WORKER_CHAIN
    global _WORKER_TOP_K_MAX
    global _WORKER_USE_OLGA
    global _WORKER_USE_SONIA
    global _WORKER_WEIGHTS
    global _WORKER_V_REFS
    global _WORKER_J_REFS

    _WORKER_CHAIN = chain
    _WORKER_TOP_K_MAX = top_k_max
    _WORKER_USE_OLGA = use_olga
    _WORKER_USE_SONIA = use_sonia
    _WORKER_WEIGHTS = weights
    _WORKER_V_REFS, _WORKER_J_REFS = load_eval_references(chain)


def _evaluate_row_in_worker(row: VdjdbRow) -> Dict[str, object]:
    return evaluate_row(
        row,
        _WORKER_CHAIN,
        _WORKER_TOP_K_MAX,
        _WORKER_USE_OLGA,
        _WORKER_USE_SONIA,
        _WORKER_WEIGHTS,
        _WORKER_V_REFS,
        _WORKER_J_REFS,
    )


def _rank_column(category: str) -> str:
    return {
        "V exact": "v_exact_rank",
        "J exact": "j_exact_rank",
        "V+J exact": "pair_exact_rank",
        "V family": "v_family_rank",
        "J family": "j_family_rank",
        "V+J family": "pair_family_rank",
    }[category]


def metric_rows(df: pd.DataFrame, chain: str, top_k_max: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    missing_rank = top_k_max + 1
    for category in CATEGORIES:
        ranks = [
            int(x) if not pd.isna(x) and str(x) != "" else None
            for x in df[_rank_column(category)].tolist()
        ]
        row: Dict[str, object] = {
            "chain": chain,
            "category": category,
            "n": len(ranks),
            "mrr": mrr(ranks),
            "median_rank": median_rank(ranks, missing_rank=missing_rank),
            "recovered_at_top_k_max": topk_recovery(ranks, top_k_max),
        }
        for k in SUMMARY_KS:
            if k <= top_k_max:
                row[f"top{k}"] = topk_recovery(ranks, k)
        rows.append(row)
    return rows


def curve_rows(df: pd.DataFrame, chain: str, top_k_max: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for category in CATEGORIES:
        ranks = [
            int(x) if not pd.isna(x) and str(x) != "" else None
            for x in df[_rank_column(category)].tolist()
        ]
        for k in range(1, top_k_max + 1):
            rows.append(
                {
                    "chain": chain,
                    "category": category,
                    "k": k,
                    "fraction_recovered": topk_recovery(ranks, k),
                    "n": len(ranks),
                }
            )
    return rows


def missed_rows(df: pd.DataFrame, top_k_max: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        for category in CATEGORIES:
            rank = row[_rank_column(category)]
            missing = pd.isna(rank) or rank == "" or int(rank) > top_k_max
            if missing:
                rows.append(
                    {
                        "chain": row["chain"],
                        "cdr3": row["cdr3"],
                        "category": category,
                        "true_v": row["true_v"],
                        "true_j": row["true_j"],
                        "true_v_family": row["true_v_family"],
                        "true_j_family": row["true_j_family"],
                        "top_v": row["top_v"],
                        "top_j": row["top_j"],
                        "top_v_family": row["top_v_family"],
                        "top_j_family": row["top_j_family"],
                    }
                )
    return rows


def confusion_table(df: pd.DataFrame, chain: str, family: bool) -> pd.DataFrame:
    truth_col = "true_v_family" if family else "true_v"
    pred_col = "top_v_family" if family else "top_v"
    work = df[(df["chain"] == chain) & (df[truth_col] != df[pred_col])].copy()
    if work.empty:
        return pd.DataFrame(columns=["chain", "truth", "predicted", "n", "fraction"])
    grouped = (
        work.groupby([truth_col, pred_col], dropna=False)
        .size()
        .reset_index(name="n")
        .rename(columns={truth_col: "truth", pred_col: "predicted"})
        .sort_values("n", ascending=False)
    )
    grouped.insert(0, "chain", chain)
    grouped["fraction"] = grouped["n"] / max(1, len(df[df["chain"] == chain]))
    return grouped


def write_outputs(
    all_rows: List[Dict[str, object]],
    chains: Sequence[str],
    outdir: Path,
    top_k_max: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    per_df = pd.DataFrame(all_rows)
    per_df.to_csv(outdir / "per_sequence_results.tsv", sep="\t", index=False)

    by_chain_rows: List[Dict[str, object]] = []
    summary_parts: List[pd.DataFrame] = []
    for chain in chains:
        chain_df = per_df[per_df["chain"] == chain]
        by_chain_rows.extend(metric_rows(chain_df, chain, top_k_max))
        curve_df = pd.DataFrame(curve_rows(chain_df, chain, top_k_max))
        curve_df.to_csv(outdir / f"topk_curve_{chain}.tsv", sep="\t", index=False)
        confusion_table(chain_df, chain, family=False).to_csv(
            outdir / f"confusion_V_exact_{chain}.tsv", sep="\t", index=False
        )
        confusion_table(chain_df, chain, family=True).to_csv(
            outdir / f"confusion_V_family_{chain}.tsv", sep="\t", index=False
        )
        summary_parts.append(chain_df)

    by_chain_df = pd.DataFrame(by_chain_rows)
    by_chain_df.to_csv(outdir / "recovery_by_chain.tsv", sep="\t", index=False)

    summary_rows = metric_rows(pd.concat(summary_parts, ignore_index=True), "ALL", top_k_max)
    pd.DataFrame(summary_rows).to_csv(
        outdir / "recovery_summary.tsv", sep="\t", index=False
    )
    pd.DataFrame(missed_rows(per_df, top_k_max)).to_csv(
        outdir / "missed_candidates.tsv", sep="\t", index=False
    )


def _parse_on_off(value: str) -> bool:
    value_norm = value.strip().lower()
    if value_norm == "on":
        return True
    if value_norm == "off":
        return False
    raise argparse.ArgumentTypeError("expected 'on' or 'off'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark exact and family-level V/J recovery on VDJdb."
    )
    parser.add_argument("--vdjdb", type=Path, required=True)
    parser.add_argument("--chains", nargs="+", default=["TRA", "TRB"])
    parser.add_argument("--top-k-max", type=int, default=50)
    parser.add_argument("--n", type=int, default=None, help="Optional rows per chain")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sample-mode",
        choices=["first", "random"],
        default="first",
        help="How --n selects rows within each chain.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/vdjdb_recovery"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--olga", type=_parse_on_off, default=False, metavar="on/off")
    parser.add_argument("--sonia", type=_parse_on_off, default=False, metavar="on/off")
    parser.add_argument("--olga-weight", type=float, default=0.1)
    parser.add_argument("--sonia-weight", type=float, default=0.1)
    parser.add_argument("--motif-weight", type=float, default=1.0)
    parser.add_argument("--usage-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    chains = [chain.upper() for chain in args.chains]
    weights = {
        "olga_weight": args.olga_weight,
        "sonia_weight": args.sonia_weight,
        "motif_weight": args.motif_weight,
        "usage_weight": args.usage_weight,
        "boundary_weight": args.boundary_weight,
    }
    all_rows: List[Dict[str, object]] = []
    for i, chain in enumerate(chains):
        all_rows.extend(
            evaluate_chain(
                vdjdb_path=args.vdjdb,
                chain=chain,
                top_k_max=args.top_k_max,
                n=args.n,
                seed=args.seed + i,
                sample_mode=args.sample_mode,
                workers=args.workers,
                use_olga=args.olga,
                use_sonia=args.sonia,
                weights=weights,
            )
        )
    write_outputs(all_rows, chains=chains, outdir=args.outdir, top_k_max=args.top_k_max)
    print(f"wrote recovery benchmark outputs to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
