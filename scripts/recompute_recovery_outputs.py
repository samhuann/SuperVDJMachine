"""Recompute recovery metrics from saved per-sequence candidate JSON.

This is useful after changing how marginal V/J ranks are derived, because the
expensive OLGA/SONIA candidate scores can be reused without rerunning the full
benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_vdjdb_recovery import (
    exact_gene_match,
    family_match,
    gene_family,
    strip_allele,
    write_outputs,
)


def _rank_from_marginal(
    ranking: Iterable[Tuple[str, float]],
    truth: str,
    matcher,
) -> Optional[int]:
    for rank, (gene, _) in enumerate(ranking, start=1):
        if matcher(gene, truth):
            return rank
    return None


def _rank_pair(candidates: Sequence[Dict[str, object]], true_v: str, true_j: str, matcher) -> Optional[int]:
    for rank, candidate in enumerate(candidates, start=1):
        if matcher(candidate.get("v_gene", ""), true_v) and matcher(candidate.get("j_gene", ""), true_j):
            return rank
    return None


def _marginal_ranking(
    candidates: Sequence[Dict[str, object]],
    gene_key: str,
) -> list[Tuple[str, float]]:
    scores: Dict[str, float] = defaultdict(float)
    for candidate in candidates:
        gene = strip_allele(candidate.get(gene_key, ""))
        if not gene:
            continue
        weight = candidate.get("posterior_probability", 0.0)
        try:
            scores[gene] += float(weight)
        except (TypeError, ValueError):
            continue
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def recompute_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, source in df.iterrows():
        candidates = json.loads(source.get("ranked_candidates_json", "[]") or "[]")
        true_v = strip_allele(source.get("true_v", ""))
        true_j = strip_allele(source.get("true_j", ""))
        v_ranking = _marginal_ranking(candidates, "v_gene")
        j_ranking = _marginal_ranking(candidates, "j_gene")
        top_v = v_ranking[0][0] if v_ranking else ""
        top_j = j_ranking[0][0] if j_ranking else ""
        top_pair = candidates[0] if candidates else {}

        record = source.to_dict()
        record.update(
            {
                "true_v": true_v,
                "true_j": true_j,
                "true_v_family": gene_family(true_v),
                "true_j_family": gene_family(true_j),
                "top_v": top_v,
                "top_j": top_j,
                "top_v_family": gene_family(top_v),
                "top_j_family": gene_family(top_j),
                "top_pair_v": strip_allele(top_pair.get("v_gene", "")),
                "top_pair_j": strip_allele(top_pair.get("j_gene", "")),
                "v_exact_rank": _rank_from_marginal(v_ranking, true_v, exact_gene_match),
                "j_exact_rank": _rank_from_marginal(j_ranking, true_j, exact_gene_match),
                "pair_exact_rank": _rank_pair(candidates, true_v, true_j, exact_gene_match),
                "v_family_rank": _rank_from_marginal(v_ranking, true_v, family_match),
                "j_family_rank": _rank_from_marginal(j_ranking, true_j, family_match),
                "pair_family_rank": _rank_pair(candidates, true_v, true_j, family_match),
                "v_marginal_exact_rank": _rank_from_marginal(v_ranking, true_v, exact_gene_match),
                "j_marginal_exact_rank": _rank_from_marginal(j_ranking, true_j, exact_gene_match),
                "v_marginal_family_rank": _rank_from_marginal(v_ranking, true_v, family_match),
                "j_marginal_family_rank": _rank_from_marginal(j_ranking, true_j, family_match),
            }
        )
        record.pop("v_model_used", None)
        rows.append(record)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute recovery outputs from saved candidates.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("results/vdjdb_recovery"))
    parser.add_argument("--top-k-max", type=int, default=50)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    frames = [pd.read_csv(path, sep="\t") for path in args.inputs]
    df = pd.concat(frames, ignore_index=True)
    rows = recompute_rows(df)
    chains = sorted({str(row["chain"]).upper() for row in rows})
    write_outputs(rows, chains=chains, outdir=args.outdir, top_k_max=args.top_k_max)
    print(f"wrote recomputed recovery outputs to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
