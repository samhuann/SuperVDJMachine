"""VDJdb loader — empirical V/J usage summaries and evaluation harness.

``Best_VDJdb.tsv`` ships at the repo root. Columns of interest:

* ``Gene``  — ``TRA`` or ``TRB`` (per-row chain).
* ``CDR3``  — CDR3 amino-acid sequence.
* ``V``     — annotated V gene including allele, e.g., ``TRBV19*01``.
* ``J``     — annotated J gene including allele, e.g., ``TRBJ2-1*01``.
* ``Species`` — restricted to ``HomoSapiens`` here.

We use these labels for two things:

1. **Empirical V/J usage summaries** — frequency of each V and J gene per
   chain across labelled rows. These are descriptive only and are not fed
   back into prediction during validation.

2. **Evaluation** — a benchmark harness that runs :func:`supervdj.predict`
   on every labelled row and reports top-1 / top-5 / top-10 V, J, and
   joint V+J accuracy plus median rank and mean reciprocal rank.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

from supervdj._tables import read_table as _read_table
from supervdj._tables import resolve_column as _resolve_column
from supervdj._tables import strip_allele as _strip_allele
from supervdj.models import GeneReference, PredictionConfig
from supervdj.scoring import predict


@dataclass
class VdjdbRow:
    """Minimal projection of a VDJdb row used by this package."""

    chain: str
    cdr3: str
    v: str
    j: str
    species: str = ""
    epitope: str = ""


def load_vdjdb(
    path: Path,
    species: str = "HomoSapiens",
    chain: Optional[str] = None,
) -> List[VdjdbRow]:
    """Load and lightly filter a VDJdb-style TSV.

    Args:
        path: Path to ``Best_VDJdb.tsv``.
        species: Restrict to this ``Species`` value.
        chain: Optional ``TRA``/``TRB`` filter.

    Returns:
        List of :class:`VdjdbRow`. Rows missing CDR3, V, or J are dropped.
    """
    df = _read_table(path)
    chain_col = _resolve_column(df, "gene", "Gene")
    cdr3_col = _resolve_column(df, "cdr3", "CDR3")
    v_col = _resolve_column(df, "v", "V", "v.segm")
    j_col = _resolve_column(df, "j", "J", "j.segm")
    species_col = next((c for c in df.columns if c.lower() == "species"), None)
    epitope_col = next((c for c in df.columns if c.lower() == "epitope"), None)

    if species_col and species:
        df = df[df[species_col] == species]
    if chain is not None:
        df = df[df[chain_col].str.upper() == chain.upper()]
    df = df.dropna(subset=[cdr3_col, v_col, j_col])
    rows: List[VdjdbRow] = []
    for r in df.to_dict("records"):
        chain_val = str(r.get(chain_col, "")).upper()
        if chain_val not in {"TRA", "TRB"}:
            continue
        rows.append(
            VdjdbRow(
                chain=chain_val,
                cdr3=str(r[cdr3_col]).strip().upper(),
                v=_strip_allele(str(r[v_col])),
                j=_strip_allele(str(r[j_col])),
                species=str(r.get(species_col, "") or "") if species_col else "",
                epitope=str(r.get(epitope_col, "") or "") if epitope_col else "",
            )
        )
    return rows


def gene_usage_frequencies(
    rows: Iterable[VdjdbRow], chain: str
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return ``(v_frequency_map, j_frequency_map)`` normalised by type."""
    chain = chain.upper()
    v_counter: Counter = Counter()
    j_counter: Counter = Counter()
    for row in rows:
        if row.chain != chain:
            continue
        if row.v:
            v_counter[row.v] += 1
        if row.j:
            j_counter[row.j] += 1
    v_total = sum(v_counter.values()) or 1
    j_total = sum(j_counter.values()) or 1
    v_prior = {g: c / v_total for g, c in v_counter.items()}
    j_prior = {g: c / j_total for g, c in j_counter.items()}
    return v_prior, j_prior


# --- Evaluation harness -----------------------------------------------------


@dataclass
class EvalRowResult:
    """Per-row evaluation outcome."""

    cdr3: str
    chain: str
    true_v: str
    true_j: str
    v_rank: Optional[int]
    j_rank: Optional[int]
    pair_rank: Optional[int]
    top_v: str
    top_j: str


@dataclass
class EvalSummary:
    """Aggregate metrics over an evaluation run."""

    n: int
    chain: str
    v_top1: float
    v_top5: float
    v_top10: float
    j_top1: float
    j_top5: float
    j_top10: float
    pair_top1: float
    pair_top5: float
    pair_top10: float
    v_mrr: float
    j_mrr: float
    pair_mrr: float
    rows: List[EvalRowResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        return {
            "n": self.n,
            "chain": self.chain,
            "v_top1": self.v_top1,
            "v_top5": self.v_top5,
            "v_top10": self.v_top10,
            "j_top1": self.j_top1,
            "j_top5": self.j_top5,
            "j_top10": self.j_top10,
            "pair_top1": self.pair_top1,
            "pair_top5": self.pair_top5,
            "pair_top10": self.pair_top10,
            "v_mrr": self.v_mrr,
            "j_mrr": self.j_mrr,
            "pair_mrr": self.pair_mrr,
        }


def _topk(ranks: List[Optional[int]], k: int) -> float:
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks)


def _mrr(ranks: List[Optional[int]]) -> float:
    if not ranks:
        return 0.0
    return sum((1.0 / r) if r else 0.0 for r in ranks) / len(ranks)


def evaluate(
    rows: Sequence[VdjdbRow],
    chain: str,
    config: Optional[PredictionConfig] = None,
    v_refs: Optional[Sequence[GeneReference]] = None,
    j_refs: Optional[Sequence[GeneReference]] = None,
    top_k: int = 50,
    limit: Optional[int] = None,
    progress: bool = False,
) -> EvalSummary:
    """Run :func:`supervdj.predict` on labelled rows and aggregate metrics.

    Args:
        rows: VDJdb-style labelled rows. Only those matching ``chain`` are
            scored; others are silently skipped.
        chain: ``TRA`` or ``TRB``.
        config: Optional prediction config (defaults to top_k=50 to give
            ranks for less-confident V/J).
        v_refs, j_refs: Optional reference overrides.
        top_k: Max rank tracked per row.
        limit: Optional cap on number of rows scored (for fast iteration).

    Returns:
        :class:`EvalSummary`.
    """
    chain = chain.upper()
    cfg = config or PredictionConfig(chain=chain, top_k=top_k)
    cfg.top_k = max(cfg.top_k, top_k)

    eligible = [r for r in rows if r.chain == chain]
    if limit is not None:
        eligible = eligible[:limit]

    per_row: List[EvalRowResult] = []
    iterator = tqdm(
        eligible,
        disable=not progress,
        desc=f"Evaluating {chain}",
        unit="row",
    )
    for row in iterator:
        result = predict(
            row.cdr3, chain, config=cfg, v_refs=v_refs, j_refs=j_refs
        )
        if not result.candidates:
            per_row.append(
                EvalRowResult(
                    cdr3=row.cdr3, chain=chain,
                    true_v=row.v, true_j=row.j,
                    v_rank=None, j_rank=None, pair_rank=None,
                    top_v="", top_j="",
                )
            )
            continue
        v_rank: Optional[int] = None
        for rk, (gene, _) in enumerate(result.v_ranking, start=1):
            if gene == row.v:
                v_rank = rk
                break
        j_rank: Optional[int] = None
        for rk, (gene, _) in enumerate(result.j_ranking, start=1):
            if gene == row.j:
                j_rank = rk
                break
        pair_rank: Optional[int] = None
        for rk, c in enumerate(result.candidates, start=1):
            if (
                pair_rank is None
                and c.v_gene == row.v
                and c.j_gene == row.j
            ):
                pair_rank = rk
        per_row.append(
            EvalRowResult(
                cdr3=row.cdr3, chain=chain,
                true_v=row.v, true_j=row.j,
                v_rank=v_rank,
                j_rank=j_rank,
                pair_rank=pair_rank,
                top_v=result.v_ranking[0][0] if result.v_ranking else "",
                top_j=result.j_ranking[0][0] if result.j_ranking else "",
            )
        )

    v_ranks = [r.v_rank for r in per_row]
    j_ranks = [r.j_rank for r in per_row]
    p_ranks = [r.pair_rank for r in per_row]

    return EvalSummary(
        n=len(per_row),
        chain=chain,
        v_top1=_topk(v_ranks, 1),
        v_top5=_topk(v_ranks, 5),
        v_top10=_topk(v_ranks, 10),
        j_top1=_topk(j_ranks, 1),
        j_top5=_topk(j_ranks, 5),
        j_top10=_topk(j_ranks, 10),
        pair_top1=_topk(p_ranks, 1),
        pair_top5=_topk(p_ranks, 5),
        pair_top10=_topk(p_ranks, 10),
        v_mrr=_mrr(v_ranks),
        j_mrr=_mrr(j_ranks),
        pair_mrr=_mrr(p_ranks),
        rows=per_row,
    )
