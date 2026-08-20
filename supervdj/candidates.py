"""Calibrated V-gene candidate sets for a CDR3.

The measurement this repo is built around says a single V-gene call is usually not
supportable: the recoverable unit is a *group* of mutually confusable genes. This
module is the tool that follows from that. Given a CDR3 it returns the smallest
candidate set that carries a requested amount of posterior mass, the confusion
groups those genes belong to, and -- the part that makes it calibrated -- the
coverage that set size actually achieved on the canonical data, measured rather
than assumed.

    from supervdj.candidates import calibrated_candidates
    r = calibrated_candidates("CASSLGQAYEQYF", "TRB", coverage=0.9)
    r["genes"]           # [('TRBV5-1', 0.19), ('TRBV7-9', 0.11), ...]
    r["achieved_mass"]   # 0.91
    r["groups"]          # [{'group': 3, 'mass': 0.42, 'genes': [...]}, ...]

Command line, one sequence or a batch::

    python -m supervdj.candidates --chain TRB --cdr3 CASSLGQAYEQYF --coverage 0.9
    python -m supervdj.candidates --chain TRA --input cdr3s.txt --out sets.tsv

Two ways to ask for a set:

``coverage=p``  (default 0.9)
    Size the set so the annotated gene is inside about ``p`` of the time. The
    nominal posterior mass needed for that is read from the shipped calibration
    curve (``supervdj/data/calibration_{chain}.tsv``), measured on 37,687 alpha and
    80,409 beta canonical rearrangements. This is the honest option: for the alpha
    chain the model posterior is over-confident, so 90% coverage needs ~0.978
    nominal mass, not 0.90.

``mass=p``
    Take the posterior at face value and stop once the cumulative mass reaches
    ``p``. The reported ``expected_coverage`` then says what that nominal mass is
    actually worth.

The confusion groups (``supervdj/data/confusion_groups_{chain}.tsv``) come from
cutting the V-by-V posterior-leakage dendrogram at the number of IMGT families --
the same grouping the paper's confusion analysis uses. Genes seen in fewer than 20
canonical sequences have no group and are reported as ``group: null``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DATA_DIR = Path(__file__).resolve().parent / "data"
CHAINS = ("TRA", "TRB")


@lru_cache(maxsize=None)
def _groups(chain: str) -> Dict[str, int]:
    """gene -> confusion group id."""
    path = DATA_DIR / f"confusion_groups_{chain}.tsv"
    with open(path) as fh:
        return {r["gene"]: int(r["group"]) for r in csv.DictReader(fh, delimiter="\t")}


@lru_cache(maxsize=None)
def _calibration(chain: str) -> List[Tuple[float, float, float]]:
    """[(nominal_mass, empirical_coverage, mean_set_size)] ascending by mass."""
    path = DATA_DIR / f"calibration_{chain}.tsv"
    with open(path) as fh:
        rows = [(float(r["nominal_mass"]), float(r["empirical_coverage"]),
                 float(r["mean_set_size"])) for r in csv.DictReader(fh, delimiter="\t")]
    return sorted(rows)


def _interp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Linear interpolation, clamped at both ends (no numpy dependency here)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), list(zip(xs, ys))[1:]):
        if x0 <= x <= x1:
            t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return ys[-1]


def mass_for_coverage(chain: str, coverage: float) -> Tuple[float, bool]:
    """Nominal posterior mass whose measured coverage is ``coverage``.

    Returns ``(mass, reachable)``; ``reachable`` is False when the requested
    coverage lies past the end of the measured curve, in which case the largest
    calibrated mass is returned instead.
    """
    cal = _calibration(chain)
    masses = [m for m, _, _ in cal]
    covs = [c for _, c, _ in cal]
    if coverage > covs[-1]:
        return masses[-1], False
    return _interp(coverage, covs, masses), True


def coverage_for_mass(chain: str, mass: float) -> float:
    """Measured coverage of a set built to carry ``mass`` nominal posterior mass."""
    cal = _calibration(chain)
    return _interp(mass, [m for m, _, _ in cal], [c for _, c, _ in cal])


def candidate_set(posterior: Dict[str, float], target_mass: float) -> Tuple[List[Tuple[str, float]], float]:
    """Smallest set of genes whose cumulative posterior reaches ``target_mass``."""
    ranked = sorted(posterior.items(), key=lambda kv: -kv[1])
    chosen: List[Tuple[str, float]] = []
    cum = 0.0
    for gene, mass in ranked:
        chosen.append((gene, mass))
        cum += mass
        if cum >= target_mass:
            break
    return chosen, cum


def group_breakdown(chain: str, genes: Sequence[Tuple[str, float]]) -> List[dict]:
    """Collapse a candidate set into its confusion groups, heaviest first."""
    gmap = _groups(chain)
    acc: Dict[Optional[int], dict] = {}
    for gene, mass in genes:
        gid = gmap.get(gene)
        slot = acc.setdefault(gid, {"group": gid, "mass": 0.0, "genes": []})
        slot["mass"] += mass
        slot["genes"].append(gene)
    out = sorted(acc.values(), key=lambda d: -d["mass"])
    for d in out:
        d["mass"] = round(d["mass"], 6)
    return out


def calibrated_candidates(cdr3: str, chain: str, coverage: Optional[float] = 0.9,
                          mass: Optional[float] = None, models=None, cache=None) -> dict:
    """Calibrated V candidate set for one CDR3.

    Give either ``coverage`` (size the set by measured coverage, the default) or
    ``mass`` (size it by nominal posterior mass). ``models``/``cache`` can be passed
    in to avoid reloading OLGA for every sequence in a batch.
    """
    from supervdj.cache import ValueCache
    from supervdj.models import load_chain_models
    from supervdj.posterior import preselection_posterior, summarize

    chain = chain.upper()
    if chain not in CHAINS:
        raise ValueError(f"chain must be one of {CHAINS}, got {chain!r}")
    models = models or load_chain_models(chain, use_sonia=False)
    cache = cache if cache is not None else ValueCache(None)   # in-memory only

    reachable = True
    if mass is None:
        if coverage is None:
            raise ValueError("give either coverage= or mass=")
        target, reachable = mass_for_coverage(chain, coverage)
    else:
        target = mass

    posterior = preselection_posterior(models, cache, cdr3, "V")
    if not posterior or sum(posterior.values()) == 0:
        return {"cdr3": cdr3, "chain": chain, "status": "impossible_cdr3",
                "genes": [], "achieved_mass": 0.0, "groups": []}

    genes, achieved = candidate_set(posterior, target)
    stats = summarize(posterior, "")
    result = {
        "cdr3": cdr3,
        "chain": chain,
        "status": "ok",
        "requested": {"coverage": coverage, "mass": mass},
        "nominal_mass": round(target, 6),
        "achieved_mass": round(achieved, 6),
        "expected_coverage": round(coverage_for_mass(chain, target), 4),
        "n_candidates": len(genes),
        "entropy_nats": round(stats.entropy_nats, 4),
        "genes": [(g, round(m, 6)) for g, m in genes],
        "groups": group_breakdown(chain, genes),
    }
    if not reachable:
        result["warning"] = (f"coverage {coverage} is past the measured curve for {chain}; "
                             f"used its largest calibrated mass {target:g} "
                             f"(coverage {result['expected_coverage']})")
    return result


def _read_inputs(path: Path) -> List[str]:
    """One CDR3 per line, or a TSV/CSV with a cdr3 column."""
    text = path.read_text().splitlines()
    if not text:
        return []
    head = text[0].lower()
    if "cdr3" in head:
        sep = "\t" if "\t" in text[0] else ","
        cols = [c.strip().lower() for c in text[0].split(sep)]
        i = cols.index("cdr3") if "cdr3" in cols else cols.index("cdr3_aa")
        return [ln.split(sep)[i].strip() for ln in text[1:] if ln.strip()]
    return [ln.strip() for ln in text if ln.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", required=True, choices=list(CHAINS))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cdr3", help="a single CDR3 amino-acid sequence")
    src.add_argument("--input", type=Path, help="file of CDR3s (one per line, or a cdr3 column)")
    size = ap.add_mutually_exclusive_group()
    size.add_argument("--coverage", type=float, default=0.9,
                      help="target measured coverage (default 0.9)")
    size.add_argument("--mass", type=float, help="target nominal posterior mass instead")
    ap.add_argument("--out", type=Path, help="write TSV here (default: JSON to stdout)")
    args = ap.parse_args(argv)

    from supervdj.cache import ValueCache
    from supervdj.models import load_chain_models

    models = load_chain_models(args.chain, use_sonia=False)
    cache = ValueCache(None)
    cdr3s = [args.cdr3] if args.cdr3 else _read_inputs(args.input)
    results = [calibrated_candidates(c, args.chain, coverage=args.coverage,
                                     mass=args.mass, models=models, cache=cache)
               for c in cdr3s]

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["cdr3", "chain", "status", "n_candidates", "nominal_mass",
                        "achieved_mass", "expected_coverage", "entropy_nats",
                        "genes", "groups"])
            for r in results:
                w.writerow([r["cdr3"], r["chain"], r["status"], r.get("n_candidates", 0),
                            r.get("nominal_mass", ""), r.get("achieved_mass", ""),
                            r.get("expected_coverage", ""), r.get("entropy_nats", ""),
                            ";".join(g for g, _ in r["genes"]),
                            ";".join(str(d["group"]) for d in r["groups"])])
        print(f"wrote {args.out} ({len(results)} sequences)")
    else:
        json.dump(results[0] if len(results) == 1 else results, sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
