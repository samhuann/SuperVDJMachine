"""SONIA / SONNIA wrapper for the selection factor Q.

Falls back to a neutral log-selection only when the SONIA model fails to
load. Q == 0 is reported as ``impossible`` so the scoring layer can drop
the candidate.
"""

from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from supervdj.models import GeneReference

_NEUTRAL_LOG_SELECTION = 0.0
_SONIA_BUNDLES: Dict[str, Tuple[object, object]] = {}
_LOAD_ERRORS: Dict[str, str] = {}

_ALLELE_RE = re.compile(r"\*\d+$")


@dataclass
class SoniaResult:
    """Per-candidate selection result."""

    log_selection: float
    available: bool
    impossible: bool = False
    warning: Optional[str] = None


def _normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    s = _ALLELE_RE.sub("", s)
    if s[:4].upper() in {"TRAV", "TRAJ", "TRBV", "TRBJ", "TRGV", "TRGJ", "TRDV", "TRDJ"}:
        s = s[:4].upper() + s[4:]
    return s


def _default_dir(chain: str) -> str:
    import sonia

    chain = chain.upper()
    sub = "human_T_alpha" if chain == "TRA" else "human_T_beta"
    return os.path.join(os.path.dirname(sonia.__file__), "default_models", sub)


def _load(chain: str) -> Optional[Tuple[object, object]]:
    """Load the default SONIA model + evaluator for the chain (memoized)."""
    chain = chain.upper()
    if chain in _SONIA_BUNDLES:
        return _SONIA_BUNDLES[chain]
    if chain in _LOAD_ERRORS:
        return None
    try:
        from sonia.evaluate_model import EvaluateModel
        from sonia.sonia_leftpos_rightpos import SoniaLeftposRightpos

        chain_type = "humanTRA" if chain == "TRA" else "humanTRB"
        load_dir = _default_dir(chain)
        if not os.path.isdir(load_dir):
            raise FileNotFoundError(f"SONIA default model dir missing: {load_dir}")
        model = SoniaLeftposRightpos(
            chain_type=chain_type,
            load_dir=load_dir,
            load_seqs=False,
        )
        evaluator = EvaluateModel(sonia_model=model)
        _SONIA_BUNDLES[chain] = (model, evaluator)
        return _SONIA_BUNDLES[chain]
    except Exception as exc:
        _LOAD_ERRORS[chain] = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"SONIA model load failed for {chain}: {exc}")
        return None


def compute_log_selection(
    cdr3: str,
    v: GeneReference,
    j: GeneReference,
    chain: str,
) -> SoniaResult:
    """Return log of SONIA's selection factor Q for ``(cdr3, V, J)``."""
    chain = chain.upper()
    bundle = _load(chain)
    if bundle is None:
        return SoniaResult(
            log_selection=_NEUTRAL_LOG_SELECTION,
            available=False,
            warning=(
                f"SONIA model unavailable for {chain} "
                f"({_LOAD_ERRORS.get(chain, 'unknown')}); using neutral log-selection."
            ),
        )
    _, evaluator = bundle
    v_name = _normalize(v.gene)
    j_name = _normalize(j.gene)
    q = evaluator.evaluate_selection_factors([[cdr3, v_name, j_name]])
    q_value = float(q[0])
    if not q_value or q_value <= 0:
        return SoniaResult(
            log_selection=_NEUTRAL_LOG_SELECTION,
            available=True,
            impossible=True,
        )
    return SoniaResult(log_selection=float(math.log(q_value)), available=True)


def compute_log_selection_many(
    cdr3: str,
    pairs: Sequence[Tuple[GeneReference, GeneReference]],
    chain: str,
) -> List[SoniaResult]:
    """Return SONIA log-selection results for many ``(V, J)`` pairs at once."""
    if not pairs:
        return []
    chain = chain.upper()
    bundle = _load(chain)
    if bundle is None:
        warning = (
            f"SONIA model unavailable for {chain} "
            f"({_LOAD_ERRORS.get(chain, 'unknown')}); using neutral log-selection."
        )
        return [
            SoniaResult(
                log_selection=_NEUTRAL_LOG_SELECTION,
                available=False,
                warning=warning,
            )
            for _ in pairs
        ]

    _, evaluator = bundle
    seqs = [
        [cdr3, _normalize(v.gene), _normalize(j.gene)]
        for v, j in pairs
    ]
    q_values = evaluator.evaluate_selection_factors(seqs)
    out: List[SoniaResult] = []
    for q in q_values:
        q_value = float(q)
        if not q_value or q_value <= 0:
            out.append(
                SoniaResult(
                    log_selection=_NEUTRAL_LOG_SELECTION,
                    available=True,
                    impossible=True,
                )
            )
        else:
            out.append(
                SoniaResult(
                    log_selection=float(math.log(q_value)),
                    available=True,
                )
            )
    return out
