"""OLGA wrapper for CDR3/V/J log-Pgen.

Pgen == 0 is OLGA's authoritative ``impossible`` answer — the scoring
layer hard-excludes those candidates. The wrapper only falls back to a
neutral log-Pgen when the OLGA model itself fails to load.
"""

from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Set

from supervdj.models import GeneReference

_NEUTRAL_LOG_PGEN = math.log(1e-12)
_PGEN_MODELS: Dict[str, "object"] = {}
_KNOWN_GENES: Dict[str, Set[str]] = {}
_LOAD_ERRORS: Dict[str, str] = {}

_ALLELE_RE = re.compile(r"\*\d+$")


@dataclass
class OlgaResult:
    """Per-candidate Pgen result."""

    log_pgen: float
    available: bool
    impossible: bool = False
    warning: Optional[str] = None


def _normalize(name: str) -> str:
    """Strip ``*allele`` suffix, whitespace, and upper-case ``TRAV``/``TRBV``."""
    if not isinstance(name, str):
        return ""
    s = name.strip()
    s = _ALLELE_RE.sub("", s)
    # Upper-case the locus prefix (TRAV/TRBV/TRAJ/TRBJ) but keep gene numbering as-is.
    if s[:4].upper() in {"TRAV", "TRAJ", "TRBV", "TRBJ", "TRGV", "TRGJ", "TRDV", "TRDJ"}:
        s = s[:4].upper() + s[4:]
    return s


def _model_folder(chain: str) -> str:
    import olga

    chain = chain.upper()
    sub = "human_T_alpha" if chain == "TRA" else "human_T_beta"
    return os.path.join(os.path.dirname(olga.__file__), "default_models", sub)


def _load_known_genes(folder: str) -> Set[str]:
    """Read OLGA's anchor CSVs and return the set of accepted gene names."""
    names: Set[str] = set()
    for fname in ("V_gene_CDR3_anchors.csv", "J_gene_CDR3_anchors.csv"):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            for line in fh.read().splitlines()[1:]:
                if not line:
                    continue
                full = line.split(",", 1)[0]
                names.add(full)              # TRBV6-5*01
                names.add(full.split("*")[0])  # TRBV6-5
    return names


def _load_pgen_model(chain: str):
    """Load (and memoize) the default human OLGA Pgen model for the chain."""
    chain = chain.upper()
    if chain in _PGEN_MODELS:
        return _PGEN_MODELS[chain]
    if chain in _LOAD_ERRORS:
        return None
    try:
        import olga.generation_probability as pgen
        import olga.load_model as load_model

        folder = _model_folder(chain)
        params = os.path.join(folder, "model_params.txt")
        marginals = os.path.join(folder, "model_marginals.txt")
        v_anchors = os.path.join(folder, "V_gene_CDR3_anchors.csv")
        j_anchors = os.path.join(folder, "J_gene_CDR3_anchors.csv")

        if chain == "TRA":
            genomic_data = load_model.GenomicDataVJ()
            genomic_data.load_igor_genomic_data(params, v_anchors, j_anchors)
            generative_model = load_model.GenerativeModelVJ()
            generative_model.load_and_process_igor_model(marginals)
            model = pgen.GenerationProbabilityVJ(generative_model, genomic_data)
        else:
            genomic_data = load_model.GenomicDataVDJ()
            genomic_data.load_igor_genomic_data(params, v_anchors, j_anchors)
            generative_model = load_model.GenerativeModelVDJ()
            generative_model.load_and_process_igor_model(marginals)
            model = pgen.GenerationProbabilityVDJ(generative_model, genomic_data)

        _PGEN_MODELS[chain] = model
        _KNOWN_GENES[chain] = _load_known_genes(folder)
        return model
    except Exception as exc:
        _LOAD_ERRORS[chain] = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"OLGA model load failed for {chain}: {exc}")
        return None


def compute_log_pgen(
    cdr3: str,
    v: GeneReference,
    j: GeneReference,
    chain: str,
) -> OlgaResult:
    """Return log Pgen for ``(cdr3, V, J)`` from OLGA.

    Pgen == 0 is reported as ``impossible=True`` — the scoring layer
    drops impossible candidates rather than carrying neutral scores.
    """
    chain = chain.upper()
    model = _load_pgen_model(chain)
    if model is None:
        return OlgaResult(
            log_pgen=_NEUTRAL_LOG_PGEN,
            available=False,
            warning=(
                f"OLGA model unavailable for {chain} "
                f"({_LOAD_ERRORS.get(chain, 'unknown')}); using neutral log-Pgen."
            ),
        )
    v_name = _normalize(v.gene)
    j_name = _normalize(j.gene)
    known = _KNOWN_GENES.get(chain, set())
    if known and v_name not in known:
        return OlgaResult(
            log_pgen=_NEUTRAL_LOG_PGEN,
            available=True,
            impossible=True,
            warning=f"V gene {v_name!r} is not in OLGA's known set for {chain}.",
        )
    if known and j_name not in known:
        return OlgaResult(
            log_pgen=_NEUTRAL_LOG_PGEN,
            available=True,
            impossible=True,
            warning=f"J gene {j_name!r} is not in OLGA's known set for {chain}.",
        )
    pgen_value = model.compute_aa_CDR3_pgen(
        cdr3, v_name, j_name, print_warnings=False
    )
    if not pgen_value or pgen_value <= 0:
        # OLGA's authoritative "impossible" answer for this CDR3 / V / J.
        return OlgaResult(
            log_pgen=_NEUTRAL_LOG_PGEN,
            available=True,
            impossible=True,
        )
    return OlgaResult(log_pgen=float(math.log(pgen_value)), available=True)


def compute_log_v_pgen(
    cdr3: str,
    v: GeneReference,
    chain: str,
) -> OlgaResult:
    """Return log Pgen for ``cdr3`` conditioned on V and marginalized over J."""
    chain = chain.upper()
    model = _load_pgen_model(chain)
    if model is None:
        return OlgaResult(
            log_pgen=_NEUTRAL_LOG_PGEN,
            available=False,
            warning=(
                f"OLGA model unavailable for {chain} "
                f"({_LOAD_ERRORS.get(chain, 'unknown')}); using neutral V log-Pgen."
            ),
        )
    v_name = _normalize(v.gene)
    known = _KNOWN_GENES.get(chain, set())
    if known and v_name not in known:
        return OlgaResult(
            log_pgen=_NEUTRAL_LOG_PGEN,
            available=True,
            impossible=True,
            warning=f"V gene {v_name!r} is not in OLGA's known set for {chain}.",
        )
    pgen_value = model.compute_aa_CDR3_pgen(
        cdr3, v_name, None, print_warnings=False
    )
    if not pgen_value or pgen_value <= 0:
        return OlgaResult(
            log_pgen=_NEUTRAL_LOG_PGEN,
            available=True,
            impossible=True,
        )
    return OlgaResult(log_pgen=float(math.log(pgen_value)), available=True)
