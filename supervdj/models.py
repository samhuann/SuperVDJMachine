"""Pretrained model loaders for the V/J posterior pipeline.

Two pretrained models back the pipeline; neither is ever trained here:

* **OLGA** supplies the candidate gene set and the generation probability
  ``Pgen``.  It is the single source of the gene universe, so the
  pre-selection and post-selection posteriors are defined over exactly the
  same candidates (per the spec: "the candidate gene set is whatever V and J
  genes exist in the loaded OLGA model").
* **SONNIA** (the ``sonnia`` package) supplies the post-selection factor
  ``Q``.  Only the *pretrained linear* T-cell models ship with weights, so
  that is what we load; the deep ``SoNNia`` variant would require training,
  which the constraints forbid.

``Pgen`` is taken from OLGA for both arms so the two posteriors share one
gene universe; SONNIA contributes ``Q`` only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

# Keras 3 (used by sonnia) selects its backend at import time.  Torch is the
# backend present in this environment; pin it before sonnia/keras import.
os.environ.setdefault("KERAS_BACKEND", "torch")

_ALLELE_RE = re.compile(r"\*\d+$")

#: Map a chain to the OLGA/SONNIA default-model folder name.
_CHAIN_TO_MODEL = {
    "TRA": "human_T_alpha",
    "TRB": "human_T_beta",
}


def strip_allele(name: str) -> str:
    """``TRBV19*01`` -> ``TRBV19``; trims whitespace, leaves locus casing."""
    if not isinstance(name, str):
        return ""
    return _ALLELE_RE.sub("", name.strip())


@dataclass
class ChainModels:
    """Loaded pretrained models and candidate gene sets for one chain.

    Attributes:
        chain: ``TRA`` or ``TRB``.
        olga: OLGA ``GenerationProbability{VJ,VDJ}`` instance.
        sonia: SONNIA selection model (pretrained linear ``Sonia``), or
            ``None`` when the post-selection arm is disabled.
        v_genes: Sorted gene-level V candidates pulled from the OLGA model.
        j_genes: Sorted gene-level J candidates pulled from the OLGA model.
    """

    chain: str
    olga: object
    sonia: Optional[object]
    v_genes: List[str] = field(default_factory=list)
    j_genes: List[str] = field(default_factory=list)

    def candidates(self, axis: str) -> List[str]:
        return self.v_genes if axis.upper() == "V" else self.j_genes


def _olga_model_dir(chain: str) -> str:
    import olga

    sub = _CHAIN_TO_MODEL[chain.upper()]
    return os.path.join(os.path.dirname(olga.__file__), "default_models", sub)


def load_olga(chain: str):
    """Load the default human OLGA Pgen model for ``chain`` (TRA=VJ, TRB=VDJ)."""
    import olga.generation_probability as pgen
    import olga.load_model as load_model

    chain = chain.upper()
    if chain not in _CHAIN_TO_MODEL:
        raise ValueError(f"Unsupported chain: {chain!r} (expected TRA or TRB)")

    folder = _olga_model_dir(chain)
    params = os.path.join(folder, "model_params.txt")
    marginals = os.path.join(folder, "model_marginals.txt")
    v_anchors = os.path.join(folder, "V_gene_CDR3_anchors.csv")
    j_anchors = os.path.join(folder, "J_gene_CDR3_anchors.csv")

    if chain == "TRA":
        genomic = load_model.GenomicDataVJ()
        genomic.load_igor_genomic_data(params, v_anchors, j_anchors)
        gen = load_model.GenerativeModelVJ()
        gen.load_and_process_igor_model(marginals)
        model = pgen.GenerationProbabilityVJ(gen, genomic)
    else:
        genomic = load_model.GenomicDataVDJ()
        genomic.load_igor_genomic_data(params, v_anchors, j_anchors)
        gen = load_model.GenerativeModelVDJ()
        gen.load_and_process_igor_model(marginals)
        model = pgen.GenerationProbabilityVDJ(gen, genomic)

    # Stash the genomic data so the candidate gene list is reachable.
    model._vjp_genomic = genomic
    return model


def candidate_genes_from_olga(olga_model) -> Tuple[List[str], List[str]]:
    """Pull the gene-level V and J candidate sets from the loaded OLGA model.

    OLGA stores alleles (``TRBV10-1*01``, ``*02`` ...); we collapse to genes.
    """
    genomic = olga_model._vjp_genomic
    v_genes = sorted({strip_allele(g[0]) for g in genomic.genV})
    j_genes = sorted({strip_allele(g[0]) for g in genomic.genJ})
    return v_genes, j_genes


def load_sonia(chain: str):
    """Load the pretrained linear SONNIA selection model for ``chain``."""
    from sonnia.sonia import Sonia

    chain = chain.upper()
    if chain not in _CHAIN_TO_MODEL:
        raise ValueError(f"Unsupported chain: {chain!r} (expected TRA or TRB)")
    return Sonia(ppost_model=_CHAIN_TO_MODEL[chain])


@lru_cache(maxsize=None)
def load_chain_models(chain: str, use_sonia: bool = True) -> ChainModels:
    """Load (and memoize) OLGA + optional SONNIA models for ``chain``.

    The candidate gene universe is taken from the OLGA model so both the
    pre- and post-selection posteriors range over identical candidates.
    """
    chain = chain.upper()
    olga_model = load_olga(chain)
    v_genes, j_genes = candidate_genes_from_olga(olga_model)
    sonia_model = load_sonia(chain) if use_sonia else None
    return ChainModels(
        chain=chain,
        olga=olga_model,
        sonia=sonia_model,
        v_genes=v_genes,
        j_genes=j_genes,
    )
