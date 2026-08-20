"""supervdj: OLGA/SONNIA V and J gene posteriors from CDR3 sequences.

Pretrained models only (OLGA for Pgen + candidate gene set, SONNIA for the
post-selection factor Q).  The per-sequence posterior table written by
:mod:`supervdj.run` is the single source of truth for all later analysis.
"""

from supervdj.analyze import analyze_sequence
from supervdj.cache import ValueCache
from supervdj.io import CDR3Row, load_dataset
from supervdj.models import ChainModels, load_chain_models
from supervdj.posterior import (
    postselection_posterior,
    preselection_posterior,
    summarize,
)

__all__ = [
    "CDR3Row",
    "ChainModels",
    "ValueCache",
    "analyze_sequence",
    "load_chain_models",
    "load_dataset",
    "postselection_posterior",
    "preselection_posterior",
    "summarize",
]
__version__ = "0.1.0"
