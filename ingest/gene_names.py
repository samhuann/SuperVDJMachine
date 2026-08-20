"""Reconcile each source's V/J gene names onto the OLGA model's gene list.

The OLGA model defines the candidate set, so OLGA's gene names are the only
naming that counts.  A source name is reconciled by generating an ordered list
of IMGT-style candidate names (most specific first) and returning the first
candidate that exists in the OLGA gene set.  Names that map to nothing are
reported with a reason; they are never silently dropped and never kept with a
non-OLGA name.

Handled source conventions (all observed in the real inputs):

* IMGT with allele:        ``TRBV27*01`` -> ``TRBV27``
* IMGT with ``/DV``:       ``TRAV38-2/DV8*01`` -> ``TRAV38-2/DV8``
* Adaptive nomenclature:   ``TCRBV01-01`` -> ``TRBV1``; ``TCRBV20-01`` -> ``TRBV20-1``
* leading zeros:           ``TRAV1-01`` -> ``TRAV1-1``
* colon allele separator:  ``TRAV1-1:01`` -> ``TRAV1-1``

Family-only names (e.g. ``TRAV1`` where OLGA has ``TRAV1-1`` and ``TRAV1-2``)
are genuinely ambiguous and are rejected, not guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from supervdj.models import load_chain_models

_ALLELE = re.compile(r"[\*:]\d+.*$")     # *01 / :01 and anything after
_DIGITS = re.compile(r"\d+")


def _strip_leading_zeros(name: str) -> str:
    """``TRBV01-01`` -> ``TRBV1-1`` (normalize each numeric group)."""
    return _DIGITS.sub(lambda m: str(int(m.group())), name)


def _candidates(raw: str) -> List[str]:
    """Ordered IMGT-style candidate names for a raw source name (most specific first)."""
    s = raw.strip().upper().replace(" ", "")
    if not s:
        return []
    # Adaptive 'TCRBV...' / 'TCRBJ...' -> IMGT 'TRBV...' / 'TRBJ...'
    if s.startswith("TCR"):
        s = "TR" + s[3:]
    s = _ALLELE.sub("", s)                # drop allele suffix
    s = s.rstrip("*").rstrip("-")
    if not s:
        return []
    s = _strip_leading_zeros(s)
    cands = [s]
    # Family fallback: drop a trailing '-<n>' subfamily (handles Adaptive
    # '-01' on single-gene families, e.g. TRBV13-2 -> TRBV13, TRBV1-1 -> TRBV1).
    m = re.match(r"^(TR[AB][VJ]\d+)-\d+$", s)
    if m:
        cands.append(m.group(1))
    return cands


@dataclass
class GeneReconciler:
    """Maps source V/J names onto the loaded OLGA model's gene set per chain."""

    v_sets: dict   # chain -> set of OLGA V gene names
    j_sets: dict   # chain -> set of OLGA J gene names

    @classmethod
    def from_olga(cls, chains=("TRA", "TRB")) -> "GeneReconciler":
        v_sets, j_sets = {}, {}
        for ch in chains:
            m = load_chain_models(ch, use_sonia=False)
            v_sets[ch] = set(m.v_genes)
            j_sets[ch] = set(m.j_genes)
        return cls(v_sets=v_sets, j_sets=j_sets)

    def _map(self, raw: str, chain: str, valid: Set[str]) -> Tuple[Optional[str], str]:
        if raw is None or str(raw).strip() == "" or str(raw).lower() == "nan":
            return None, "empty"
        cands = _candidates(str(raw))
        if not cands:
            return None, f"unparseable:{raw!r}"
        for c in cands:
            if c in valid:
                return c, ""
        return None, f"unmapped:{raw!r}->{cands} not in OLGA {chain} set"

    def map_v(self, raw: str, chain: str) -> Tuple[Optional[str], str]:
        return self._map(raw, chain, self.v_sets[chain])

    def map_j(self, raw: str, chain: str) -> Tuple[Optional[str], str]:
        return self._map(raw, chain, self.j_sets[chain])
