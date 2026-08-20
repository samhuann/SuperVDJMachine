"""Canonical schema for the ingested dataset.

Every adapter must emit exactly these columns and nothing else.  The
``source`` column records provenance; the four-tuple
``(cdr3_aa, v_gene, j_gene, chain)`` is the deduplication key (see
:mod:`ingest.dedup`).
"""

from __future__ import annotations

# The canonical table has exactly these columns, in this order.
CANONICAL_COLUMNS = ["cdr3_aa", "v_gene", "j_gene", "chain", "source"]

# The tuple that defines a "rearrangement" for deduplication.  Deliberately
# NOT cdr3_aa alone: the same CDR3 with different V/J is a distinct row.
DEDUP_KEY = ["cdr3_aa", "v_gene", "j_gene", "chain"]

# Columns of the rejected-rows file (rows that could not be normalized).
REJECTED_COLUMNS = ["source", "chain", "cdr3_raw", "v_raw", "j_raw", "reason"]
