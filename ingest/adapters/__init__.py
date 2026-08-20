"""Raw-input adapters.

An adapter exposes ``SOURCE`` and ``read_raw(path) -> Iterator[dict]`` yielding
rows with keys ``cdr3_raw, v_raw, j_raw, chain``. Adapters parse only their own
input format; all shared normalization lives in :mod:`ingest.normalize`.

:mod:`ingest.adapters.generic` handles any CDR3 table with chain, CDR3, V and J
columns, which is every input the pipeline needs. Add a module here only if you
have a format its column resolution cannot read.
"""

from __future__ import annotations

from ingest.adapters import generic

ADAPTERS = {generic.SOURCE: generic}
