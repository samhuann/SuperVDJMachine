"""Persistent value cache so repeated OLGA/SONNIA evaluations are cheap.

A single key -> float store backed by one pickle file.  Keys are short
strings built from ``(kind, chain, cdr3, v, j)`` so OLGA ``Pgen`` and SONNIA
``Q`` values survive across runs.  This is what makes reruns cheap: the
expensive model calls happen once.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Optional


class ValueCache:
    """In-memory ``dict`` of float values flushed to a pickle file on disk."""

    def __init__(self, path: Optional[Path], flush_every: int = 20000):
        self.path = Path(path) if path is not None else None
        self.flush_every = flush_every
        self._store: Dict[str, float] = {}
        self._since_flush = 0
        self.hits = 0
        self.misses = 0
        if self.path is not None and self.path.is_file():
            with open(self.path, "rb") as fh:
                self._store = pickle.load(fh)

    @staticmethod
    def key(kind: str, chain: str, cdr3: str, v: object, j: object) -> str:
        """Build a cache key; ``None`` masks are marginalized axes."""
        return f"{kind}|{chain}|{cdr3}|{'*' if v is None else v}|{'*' if j is None else j}"

    def get(self, key: str) -> Optional[float]:
        val = self._store.get(key)
        if val is None and key not in self._store:
            self.misses += 1
            return None
        self.hits += 1
        return val

    def set(self, key: str, value: float) -> None:
        self._store[key] = value
        self._since_flush += 1
        if self.path is not None and self._since_flush >= self.flush_every:
            self.save()

    def __len__(self) -> int:
        return len(self._store)

    def save(self) -> None:
        """Atomically persist the store to ``path`` (no-op if path is None)."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(self._store, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        self._since_flush = 0

    def __enter__(self) -> "ValueCache":
        return self

    def __exit__(self, *exc) -> None:
        self.save()
