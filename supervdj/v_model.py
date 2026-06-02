"""Supervised V-gene model: a 1D-CNN over CDR3 residues.

This is the strongest CDR3-only V ranker for TRB (see the README benchmark).
TensorFlow is imported lazily so the rest of the package and the test suite
never need it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from supervdj._tables import read_table as _read_table
from supervdj._tables import resolve_column as _resolve_column
from supervdj._tables import strip_allele as _strip_allele


def _tsv_rows(path: Path, chain: str) -> Tuple[List[Tuple[str, str]], str]:
    """Load ``(cdr3, v)`` rows for ``chain`` from a gene/cdr3/v table."""
    df = _read_table(path)
    chain_col = _resolve_column(df, "gene", "Gene")
    cdr3_col = _resolve_column(df, "cdr3", "CDR3")
    v_col = _resolve_column(df, "v", "V", "v.segm")
    species_col = next((c for c in df.columns if c.lower() == "species"), None)

    chain_norm = chain.upper()
    if species_col is not None:
        df = df[df[species_col] == "HomoSapiens"]
    df = df[df[chain_col].str.upper() == chain_norm]
    rows = [
        (str(row[cdr3_col]), str(row[v_col]))
        for row in df[[cdr3_col, v_col]].dropna().to_dict("records")
    ]
    return rows, chain_norm


_AA = "ACDEFGHIKLMNPQRSTVWY"


class VGeneCNN:
    """1D-CNN over residue embeddings for V-gene scoring from a CDR3 sequence.

    On a CDR3-disjoint PRJNA280417 split it reaches Top-5 ~= 68% and Top-10 ~=
    82%, where both added model capacity and added training data plateau -- i.e.
    near the information limit of the amino-acid CDR3 for V assignment.

    The model carries no motif/usage prior: blending those terms only dilutes
    its ranking, so it scores V genes alone. TensorFlow is imported lazily.
    """

    MAXLEN = 24

    def __init__(self, chain: str, model, classes: Sequence[str], maxlen: int = MAXLEN):
        self.chain = chain.upper()
        self._model = model
        self.classes = list(classes)
        self.genes = set(self.classes)
        self.maxlen = int(maxlen)
        self._last_cdr3: Optional[str] = None
        self._last_logp: Dict[str, float] = {}

    @staticmethod
    def _encode(seqs: Sequence[str], maxlen: int):
        idx = {a: i + 1 for i, a in enumerate(_AA)}  # 0 = pad
        x = np.zeros((len(seqs), maxlen), dtype=np.int32)
        for r, s in enumerate(seqs):
            for i, aa in enumerate(str(s or "").strip().upper()[:maxlen]):
                x[r, i] = idx.get(aa, 0)
        return x

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Tuple[str, str]],
        chain: str,
        epochs: int = 12,
        batch_size: int = 256,
        seed: int = 1,
    ) -> "VGeneCNN":
        import tensorflow as tf
        from tensorflow.keras import layers, models

        data = [
            (str(c or "").strip().upper(), _strip_allele(str(v or "")))
            for c, v in rows
        ]
        data = [(c, v) for c, v in data if c and v]
        classes = sorted({v for _, v in data})
        cls_idx = {g: i for i, g in enumerate(classes)}
        x = cls._encode([c for c, _ in data], cls.MAXLEN)
        y = np.array([cls_idx[v] for _, v in data])

        tf.random.set_seed(seed)
        inp = layers.Input(shape=(cls.MAXLEN,))
        h = layers.Embedding(len(_AA) + 1, 32, mask_zero=True)(inp)
        convs = [layers.Conv1D(128, k, padding="same", activation="relu")(h)
                 for k in (2, 3, 4, 5)]
        h = layers.Concatenate()(convs)
        h = layers.GlobalMaxPooling1D()(h)
        h = layers.Dropout(0.3)(h)
        h = layers.Dense(256, activation="relu")(h)
        h = layers.Dropout(0.3)(h)
        out = layers.Dense(len(classes), activation="softmax")(h)
        model = models.Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy")
        model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0,
                  validation_split=0.05,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor="val_loss", patience=2, restore_best_weights=True)])
        return cls(chain, model, classes)

    @classmethod
    def from_tsv(cls, path: Path, chain: str, **kwargs) -> "VGeneCNN":
        rows, chain_norm = _tsv_rows(path, chain)
        return cls.from_rows(rows, chain=chain_norm, **kwargs)

    def _logp(self, cdr3: str) -> Dict[str, float]:
        cdr3_norm = str(cdr3 or "").strip().upper()
        if cdr3_norm != self._last_cdr3:
            x = self._encode([cdr3_norm], self.maxlen)
            probs = np.asarray(self._model(x, training=False))[0]
            logp = np.log(np.maximum(probs, 1e-12))
            self._last_cdr3 = cdr3_norm
            self._last_logp = {g: float(logp[i]) for i, g in enumerate(self.classes)}
        return self._last_logp

    def expand_genes(self, cdr3: str, candidate_genes: Sequence[str]) -> List[str]:
        genes = list(dict.fromkeys(candidate_genes))
        seen = set(genes)
        for gene in self.classes:
            if gene not in seen:
                seen.add(gene)
                genes.append(gene)
        return genes

    def log_scores(self, cdr3: str, genes: Sequence[str]) -> Dict[str, float]:
        logp = self._logp(cdr3)
        return {gene: logp.get(gene, -30.0) for gene in genes}

    def rank(self, cdr3: str, genes: Sequence[str]) -> List[Tuple[str, float]]:
        scores = self.log_scores(cdr3, genes)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def save(self, path: Path) -> None:
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path / "model"))  # TF SavedModel dir (TF 2.12)
        (path / "meta.json").write_text(json.dumps(
            {"chain": self.chain, "classes": self.classes, "maxlen": self.maxlen}))

    @classmethod
    def load(cls, path: Path) -> "VGeneCNN":
        import json

        import tensorflow as tf

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        model = tf.keras.models.load_model(str(path / "model"))
        return cls(meta["chain"], model, meta["classes"], meta.get("maxlen", cls.MAXLEN))
