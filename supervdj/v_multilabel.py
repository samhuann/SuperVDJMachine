"""Linear multi-label V-gene scorer from CDR3 amino-acid strings."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from supervdj._tables import strip_allele


Rows = List[Tuple[str, List[str]]]


def load_neotcr_v_rows(path: Path, chain: str) -> Rows:
    """Load real NeoTCR ``(CDR3, [V genes])`` rows for TRA or TRB."""
    chain = chain.upper()
    if chain == "TRA":
        cdr3_col, v_col = "TRA_CDR3", "TRAV"
    elif chain == "TRB":
        cdr3_col, v_col = "TRB_CDR3", "TRBV"
    else:
        raise ValueError("chain must be TRA or TRB")

    df = pd.read_excel(path, sheet_name="All")
    grouped: Dict[str, set[str]] = defaultdict(set)
    for record in df[[cdr3_col, v_col]].dropna().to_dict("records"):
        cdr3 = str(record[cdr3_col]).strip().upper()
        gene = strip_allele(str(record[v_col]))
        if cdr3 and gene:
            grouped[cdr3].add(gene)
    return [(cdr3, sorted(labels)) for cdr3, labels in grouped.items()]


def split_rows(rows: Rows, test_fraction: float = 0.2, seed: int = 1) -> Tuple[Rows, Rows]:
    """CDR3-disjoint train/test split."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    n_test = max(1, int(round(len(rows) * test_fraction)))
    test_idx = set(indices[:n_test].tolist())
    train = [row for i, row in enumerate(rows) if i not in test_idx]
    test = [row for i, row in enumerate(rows) if i in test_idx]
    return train, test


class LinearVGeneModel:
    """One-vs-rest logistic V-gene model over character n-grams.

    The model is multi-label: a CDR3 can carry more than one acceptable V label
    when duplicate rows or ambiguous annotations are present.
    """

    def __init__(self, chain: str, vectorizer, classifier, labeler: MultiLabelBinarizer):
        self.chain = chain.upper()
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.labeler = labeler
        self.classes = [str(c) for c in labeler.classes_]
        self.genes = set(self.classes)

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Tuple[str, Sequence[str]]],
        chain: str,
        min_label_count: int = 2,
        max_iter: int = 1000,
    ) -> "LinearVGeneModel":
        normalized = [
            (str(cdr3 or "").strip().upper(), [strip_allele(v) for v in labels])
            for cdr3, labels in rows
        ]
        normalized = [
            (cdr3, sorted({v for v in labels if v}))
            for cdr3, labels in normalized
            if cdr3 and labels
        ]
        counts: Dict[str, int] = defaultdict(int)
        for _, labels in normalized:
            for label in labels:
                counts[label] += 1
        normalized = [
            (cdr3, [label for label in labels if counts[label] >= min_label_count])
            for cdr3, labels in normalized
        ]
        normalized = [(cdr3, labels) for cdr3, labels in normalized if labels]
        if len(normalized) < 2:
            raise ValueError("not enough rows after label-count filtering")

        x_text = [cdr3 for cdr3, _ in normalized]
        y_labels = [labels for _, labels in normalized]
        labeler = MultiLabelBinarizer()
        y = labeler.fit_transform(y_labels)
        if y.shape[1] < 2:
            raise ValueError("need at least two V labels to train a ranker")

        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            lowercase=False,
            min_df=1,
        )
        x = vectorizer.fit_transform(x_text)
        base = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=max_iter,
        )
        classifier = OneVsRestClassifier(base)
        classifier.fit(x, y)
        return cls(chain, vectorizer, classifier, labeler)

    def _probabilities(self, cdr3: str) -> Dict[str, float]:
        x = self.vectorizer.transform([str(cdr3 or "").strip().upper()])
        probs = np.asarray(self.classifier.predict_proba(x))[0]
        return {gene: float(prob) for gene, prob in zip(self.classes, probs)}

    def expand_genes(self, cdr3: str, candidate_genes: Sequence[str]) -> List[str]:
        genes = list(dict.fromkeys(strip_allele(gene) for gene in candidate_genes if strip_allele(gene)))
        seen = set(genes)
        for gene in self.classes:
            if gene not in seen:
                seen.add(gene)
                genes.append(gene)
        return genes

    def log_scores(self, cdr3: str, genes: Sequence[str]) -> Dict[str, float]:
        probs = self._probabilities(cdr3)
        return {
            strip_allele(gene): math.log(max(probs.get(strip_allele(gene), 1e-12), 1e-12))
            for gene in genes
        }

    def rank(self, cdr3: str, genes: Sequence[str] | None = None) -> List[Tuple[str, float]]:
        genes = list(genes) if genes is not None else self.classes
        scores = self.log_scores(cdr3, genes)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "chain": self.chain,
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "labeler": self.labeler,
            },
            path / "linear_v_multilabel.joblib",
        )

    @classmethod
    def load(cls, path: Path) -> "LinearVGeneModel":
        payload = joblib.load(Path(path) / "linear_v_multilabel.joblib")
        return cls(
            payload["chain"],
            payload["vectorizer"],
            payload["classifier"],
            payload["labeler"],
        )


def topk_recall(model: LinearVGeneModel, rows: Rows, k: int) -> float:
    if not rows:
        return 0.0
    hits = 0
    for cdr3, labels in rows:
        top = {gene for gene, _ in model.rank(cdr3)[:k]}
        if any(strip_allele(label) in top for label in labels):
            hits += 1
    return hits / len(rows)


def mean_reciprocal_rank(model: LinearVGeneModel, rows: Rows) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for cdr3, labels in rows:
        truth = {strip_allele(label) for label in labels}
        for rank, (gene, _) in enumerate(model.rank(cdr3), start=1):
            if gene in truth:
                total += 1.0 / rank
                break
    return total / len(rows)
