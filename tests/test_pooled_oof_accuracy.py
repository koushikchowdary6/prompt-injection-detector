"""Regression: grouped CV accuracy must weight every held-out example equally."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train


def test_evaluate_pools_out_of_fold_predictions_instead_of_fold_means(monkeypatch):
    # An uneven grouped split has one incorrect prediction in a one-row fold
    # and five correct predictions in a five-row fold. Per-fold averaging
    # would report 50% instead of the actual 5/6 held-out accuracy.
    texts = ["attack"] + [f"benign-{i}" for i in range(5)]
    labels = [1, 0, 0, 0, 0, 0]
    monkeypatch.setattr(train.examples, "load", lambda: (texts, labels))
    folds = [
        (np.array([1, 2, 3, 4, 5]), np.array([0])),
        (np.array([0]), np.array([1, 2, 3, 4, 5])),
    ]
    monkeypatch.setattr(train, "grouped_folds", lambda *args, **kwargs: folds)

    class AlwaysBenign:
        def __init__(self, ml_weight):
            pass

        def fit(self, texts, labels):
            return self

        def is_injection(self, text, threshold=0.5):
            return False

    monkeypatch.setattr(train, "PromptInjectionDetector", AlwaysBenign)
    metrics = train.evaluate()
    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
