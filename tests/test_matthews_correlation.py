"""MCC exposes imbalanced and inverted grouped-CV classifications."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train


@pytest.mark.parametrize("counts, expected", [
    ((2, 0, 3, 0), 1.0),
    ((0, 3, 0, 2), -1.0),
    ((1, 1, 1, 1), 0.0),
    ((2, 1, 2, 1), 1 / 3),
])
def test_mcc_uses_all_four_confusion_cells(counts, expected):
    assert train.matthews_correlation(*counts) == pytest.approx(expected)


@pytest.mark.parametrize("counts", [
    (0, 0, 3, 2),  # all predicted benign
    (2, 3, 0, 0),  # all predicted attacks
    (0, 0, 0, 0),  # no observations
])
def test_mcc_is_undefined_for_degenerate_predictions(counts):
    assert train.matthews_correlation(*counts) is None


def test_mcc_flows_through_grouped_cv_threshold_sweep(monkeypatch):
    texts = ["attack-a", "benign-a", "attack-b", "benign-b"]
    labels = [1, 0, 1, 0]
    scores = dict(zip(texts, [0.9, 0.8, 0.7, 0.2]))
    monkeypatch.setattr(train.examples, "load", lambda: (texts, labels))
    monkeypatch.setattr(train, "grouped_folds", lambda *args, **kwargs: [
        (np.array([2, 3]), np.array([0, 1])),
        (np.array([0, 1]), np.array([2, 3])),
    ])

    class StubDetector:
        def __init__(self, ml_weight):
            pass

        def fit(self, texts, labels):
            return self

        def score(self, text):
            return scores[text]

    monkeypatch.setattr(train, "PromptInjectionDetector", StubDetector)
    summaries = train.evaluate_thresholds((0.5, 0.95))
    assert summaries[0]["confusion_matrix"] == {
        "tp": 2, "fp": 1, "tn": 1, "fn": 0
    }
    assert summaries[0]["mcc"] == pytest.approx(1 / 3**0.5)
    assert summaries[1]["confusion_matrix"] == {
        "tp": 0, "fp": 0, "tn": 2, "fn": 2
    }
    assert summaries[1]["mcc"] is None
