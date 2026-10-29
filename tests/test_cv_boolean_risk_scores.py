"""A binary decision is not a numeric risk score in held-out CV."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train


@pytest.mark.parametrize("invalid_score", [True, False, np.bool_(True), np.bool_(False)])
def test_boolean_scores_fail_closed_before_metrics_are_reported(monkeypatch, invalid_score):
    texts = ["attack_one", "benign_one", "attack_two", "benign_two"]
    labels = [1, 0, 1, 0]
    monkeypatch.setattr(train.examples, "load", lambda: (texts, labels))
    monkeypatch.setattr(train, "grouped_folds", lambda *args, **kwargs: [
        (np.array([2, 3]), np.array([0, 1])),
        (np.array([0, 1]), np.array([2, 3])),
    ])

    class BooleanScoreDetector:
        def __init__(self, ml_weight):
            pass

        def fit(self, texts, labels):
            return self

        def score(self, text):
            return invalid_score

    monkeypatch.setattr(train, "PromptInjectionDetector", BooleanScoreDetector)
    with pytest.raises(ValueError, match="invalid risk score"):
        train.evaluate_thresholds((0.3, 0.5, 0.7))


def test_numeric_zero_and_one_remain_valid_risk_scores(monkeypatch):
    texts = ["attack_one", "benign_one", "attack_two", "benign_two"]
    labels = [1, 0, 1, 0]
    monkeypatch.setattr(train.examples, "load", lambda: (texts, labels))
    monkeypatch.setattr(train, "grouped_folds", lambda *args, **kwargs: [
        (np.array([2, 3]), np.array([0, 1])),
        (np.array([0, 1]), np.array([2, 3])),
    ])

    class NumericScoreDetector:
        def __init__(self, ml_weight):
            pass

        def fit(self, texts, labels):
            return self

        def score(self, text):
            return 1 if text.startswith("attack") else 0

    monkeypatch.setattr(train, "PromptInjectionDetector", NumericScoreDetector)
    summary = train.evaluate_thresholds((0.5,))[0]
    assert summary["confusion_matrix"] == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}
