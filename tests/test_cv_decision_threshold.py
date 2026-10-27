"""Decision thresholds expose false-positive / missed-attack tradeoffs in held-out CV."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train


def test_threshold_changes_pooled_guardrail_errors(monkeypatch):
    texts = ["attack_one", "attack_two", "benign_one", "benign_two"]
    labels = [1, 1, 0, 0]
    scores = dict(zip(texts, [0.8, 0.45, 0.55, 0.1]))
    monkeypatch.setattr(train.examples, "load", lambda: (texts, labels))
    monkeypatch.setattr(train, "grouped_folds", lambda *a, **kw: [
        (np.array([1, 3]), np.array([0, 2])),
        (np.array([0, 2]), np.array([1, 3])),
    ])

    class StubDetector:
        def __init__(self, ml_weight):
            pass

        def fit(self, texts, labels):
            return self

        def is_injection(self, text, threshold=0.5):
            return scores[text] >= threshold

    monkeypatch.setattr(train, "PromptInjectionDetector", StubDetector)
    permissive = train.evaluate(threshold=0.3)
    strict = train.evaluate(threshold=0.6)
    assert permissive["threshold"] == 0.3
    assert strict["threshold"] == 0.6
    assert permissive["confusion_matrix"] == {"tp": 2, "fp": 1, "tn": 1, "fn": 0}
    assert strict["confusion_matrix"] == {"tp": 1, "fp": 0, "tn": 2, "fn": 1}
    assert permissive["false_positive_rate"] == pytest.approx(0.5)
    assert permissive["false_negative_rate"] == 0
    assert strict["false_positive_rate"] == 0
    assert strict["false_negative_rate"] == pytest.approx(0.5)
    assert permissive["support"] == strict["support"] == 4


@pytest.mark.parametrize("invalid", [None, True, "0.5", -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_threshold_rejected_before_loading_data(monkeypatch, invalid):
    monkeypatch.setattr(train.examples, "load",
                        lambda: pytest.fail("invalid threshold must fail before loading data"))
    with pytest.raises(ValueError, match="threshold"):
        train.evaluate(threshold=invalid)
