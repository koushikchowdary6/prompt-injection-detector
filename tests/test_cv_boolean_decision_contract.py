"""Decision-only CV adapters must return actual booleans, not truthy values."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train


def _stub_data(monkeypatch, decision):
    texts = ["attack_one", "benign_one", "attack_two", "benign_two"]
    labels = [1, 0, 1, 0]
    monkeypatch.setattr(train.examples, "load", lambda: (texts, labels))
    monkeypatch.setattr(train, "grouped_folds", lambda *a, **kw: [
        (np.array([2, 3]), np.array([0, 1])),
        (np.array([0, 1]), np.array([2, 3])),
    ])

    class DecisionOnlyDetector:
        def __init__(self, ml_weight):
            pass

        def fit(self, texts, labels):
            return self

        def is_injection(self, text, threshold=0.5):
            return decision(text)

    monkeypatch.setattr(train, "PromptInjectionDetector", DecisionOnlyDetector)


@pytest.mark.parametrize("bad", [1, 0, 2, "False", "True", None, np.int64(1), np.float64(0)])
def test_nonboolean_decisions_cannot_enter_cross_validation_metrics(monkeypatch, bad):
    _stub_data(monkeypatch, lambda text: bad)
    with pytest.raises(ValueError, match="invalid binary decision"):
        train.evaluate_thresholds((0.3, 0.5, 0.7))


@pytest.mark.parametrize("as_numpy_bool", [False, True])
def test_boolean_decisions_still_produce_correct_heldout_counts(monkeypatch, as_numpy_bool):
    def decide(text):
        value = text.startswith("attack")
        return np.bool_(value) if as_numpy_bool else value

    _stub_data(monkeypatch, decide)
    summaries = train.evaluate_thresholds((0.3, 0.5, 0.7))
    assert all(item["confusion_matrix"] == {
        "tp": 2, "fp": 0, "tn": 2, "fn": 0,
    } for item in summaries)
