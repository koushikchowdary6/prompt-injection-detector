"""Class-imbalance and error-rate uncertainty in grouped held-out reports."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train


def test_wilson_interval_handles_zero_support_and_boundary_rates():
    assert train.wilson_interval(0, 0) is None
    assert train.wilson_interval(0, 10)[0] == pytest.approx(0.0)
    assert 0 < train.wilson_interval(0, 10)[1] < 1
    assert 0 < train.wilson_interval(10, 10)[0] < 1
    assert train.wilson_interval(10, 10)[1] == pytest.approx(1.0)
    low, high = train.wilson_interval(50, 100)
    assert low == pytest.approx(1 - high)
    assert low < 0.5 < high


@pytest.mark.parametrize("successes,trials", [(-1, 5), (6, 5), (1, -1)])
def test_wilson_interval_rejects_invalid_counts(successes, trials):
    with pytest.raises(ValueError):
        train.wilson_interval(successes, trials)


def test_balanced_accuracy_exposes_missed_attacks_in_majority_benign_data(monkeypatch):
    texts = ["attack_a", "attack_b", "benign_a", "benign_b", "benign_c", "benign_d"]
    labels = [1, 1, 0, 0, 0, 0]
    scores = dict(zip(texts, [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]))
    monkeypatch.setattr(train.examples, "load", lambda: (texts, labels))
    monkeypatch.setattr(train, "grouped_folds", lambda *a, **kw: [
        (np.array([1, 4, 5]), np.array([0, 2, 3])),
        (np.array([0, 2, 3]), np.array([1, 4, 5])),
    ])

    class Detector:
        def __init__(self, ml_weight):
            pass

        def fit(self, texts, labels):
            return self

        def score(self, text):
            return scores[text]

    monkeypatch.setattr(train, "PromptInjectionDetector", Detector)
    result = train.evaluate(threshold=0.5)

    assert result["confusion_matrix"] == {"tp": 1, "fp": 0, "tn": 4, "fn": 1}
    assert result["accuracy"] == pytest.approx(5 / 6)
    assert result["balanced_accuracy"] == pytest.approx(0.75)
    assert result["positive_support"] == 2
    assert result["negative_support"] == 4
    assert result["false_positive_rate"] == 0.0
    assert result["false_negative_rate"] == 0.5
    assert result["false_positive_rate_ci95"] == train.wilson_interval(0, 4)
    assert result["false_negative_rate_ci95"] == train.wilson_interval(1, 2)
