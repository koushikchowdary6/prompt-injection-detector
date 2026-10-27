"""Threshold sweeps must reuse folds and report unbiased pooled counts."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train


def _stub_cv(monkeypatch):
    texts = ["attack_a", "benign_a", "attack_b", "benign_b"]
    labels = [1, 0, 1, 0]
    scores = dict(zip(texts, [0.9, 0.4, 0.6, 0.2]))
    monkeypatch.setattr(train.examples, "load", lambda: (texts, labels))
    monkeypatch.setattr(train, "grouped_folds", lambda *args, **kwargs: [
        (np.array([2, 3]), np.array([0, 1])),
        (np.array([0, 1]), np.array([2, 3])),
    ])
    calls = {"fits": 0, "scores": 0}

    class Detector:
        def __init__(self, ml_weight):
            pass

        def fit(self, texts, labels):
            calls["fits"] += 1
            return self

        def score(self, text):
            calls["scores"] += 1
            return scores[text]

    monkeypatch.setattr(train, "PromptInjectionDetector", Detector)
    return calls


def test_sweep_fits_once_per_fold_and_scores_each_holdout_once(monkeypatch):
    calls = _stub_cv(monkeypatch)
    result = train.evaluate_thresholds((0.3, 0.5, 0.7))
    assert calls == {"fits": 2, "scores": 4}
    assert [r["threshold"] for r in result] == [0.3, 0.5, 0.7]
    assert [r["confusion_matrix"] for r in result] == [
        {"tp": 2, "fp": 1, "tn": 1, "fn": 0},
        {"tp": 2, "fp": 0, "tn": 2, "fn": 0},
        {"tp": 1, "fp": 0, "tn": 2, "fn": 1},
    ]
    assert all(r["support"] == 4 for r in result)
    assert result[0]["false_positive_rate"] == 0.5
    assert result[2]["false_negative_rate"] == 0.5


@pytest.mark.parametrize("thresholds", [
    [], (), (True,), (None,), ("0.5",), (-0.1,), (1.1,),
    (float("nan"),), (float("inf"),), (0.5, 0.5), {0.5},
])
def test_invalid_sweep_fails_before_loading(monkeypatch, thresholds):
    monkeypatch.setattr(train.examples, "load",
                        lambda: pytest.fail("invalid thresholds must fail early"))
    with pytest.raises(ValueError, match="thresholds"):
        train.evaluate_thresholds(thresholds)


def test_sweep_rejects_overlapping_holdout_folds(monkeypatch):
    _stub_cv(monkeypatch)
    monkeypatch.setattr(train, "grouped_folds", lambda *a, **kw: [
        (np.array([2, 3]), np.array([0, 1])),
        (np.array([1, 3]), np.array([0, 2])),
    ])
    with pytest.raises(ValueError, match="overlap"):
        train.evaluate_thresholds((0.5, 0.7))


def test_sweep_rejects_missing_holdout_examples(monkeypatch):
    _stub_cv(monkeypatch)
    monkeypatch.setattr(train, "grouped_folds", lambda *a, **kw: [
        (np.array([2, 3]), np.array([0, 1])),
    ])
    with pytest.raises(ValueError, match="exactly once"):
        train.evaluate_thresholds((0.5,))


def test_sweep_rejects_invalid_model_scores(monkeypatch):
    _stub_cv(monkeypatch)

    class InvalidDetector:
        def __init__(self, ml_weight):
            pass

        def fit(self, texts, labels):
            return self

        def score(self, text):
            return float("nan")

    monkeypatch.setattr(train, "PromptInjectionDetector", InvalidDetector)
    with pytest.raises(ValueError, match="risk score"):
        train.evaluate_thresholds((0.5,))
