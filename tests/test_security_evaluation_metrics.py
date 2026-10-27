"""Security-facing metrics must use pooled held-out predictions and expose errors."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train


def _stub_evaluation(monkeypatch, labels, predicted):
    texts = [f"example_{i}" for i in range(len(labels))]
    monkeypatch.setattr(train.examples, "load", lambda: (texts, labels))
    # Deterministic held-out partitions; avoid fitting a real classifier in
    # tests focused on metric arithmetic rather than model behavior.
    mid = len(texts) // 2
    first = np.arange(mid)
    second = np.arange(mid, len(texts))
    monkeypatch.setattr(train, "grouped_folds", lambda *a, **kw: [
        (second, first), (first, second),
    ])
    decisions = dict(zip(texts, predicted))

    class StubDetector:
        def __init__(self, ml_weight):
            pass

        def fit(self, texts, labels):
            return self

        def is_injection(self, text, threshold=0.5):
            return decisions[text]

    monkeypatch.setattr(train, "PromptInjectionDetector", StubDetector)
    return train.evaluate()


def test_confusion_counts_f1_and_false_positive_rate(monkeypatch):
    # TP=1, FN=1, FP=1, TN=3. Benign false alarms matter just as much as
    # missed injections for a guardrail used on ordinary user requests.
    m = _stub_evaluation(monkeypatch,
                         [1, 1, 0, 0, 0, 0],
                         [True, False, True, False, False, False])
    assert m["confusion_matrix"] == {"tp": 1, "fp": 1, "tn": 3, "fn": 1}
    assert m["support"] == 6
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["false_positive_rate"] == pytest.approx(0.25)


def test_no_predicted_attacks_has_zero_f1_and_no_division_error(monkeypatch):
    m = _stub_evaluation(monkeypatch, [1, 0, 1, 0],
                         [False, False, False, False])
    assert m["confusion_matrix"] == {"tp": 0, "fp": 0, "tn": 2, "fn": 2}
    assert m["precision"] == 0
    assert m["recall"] == 0
    assert m["f1"] == 0
    assert m["false_positive_rate"] == 0


def test_fpr_is_undefined_without_benign_held_out_examples(monkeypatch):
    # This is a defensive metric contract for direct callers; the normal
    # grouped CV validator rejects single-class folds before evaluation.
    m = _stub_evaluation(monkeypatch, [1, 1, 1, 1],
                         [True, False, True, False])
    assert m["false_positive_rate"] is None
    assert m["confusion_matrix"] == {"tp": 2, "fp": 0, "tn": 0, "fn": 2}
