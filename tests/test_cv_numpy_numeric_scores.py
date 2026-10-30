"""CV adapters may emit NumPy real scalars without weakening score validation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train


def _run_with_score(monkeypatch, score):
    texts = ["attack_one", "benign_one", "attack_two", "benign_two"]
    labels = [1, 0, 1, 0]
    monkeypatch.setattr(train.examples, "load", lambda: (texts, labels))
    monkeypatch.setattr(train, "grouped_folds", lambda *args, **kwargs: [
        (np.array([2, 3]), np.array([0, 1])),
        (np.array([0, 1]), np.array([2, 3])),
    ])

    class NumericAdapter:
        def __init__(self, ml_weight):
            pass

        def fit(self, texts, labels):
            return self

        def score(self, text):
            return score if text.startswith("attack") else type(score)(0)

    monkeypatch.setattr(train, "PromptInjectionDetector", NumericAdapter)
    return train.evaluate_thresholds((0.5,))[0]


@pytest.mark.parametrize("score", [np.float32(0.75), np.float64(0.75), np.int64(1)])
def test_numpy_numeric_adapter_scores_are_counted(monkeypatch, score):
    summary = _run_with_score(monkeypatch, score)
    assert summary["confusion_matrix"] == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}
    assert summary["support"] == 4


@pytest.mark.parametrize("score", [np.bool_(True), np.float32(float("nan")),
                                   np.float32(float("inf")), np.float32(-0.1),
                                   np.float32(1.1), "0.75"])
def test_invalid_numpy_adapter_scores_fail_closed(monkeypatch, score):
    with pytest.raises(ValueError, match="invalid risk score"):
        _run_with_score(monkeypatch, score)


@pytest.mark.parametrize("score", [
    10**400,
    -(10**400),
    np.uint64(2**64 - 1),
    np.float64(np.finfo(np.float64).max),
])
def test_overflowing_or_out_of_range_adapter_scores_fail_as_validation_errors(
    monkeypatch, score
):
    # A very large Python integer raises OverflowError in math.isfinite.
    # A bad adapter must produce the same actionable validation failure as
    # NaN, booleans, and ordinary out-of-range values, not crash the run.
    with pytest.raises(ValueError, match="invalid risk score"):
        _run_with_score(monkeypatch, score)
