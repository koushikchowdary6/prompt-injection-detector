"""Reject type-coerced ML probabilities before they influence security decisions."""

import numpy as np
import pytest

from src.detector import PromptInjectionDetector


class StubModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, texts):
        assert len(texts) == 1
        return self.probabilities


@pytest.mark.parametrize("probabilities", [
    [[False, True]],
    [[0.0, True]],
    [[False, 1.0]],
    [["0.1", "0.9"]],
    [[0.1, "0.9"]],
    np.array([[np.bool_(False), np.bool_(True)]], dtype=object),
    np.array([[0.1, np.bool_(True)]], dtype=object),
])
def test_coercible_but_invalid_probabilities_fail_closed(probabilities):
    detector = PromptInjectionDetector()
    detector.model = StubModel(probabilities)
    for method in (detector.ml_score, detector.score, detector.is_injection):
        with pytest.raises(ValueError, match="invalid class probabilities"):
            method("Summarize the project plan")


@pytest.mark.parametrize("probabilities", [
    [[0, 1]],
    np.array([[0.2, 0.8]], dtype=np.float32),
    np.array([[0.2, 0.8]], dtype=np.float64),
])
def test_real_numeric_probability_arrays_remain_supported(probabilities):
    detector = PromptInjectionDetector()
    detector.model = StubModel(probabilities)
    assert detector.ml_score("Summarize the project plan") == pytest.approx(
        float(probabilities[0][1])
    )
