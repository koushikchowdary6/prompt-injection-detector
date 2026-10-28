"""Broken ML probability outputs must never silently classify an attack as benign."""

import numpy as np
import pytest

from src.detector import PromptInjectionDetector


class StubModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = 0

    def predict_proba(self, samples):
        assert len(samples) == 1
        self.calls += 1
        return self.probabilities


@pytest.mark.parametrize("probabilities", [
    [[float("nan"), float("nan")]],
    [[float("inf"), 0.0]],
    [[0.8, -0.2]],
    [[0.1, 1.1]],
    [[0.6, 0.6]],
    [[0.5]],
    [[0.5, 0.5], [0.5, 0.5]],
    [["invalid", 0.5]],
])
def test_invalid_model_output_fails_closed_at_all_scoring_entrypoints(probabilities):
    detector = PromptInjectionDetector()
    detector.model = StubModel(probabilities)
    text = "Summarize the project plan"  # no heuristic match
    for method in (
        lambda: detector.ml_score(text),
        lambda: detector.score(text),
        lambda: detector.is_injection(text),
    ):
        with pytest.raises(ValueError, match="invalid class probabilities"):
            method()


@pytest.mark.parametrize("probability", [0.0, 0.3, 1.0])
def test_valid_probabilities_remain_bounded(probability):
    detector = PromptInjectionDetector()
    detector.model = StubModel(np.array([[1.0 - probability, probability]]))
    assert detector.ml_score("Summarize the project plan") == pytest.approx(probability)
    assert 0.0 <= detector.score("Summarize the project plan") <= 1.0


def test_explicit_zero_ml_weight_skips_invalid_model_output():
    detector = PromptInjectionDetector(ml_weight=0.0)
    model = StubModel([[float("nan"), float("nan")]])
    detector.model = model
    assert detector.score("Summarize the project plan") == 0.0
    assert not detector.is_injection("Summarize the project plan")
    assert model.calls == 0
