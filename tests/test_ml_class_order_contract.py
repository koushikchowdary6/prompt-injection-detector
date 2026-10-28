"""Do not interpret a classifier's probability column as the wrong class."""

import numpy as np
import pytest

from src.detector import PromptInjectionDetector


class ClassifierStub:
    def __init__(self, classes):
        self.classes_ = np.asarray(classes)
        self.calls = 0

    def predict_proba(self, texts):
        self.calls += 1
        return np.array([[0.1, 0.9]])


@pytest.mark.parametrize("classes", [
    [1, 0],  # Reversed classes would invert the security decision.
    [0, 2],  # A non-injection label in the positive column.
    [0], [0, 1, 2],
    ["0", "1"],  # String labels are not the trained binary contract.
    [False, True],  # Booleans must not silently substitute for labels.
])
def test_invalid_class_order_fails_closed_before_predict_proba(classes):
    detector = PromptInjectionDetector()
    model = ClassifierStub(classes)
    detector.model = model
    for method in (
        detector.ml_score, detector.score, detector.is_injection,
    ):
        with pytest.raises(ValueError, match="classes must be ordered"):
            method("Summarize the project plan")
    assert model.calls == 0


def test_expected_class_order_preserves_injection_probability():
    detector = PromptInjectionDetector()
    model = ClassifierStub([0, 1])
    detector.model = model
    assert detector.ml_score("Summarize the project plan") == pytest.approx(0.9)
    assert detector.is_injection("Summarize the project plan")
    assert model.calls == 2
