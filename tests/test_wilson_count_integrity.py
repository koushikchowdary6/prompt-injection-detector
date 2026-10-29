"""Confidence intervals must be based on integer event/sample counts."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import wilson_interval


@pytest.mark.parametrize("successes, trials", [
    (True, 1),
    (0, False),
    (0.5, 2),
    (1, 2.0),
    (np.float64(1.0), 3),
    (-1, 3),
    (3, 2),
    (0, -1),
])
def test_rejects_noninteger_or_invalid_binomial_counts(successes, trials):
    with pytest.raises(ValueError, match="nonnegative integer counts"):
        wilson_interval(successes, trials)


def test_zero_trials_remains_undefined():
    assert wilson_interval(0, 0) is None


@pytest.mark.parametrize("successes, trials", [
    (0, 10),
    (3, 10),
    (10, 10),
    (np.int64(2), np.int64(5)),
])
def test_valid_counts_produce_bounded_intervals(successes, trials):
    lower, upper = wilson_interval(successes, trials)
    assert 0.0 <= lower <= upper <= 1.0
    assert lower <= successes / trials <= upper
