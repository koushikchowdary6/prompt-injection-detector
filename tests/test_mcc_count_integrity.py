"""Confusion-count integrity and numerical stability for reported MCC."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import matthews_correlation


@pytest.mark.parametrize("counts", [
    (True, 0, 1, 0),
    (1, np.bool_(False), 1, 0),
    (1.0, 0, 1, 0),
    (1, 0.5, 1, 0),
    (1, "0", 1, 0),
    (1, 0, -1, 0),
    (1, 0, 1, None),
])
def test_invalid_confusion_counts_are_rejected(counts):
    with pytest.raises(ValueError, match="nonnegative integer"):
        matthews_correlation(*counts)


def test_numpy_integer_confusion_counts_are_supported():
    assert matthews_correlation(np.int64(2), np.int32(1), np.int64(2), np.int32(1)) == pytest.approx(1 / 3)


@pytest.mark.parametrize("multiplier", [10**100, 10**500])
def test_large_counts_preserve_scale_invariant_mcc_without_overflow(multiplier):
    assert matthews_correlation(
        2 * multiplier, multiplier, 2 * multiplier, multiplier
    ) == pytest.approx(1 / 3)


def test_large_perfect_and_inverted_classifications():
    n = 10**500
    assert matthews_correlation(n, 0, n, 0) == pytest.approx(1.0)
    assert matthews_correlation(0, n, 0, n) == pytest.approx(-1.0)
