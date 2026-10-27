"""Grouped CV must reject untrainable folds before starting model fits."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train


def test_single_class_training_fold_is_rejected_before_model_fit(monkeypatch):
    # Both classes have four independent feature groups, yet a grouped
    # splitter can still return an untrainable partition for imbalanced data.
    texts = np.array([
        "benign alpha", "benign beta", "benign gamma", "benign delta",
        "attack alpha", "attack beta", "attack gamma", "attack delta",
    ])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    class BadGroupedSplitter:
        def __init__(self, **kwargs):
            pass

        def split(self, texts, labels, groups):
            yield np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])
            yield np.array([4, 5, 6, 7]), np.array([0, 1, 2, 3])

    monkeypatch.setattr(train, "StratifiedGroupKFold", BadGroupedSplitter)

    with pytest.raises(ValueError, match="single-class training"):
        train.grouped_folds(texts, labels, n_splits=2, seed=42)
