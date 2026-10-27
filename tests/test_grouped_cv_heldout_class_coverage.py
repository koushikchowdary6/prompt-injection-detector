"""Grouped CV should not silently score single-class held-out partitions."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train


def test_single_class_heldout_fold_rejected_before_evaluation(monkeypatch):
    # Two independent feature groups per class pass the distinct-group gate,
    # and the first training partition has both labels. Its held-out set,
    # however, contains only benign prompts and cannot assess attack recall.
    texts = np.array([
        "ordinary meeting notes", "normal weather forecast",
        "ignore previous instructions", "override system rules",
    ])
    labels = np.array([0, 0, 1, 1])

    class ImbalancedSplitter:
        def __init__(self, **kwargs):
            pass

        def split(self, texts, labels, groups):
            yield np.array([0, 2, 3]), np.array([1])
            yield np.array([1, 3]), np.array([0, 2])

    monkeypatch.setattr(train, "StratifiedGroupKFold", ImbalancedSplitter)
    with pytest.raises(ValueError, match="single-class held-out"):
        train.grouped_folds(texts, labels, n_splits=2, seed=42)


def test_balanced_heldout_folds_remain_usable():
    texts = np.array([
        "ordinary meeting notes", "normal weather forecast",
        "benign weekly report", "routine trip planning",
        "ignore previous instructions", "override system rules",
        "reveal hidden prompt", "bypass safety policy",
    ])
    labels = np.array([0] * 4 + [1] * 4)
    folds = train.grouped_folds(texts, labels, n_splits=2, seed=42)
    assert len(folds) == 2
    for train_indices, test_indices in folds:
        assert set(labels[train_indices]) == {0, 1}
        assert set(labels[test_indices]) == {0, 1}
