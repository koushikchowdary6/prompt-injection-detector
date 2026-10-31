"""Regression coverage for fold-level held-out guardrail diagnostics."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train


def test_fold_diagnostics_expose_concentrated_false_alarms(monkeypatch):
    texts = ["attack-a", "benign-a", "benign-b", "attack-b", "benign-c", "attack-c"]
    labels = [1, 0, 0, 1, 0, 1]
    scores = dict(zip(texts, [0.9, 0.8, 0.1, 0.2, 0.3, 0.9]))
    monkeypatch.setattr(train.examples, "load", lambda: (texts, labels))
    monkeypatch.setattr(train, "grouped_folds", lambda *a, **kw: [
        (np.array([2, 3, 4, 5]), np.array([0, 1])),
        (np.array([0, 1]), np.array([2, 3, 4, 5])),
    ])
    calls = {"fit": 0, "score": 0}

    class StubDetector:
        def __init__(self, ml_weight):
            pass

        def fit(self, texts, labels):
            calls["fit"] += 1
            return self

        def score(self, text):
            calls["score"] += 1
            return scores[text]

    monkeypatch.setattr(train, "PromptInjectionDetector", StubDetector)
    reports = train.evaluate_thresholds((0.5, 0.85))
    assert calls == {"fit": 2, "score": 6}

    first, second = reports[0]["fold_diagnostics"]
    assert first["confusion_matrix"] == {"tp": 1, "fp": 1, "tn": 0, "fn": 0}
    assert first["false_positive_rate"] == 1.0
    assert second["confusion_matrix"] == {"tp": 1, "fp": 0, "tn": 2, "fn": 1}
    assert second["false_positive_rate"] == 0.0
    assert second["false_negative_rate"] == 0.5
    assert [fold["support"] for fold in reports[0]["fold_diagnostics"]] == [2, 4]
    assert reports[0]["confusion_matrix"] == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}
    assert reports[1]["confusion_matrix"] == {"tp": 2, "fp": 0, "tn": 3, "fn": 1}
    for report in reports:
        assert sum(fold["support"] for fold in report["fold_diagnostics"]) == report["support"]
        for cell, count in report["confusion_matrix"].items():
            assert sum(fold["confusion_matrix"][cell] for fold in report["fold_diagnostics"]) == count
    json.dumps(reports, allow_nan=False)


def test_cli_json_includes_reconcilable_per_fold_metrics():
    script = Path(__file__).resolve().parents[1] / "src" / "train.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--json", "--threshold", "0.5"],
        capture_output=True, text=True, check=True, timeout=60,
    )
    report = json.loads(completed.stdout)
    assert report["folds"] == 5
    for result in report["results"]:
        folds = result["fold_diagnostics"]
        assert len(folds) == report["folds"]
        assert [fold["fold"] for fold in folds] == list(range(1, report["folds"] + 1))
        assert sum(fold["support"] for fold in folds) == result["support"]
        for key in ("tp", "fp", "tn", "fn"):
            assert sum(fold["confusion_matrix"][key] for fold in folds) == result["confusion_matrix"][key]
