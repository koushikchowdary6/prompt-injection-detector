"""The CLI JSON mode is parseable, reproducible and preserves metric caveats."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


TRAIN = Path(__file__).resolve().parents[1] / "src" / "train.py"


@pytest.mark.parametrize("args,expected_thresholds", [
    (["--json"], [0.5]),
    (["--json", "--thresholds", "0.3", "0.7"], [0.3, 0.7]),
])
def test_json_mode_reports_complete_pooled_cv_metrics(args, expected_thresholds):
    process = subprocess.run(
        [sys.executable, str(TRAIN), *args],
        capture_output=True, text=True, check=True, timeout=60,
    )
    data = json.loads(process.stdout)
    assert process.stderr == ""
    assert data["evaluation"] == "stratified_grouped_cross_validation"
    assert data["folds"] == 5
    assert data["threshold_selection"] == "exploratory_same_folds"
    assert data["risk_scores_calibrated"] is False
    assert [item["threshold"] for item in data["results"]] == expected_thresholds
    assert len({item["support"] for item in data["results"]}) == 1
    for result in data["results"]:
        matrix = result["confusion_matrix"]
        assert sum(matrix.values()) == result["support"]
        assert result["positive_support"] == matrix["tp"] + matrix["fn"]
        assert result["negative_support"] == matrix["tn"] + matrix["fp"]
        assert result["false_positive_rate_ci95"] is None or len(result["false_positive_rate_ci95"]) == 2
        assert result["false_negative_rate_ci95"] is None or len(result["false_negative_rate_ci95"]) == 2
    assert "Fresh examples:" not in process.stdout


def test_invalid_json_mode_threshold_fails_without_fake_metrics():
    process = subprocess.run(
        [sys.executable, str(TRAIN), "--json", "--threshold", "1.5"],
        capture_output=True, text=True, timeout=30,
    )
    assert process.returncode != 0
    assert process.stdout == ""
    assert "thresholds" in process.stderr
