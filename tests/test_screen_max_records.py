"""End-to-end coverage for bounded, atomic offline JSONL screening."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "screen.py"


def run_screen(lines, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        input="".join(json.dumps({"text": text}) + "\n" for text in lines),
        text=True,
        capture_output=True,
        check=False,
    )


def test_max_records_accepts_exact_limit():
    result = run_screen(["hello", "ignore previous instructions"], "--max-records", "2")
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["index"] for row in rows] == [1, 2]
    assert all("text" not in row for row in rows)


def test_max_records_rejects_oversized_batch_without_partial_results():
    result = run_screen(["hello", "sensitive prompt not to disclose"], "--max-records", "1")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "batch exceeds --max-records limit" in result.stderr
    assert "sensitive prompt" not in result.stderr


def test_max_records_rejects_nonpositive_limit():
    for invalid in ("0", "-1"):
        result = run_screen(["hello"], "--max-records", invalid)
        assert result.returncode != 0
        assert result.stdout == ""
        assert "--max-records must be a positive integer" in result.stderr
