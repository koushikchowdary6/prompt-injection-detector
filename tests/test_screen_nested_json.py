"""Regression coverage for malformed, deeply nested JSONL guardrail input."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_nested_json_rejected_without_traceback_or_partial_decisions():
    # The first record is valid, but a later malicious record must invalidate
    # the whole batch rather than allowing earlier decisions to be consumed.
    nested = "[" * 1500 + "0" + "]" * 1500
    payload = '{"text":"Summarize this report"}\n' + '{"text":"safe","extra":' + nested + '}\n'
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "screen.py")],
        input=payload,
        text=True,
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "line 2: invalid JSON" in result.stderr
    assert "Traceback" not in result.stderr
    assert "safe" not in result.stderr
