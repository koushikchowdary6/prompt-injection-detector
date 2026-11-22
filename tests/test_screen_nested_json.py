"""Regression coverage for JSON parser recursion failures in guardrail batches."""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import screen  # noqa: E402


def test_json_recursion_error_rejected_without_partial_decisions(monkeypatch):
    # Force the parser's recursion failure deterministically across Python
    # versions and JSON decoder implementations.
    original_loads = screen.json.loads
    calls = 0

    def parse_with_nested_failure(value, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RecursionError("maximum nesting exceeded")
        return original_loads(value, **kwargs)

    monkeypatch.setattr(screen.json, "loads", parse_with_nested_failure)
    stdin = io.StringIO('{"text":"Summarize this report"}\n{"text":"safe"}\n')
    stdout, stderr = io.StringIO(), io.StringIO()
    result = screen.main([], stdin=stdin, stdout=stdout, stderr=stderr)
    assert result == 2
    assert stdout.getvalue() == ""
    assert "line 2: invalid JSON" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()
    assert "safe" not in stderr.getvalue()
