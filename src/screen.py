"""Offline JSONL guardrail CLI for batch prompt screening.

Usage: python src/screen.py --threshold 0.5 < prompts.jsonl
Input: one JSON object per line with a required string "text" field.
Output: one JSON object per line containing index, risk_score, flagged.
No input text is echoed in output or error messages.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile

from detector import PromptInjectionDetector


def _unique_object(pairs):
    """Reject ambiguous JSON records rather than silently keeping the last key."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value):
    """JSONL requires RFC 8259 numbers, not Python JSON NaN/Infinity extensions."""
    raise ValueError("non-finite JSON numeric constant")


def screen_lines(lines, threshold: float, max_records: int | None = None):
    detector = PromptInjectionDetector(ml_weight=0.0)
    # Bound each read before parsing: iterating over stdin lines can allocate an
    # arbitrarily large attacker-controlled line before the size check runs.
    # readline(size) is supported by standard text streams (including sys.stdin).
    index = 0
    while True:
        line = lines.readline(400_001)
        if not line:
            break
        index += 1
        if max_records is not None and index > max_records:
            raise ValueError(f"line {index}: batch exceeds --max-records limit")
        if len(line) > 400_000:
            raise ValueError(f"line {index}: input record exceeds 400000 characters")
        try:
            record = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        except (ValueError, TypeError, RecursionError) as exc:
            # A deeply nested JSON record can exceed the parser's recursion
            # limit before any schema validation runs. Fail closed without
            # exposing a traceback or emitting partial screening decisions.
            raise ValueError(f"line {index}: invalid JSON") from exc
        if not isinstance(record, dict) or type(record.get("text")) is not str:
            raise ValueError(f"line {index}: expected object with string text")
        try:
            risk = detector.score(record["text"])
        except ValueError as exc:
            raise ValueError(f"line {index}: rejected prompt input") from exc
        yield {"index": index, "risk_score": risk, "flagged": risk >= threshold}


def main(argv=None, stdin=None, stdout=None, stderr=None):
    parser = argparse.ArgumentParser(description="Screen JSONL prompts offline without echoing input")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-records", type=int, default=None,
                        help="Optional positive cap on JSONL records; reject oversized batches atomically")
    args = parser.parse_args(argv)
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("--threshold must be a finite number between 0 and 1")
    if args.max_records is not None and args.max_records < 1:
        parser.error("--max-records must be a positive integer")
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    try:
        # Stage decisions before emitting anything: an invalid later record must
        # not leave earlier results looking like a successful partial batch.
        with tempfile.SpooledTemporaryFile(mode="w+t", max_size=1024 * 1024, encoding="utf-8") as staged:
            for row in screen_lines(stdin, args.threshold, args.max_records):
                staged.write(json.dumps(row, allow_nan=False) + "\n")
            staged.seek(0)
            for chunk in iter(lambda: staged.read(65536), ""):
                stdout.write(chunk)
    except ValueError as exc:
        print(f"screening error: {exc}", file=stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
