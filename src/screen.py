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

from detector import PromptInjectionDetector


def screen_lines(lines, threshold: float):
    detector = PromptInjectionDetector(ml_weight=0.0)
    for index, line in enumerate(lines, start=1):
        if len(line) > 400_000:
            raise ValueError(f"line {index}: input record exceeds 400000 characters")
        try:
            record = json.loads(line)
        except (ValueError, TypeError) as exc:
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
    args = parser.parse_args(argv)
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("--threshold must be a finite number between 0 and 1")
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    try:
        for row in screen_lines(stdin, args.threshold):
            stdout.write(json.dumps(row, allow_nan=False) + "\n")
    except ValueError as exc:
        print(f"screening error: {exc}", file=stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
