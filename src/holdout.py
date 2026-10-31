"""Evaluate a frozen detector on a separately supplied, labeled JSONL holdout.

This tool never tunes a threshold on holdout labels and refuses examples whose
TF-IDF feature multisets overlap the training corpus. Independence of data
collection must still be established by the experimenter.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import sys
from numbers import Integral

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from sklearn.feature_extraction.text import TfidfVectorizer

from detector import PromptInjectionDetector, normalize_text
from train import dataset_provenance, matthews_correlation, wilson_interval
import examples

MAX_HOLDOUT_BYTES = 2_000_000
MAX_REPORTED_ERROR_ROWS_PER_CLASS = 20
MAX_HOLDOUT_CATEGORIES = 20
_CATEGORY = re.compile(r"[a-z][a-z0-9_-]{0,39}\\Z")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_holdout(path: str | Path, *, with_categories=False):
    """Read bounded JSONL; optionally return validated tactic category labels."""
    # A holdout is untrusted input: reject symlinks and special files before
    # opening them. In particular, opening a FIFO without O_NONBLOCK can hang
    # the evaluator indefinitely, while following a symlink can read a file
    # outside the dataset selected by the experimenter.
    source = Path(path)
    metadata = source.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("holdout must be a regular file, not a symlink or special file")
    if metadata.st_size > MAX_HOLDOUT_BYTES:
        raise ValueError("holdout exceeds 2,000,000 bytes")
    # Recheck the *opened descriptor* to avoid a path-swap between lstat and
    # open. O_NOFOLLOW prevents a replacement symlink on supporting systems;
    # O_NONBLOCK prevents a swapped FIFO from blocking before fstat rejects it.
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(source, flags), "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("holdout must be a regular file")
        raw = handle.read(MAX_HOLDOUT_BYTES + 1)
    if len(raw) > MAX_HOLDOUT_BYTES:
        raise ValueError("holdout exceeds 2,000,000 bytes")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("holdout must be UTF-8") from exc
    # JSONL records are delimited by LF, not by every Unicode line-break
    # character. splitlines() corrupts valid JSON strings containing U+2028,
    # U+2029, or U+0085 and can silently change the evaluation sample count.
    lines = decoded.split("\n")
    if lines[-1] == "":
        lines.pop()  # One conventional terminal LF is not an empty record.
    texts, labels, categories = [], [], []
    category_mode = None
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"holdout line {number}: empty row")
        try:
            row = json.loads(
                line, object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {value}")
                ),
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"holdout line {number}: invalid JSON: {exc}") from exc
        if (not isinstance(row, dict)
                or set(row) not in ({"text", "label"}, {"text", "label", "category"})
                or not isinstance(row["text"], str)
                or not row["text"].strip()
                or type(row["label"]) is not int or row["label"] not in (0, 1)):
            raise ValueError(f"holdout line {number}: expected text and binary integer label")
        has_category = "category" in row
        if category_mode is None:
            category_mode = has_category
        elif category_mode != has_category:
            raise ValueError("holdout categories must be supplied for every row or none")
        if has_category:
            category = row["category"]
            if not isinstance(category, str) or not _CATEGORY.fullmatch(category):
                raise ValueError(f"holdout line {number}: invalid category slug")
            categories.append(category)
            if len(set(categories)) > MAX_HOLDOUT_CATEGORIES:
                raise ValueError("holdout exceeds category count limit")
        texts.append(row["text"])
        labels.append(row["label"])
    if set(labels) != {0, 1}:
        raise ValueError("holdout must include both benign and injection examples")
    if with_categories:
        return texts, labels, (categories if category_mode else None)
    return texts, labels


def _feature_keys(texts):
    # Match the actual vectorizer feature representation used in fit() and CV.
    analyzer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, sublinear_tf=True
    ).build_analyzer()
    keys = []
    for text in texts:
        key = tuple(sorted(analyzer(normalize_text(text))))
        if not key:
            raise ValueError("holdout or training text has no TF-IDF features")
        keys.append(key)
    return keys


def evaluate_holdout(texts, labels, *, threshold=0.5, train_data=None, categories=None):
    """Fit on training examples only, score each disjoint holdout item once."""
    if (type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not 0 <= threshold <= 1):
        raise ValueError("threshold must be a finite number in [0, 1]")
    if (len(texts) != len(labels) or not texts or
            any(type(label) is not int or label not in (0, 1) for label in labels)
            or set(labels) != {0, 1}):
        raise ValueError("holdout requires aligned examples from both classes")
    if categories is not None:
        if (len(categories) != len(texts)
                or any(not isinstance(category, str)
                       or not _CATEGORY.fullmatch(category) for category in categories)
                or len(set(categories)) > MAX_HOLDOUT_CATEGORIES):
            raise ValueError("categories must be aligned, valid slugs with at most 20 groups")
    training_texts, training_labels = train_data if train_data is not None else examples.load()
    if (len(training_texts) != len(training_labels) or not training_texts
            or any(isinstance(label, bool) or not isinstance(label, Integral)
                   or label not in (0, 1) for label in training_labels)
            or set(training_labels) != {0, 1}):
        raise ValueError("training examples must be aligned and contain both binary classes")
    # The training set is part of the evaluation contract, too. Opposite
    # labels on feature-equivalent prompts make the fitted model and reported
    # holdout scores ambiguous, even if the holdout itself is disjoint.
    training_keys = _feature_keys(training_texts)
    observed_labels = {}
    for key, label in zip(training_keys, training_labels):
        if key in observed_labels and observed_labels[key] != label:
            raise ValueError("training labels conflict for feature-equivalent prompts")
        observed_labels[key] = label
    holdout_keys = _feature_keys(texts)
    if len(set(holdout_keys)) != len(holdout_keys):
        raise ValueError("holdout contains feature-equivalent duplicate prompts")
    if set(training_keys).intersection(holdout_keys):
        raise ValueError("holdout overlaps training data under TF-IDF normalization")

    detector = PromptInjectionDetector().fit(training_texts, training_labels)
    tp = fp = tn = fn = 0
    false_positive_rows, false_negative_rows = [], []
    category_cells = (
        {category: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
         for category in sorted(set(categories))}
        if categories is not None else None
    )
    for row_number, (text, actual) in enumerate(zip(texts, labels), 1):
        raw_prediction = detector.is_injection(text, threshold=threshold)
        # Never coerce arbitrary numeric/string values to 0 or 1: an
        # incorrect adapter could otherwise fabricate a plausible confusion
        # matrix (for example int(0.9) silently becomes a negative).
        if not isinstance(raw_prediction, (bool, np.bool_)):
            raise ValueError("detector must return boolean holdout decisions")
        prediction = int(raw_prediction)
        tp += int(prediction == 1 and actual == 1)
        fp += int(prediction == 1 and actual == 0)
        tn += int(prediction == 0 and actual == 0)
        fn += int(prediction == 0 and actual == 1)
        if category_cells is not None:
            cell = ("tp" if prediction == 1 and actual == 1
                    else "fp" if prediction == 1 else
                    "fn" if actual == 1 else "tn")
            category_cells[categories[row_number - 1]][cell] += 1
        # Refer back to the supplied JSONL without publishing prompt text.
        # Bound the error examples in large holdouts; totals remain exact.
        if prediction == 1 and actual == 0:
            if len(false_positive_rows) < MAX_REPORTED_ERROR_ROWS_PER_CLASS:
                false_positive_rows.append(row_number)
        elif prediction == 0 and actual == 1:
            if len(false_negative_rows) < MAX_REPORTED_ERROR_ROWS_PER_CLASS:
                false_negative_rows.append(row_number)
    category_breakdown = None
    if category_cells is not None:
        category_breakdown = {
            "scope": "user_supplied_categories_exploratory_not_independent",
            "groups": {},
        }
        for category, counts in category_cells.items():
            category_benign = counts["fp"] + counts["tn"]
            category_attacks = counts["tp"] + counts["fn"]
            category_breakdown["groups"][category] = {
                "support": category_benign + category_attacks,
                "confusion_matrix": counts,
                "false_positive_rate": (
                    counts["fp"] / category_benign if category_benign else None
                ),
                "false_negative_rate": (
                    counts["fn"] / category_attacks if category_attacks else None
                ),
            }
    benign, attacks = fp + tn, tp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / attacks
    return {
        "evaluation": "user_supplied_disjoint_holdout",
        "independent_collection_verified": False,
        "threshold_preselection_verified": False,
        "risk_scores_calibrated": False,
        "threshold": threshold,
        "training": dataset_provenance(training_texts, training_labels),
        "holdout": dataset_provenance(texts, labels),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "category_breakdown": category_breakdown,
        "error_audit": {
            "row_numbering": "one_based_holdout_input_order",
            "max_rows_per_class": MAX_REPORTED_ERROR_ROWS_PER_CLASS,
            "false_positive_rows": false_positive_rows,
            "false_negative_rows": false_negative_rows,
            "false_positive_omitted": fp - len(false_positive_rows),
            "false_negative_omitted": fn - len(false_negative_rows),
            "raw_prompt_text_included": False,
        },
        "support": len(texts),
        "precision": precision,
        "precision_ci95": wilson_interval(tp, tp + fp),
        "mcc": matthews_correlation(tp, fp, tn, fn),
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "balanced_accuracy": (recall + tn / benign) / 2,
        "false_positive_rate": fp / benign,
        "false_negative_rate": fn / attacks,
        "false_positive_rate_ci95": wilson_interval(fp, benign),
        "false_negative_rate_ci95": wilson_interval(fn, attacks),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate on external labeled JSONL without training/holdout overlap"
    )
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        texts, labels, categories = load_holdout(args.holdout, with_categories=True)
        report = evaluate_holdout(
            texts, labels, threshold=args.threshold, categories=categories
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
