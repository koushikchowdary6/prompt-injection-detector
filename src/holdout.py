"""Evaluate a frozen detector on a separately supplied, labeled JSONL holdout.

This tool never tunes a threshold on holdout labels and refuses examples whose
TF-IDF feature multisets overlap the training corpus. Independence of data
collection must still be established by the experimenter.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
# Repeated common features can turn even a bounded JSONL file into millions
# of Jaccard pair checks. Fail explicitly rather than publishing a partial audit.
MAX_NEAR_OVERLAP_PAIR_COMPARISONS = 1_000_000
_CATEGORY = re.compile(r"[a-z][a-z0-9_-]{0,39}")


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



def _near_overlap_audit(training_keys, holdout_keys, *, threshold=0.8,
                        training_labels=None, holdout_labels=None):
    """Audit lexical train/holdout overlap and possible label disagreement.

    The audit uses Jaccard similarity over normalized unigram/bigram feature
    *sets*, not semantic equivalence. Opposite-label pairs are reviewer-facing
    evidence, not proof of an annotation error or independent collection.
    Raw prompt text is never included in the report.
    """
    if (training_labels is None) != (holdout_labels is None):
        raise ValueError("both training and holdout labels are required for cross-label audit")
    if training_labels is not None and (
        len(training_labels) != len(training_keys)
        or len(holdout_labels) != len(holdout_keys)
    ):
        raise ValueError("near-overlap labels must align with feature rows")

    training_sets = [set(key) for key in training_keys]
    postings = {}
    for index, features in enumerate(training_sets):
        for feature in features:
            postings.setdefault(feature, set()).add(index)

    near_rows = []
    cross_label_rows = []
    cross_label_examples = []
    maximum_similarity = 0.0
    pair_comparisons = 0
    for row_number, key in enumerate(holdout_keys, 1):
        features = set(key)
        candidate_ids = set()
        for feature in features:
            candidate_ids.update(postings.get(feature, ()))

        best = 0.0
        best_cross_similarity, best_cross_index = 0.0, None
        for index in candidate_ids:
            pair_comparisons += 1
            if pair_comparisons > MAX_NEAR_OVERLAP_PAIR_COMPARISONS:
                raise ValueError("cross-split near-overlap audit exceeds pair comparison budget")
            similarity = len(features & training_sets[index]) / len(features | training_sets[index])
            best = max(best, similarity)
            if (training_labels is not None
                    and training_labels[index] != holdout_labels[row_number - 1]
                    and (similarity > best_cross_similarity
                         or (similarity == best_cross_similarity
                             and best_cross_index is not None and index < best_cross_index))):
                best_cross_similarity, best_cross_index = similarity, index

        maximum_similarity = max(maximum_similarity, best)
        if best >= threshold:
            near_rows.append(row_number)
        if best_cross_index is not None and best_cross_similarity >= threshold:
            cross_label_rows.append(row_number)
            if len(cross_label_examples) < MAX_REPORTED_ERROR_ROWS_PER_CLASS:
                cross_label_examples.append({
                    "holdout_row": row_number,
                    "training_row": best_cross_index + 1,
                    "similarity": best_cross_similarity,
                })
    return {
        "method": "jaccard_normalized_tfidf_unigram_bigram_feature_sets",
        "threshold": threshold,
        "near_overlap_count": len(near_rows),
        "near_overlap_rows": near_rows[:MAX_REPORTED_ERROR_ROWS_PER_CLASS],
        "near_overlap_rows_omitted": max(
            0, len(near_rows) - MAX_REPORTED_ERROR_ROWS_PER_CLASS
        ),
        "cross_label_audit_enabled": training_labels is not None,
        "cross_label_near_overlap_count": len(cross_label_rows),
        "cross_label_example_row_pairs": cross_label_examples,
        "cross_label_rows_omitted": max(
            0, len(cross_label_rows) - MAX_REPORTED_ERROR_ROWS_PER_CLASS
        ),
        "maximum_similarity": maximum_similarity,
        "raw_prompt_text_included": False,
        "independent_collection_verified": False,
        "note": "Heuristic lexical near-overlap and label disagreement; not proof of leakage, incorrect labels, or independent sampling.",
    }


def _internal_near_overlap_audit(holdout_keys, *, threshold=0.8, dataset_name="holdout",
                                 labels=None):
    """Find near-duplicate holdout rows without comparing a row to itself.

    Each row is compared only with earlier rows using the same normalized
    unigram/bigram feature sets as the training/holdout overlap check. A
    duplicate sample can make holdout uncertainty look artificially narrow
    even when there is no overlap with the training set.
    """
    if labels is not None and len(labels) != len(holdout_keys):
        raise ValueError("near-overlap labels must align with feature rows")
    postings = {}
    seen_sets = []
    near_rows = []
    examples = []
    cross_label_rows = []
    cross_label_examples = []
    maximum_similarity = 0.0
    pair_comparisons = 0
    for row_number, key in enumerate(holdout_keys, 1):
        features = set(key)
        candidates = set()
        for feature in features:
            candidates.update(postings.get(feature, ()))
        best_similarity, best_prior = 0.0, None
        cross_similarity, cross_prior = 0.0, None
        for index in candidates:
            pair_comparisons += 1
            if pair_comparisons > MAX_NEAR_OVERLAP_PAIR_COMPARISONS:
                raise ValueError(f"within-{dataset_name} near-overlap audit exceeds pair comparison budget")
            prior = seen_sets[index]
            similarity = len(features & prior) / len(features | prior)
            if (similarity > best_similarity
                    or (similarity == best_similarity
                        and best_prior is not None and index < best_prior)):
                best_similarity, best_prior = similarity, index
            # The most similar earlier row may share the current label while
            # another highly similar row disagrees. Search *all* candidates
            # rather than inferring label consistency from the nearest pair.
            if (labels is not None and labels[index] != labels[row_number - 1]
                    and (similarity > cross_similarity
                         or (similarity == cross_similarity
                             and cross_prior is not None and index < cross_prior))):
                cross_similarity, cross_prior = similarity, index
        maximum_similarity = max(maximum_similarity, best_similarity)
        if best_prior is not None and best_similarity >= threshold:
            near_rows.append(row_number)
            if len(examples) < MAX_REPORTED_ERROR_ROWS_PER_CLASS:
                examples.append({
                    "row": row_number,
                    "earlier_row": best_prior + 1,
                    "similarity": best_similarity,
                })
        if cross_prior is not None and cross_similarity >= threshold:
            cross_label_rows.append(row_number)
            if len(cross_label_examples) < MAX_REPORTED_ERROR_ROWS_PER_CLASS:
                cross_label_examples.append({
                    "row": row_number,
                    "earlier_row": cross_prior + 1,
                    "similarity": cross_similarity,
                })
        seen_sets.append(features)
        for feature in features:
            postings.setdefault(feature, set()).add(row_number - 1)
    return {
        "method": "jaccard_normalized_tfidf_unigram_bigram_feature_sets",
        "threshold": threshold,
        "near_overlap_count": len(near_rows),
        "near_overlap_rows": near_rows[:MAX_REPORTED_ERROR_ROWS_PER_CLASS],
        "near_overlap_rows_omitted": max(
            0, len(near_rows) - MAX_REPORTED_ERROR_ROWS_PER_CLASS
        ),
        "example_row_pairs": examples,
        "cross_label_audit_enabled": labels is not None,
        "cross_label_near_overlap_count": len(cross_label_rows),
        "cross_label_example_row_pairs": cross_label_examples,
        "cross_label_rows_omitted": max(
            0, len(cross_label_rows) - MAX_REPORTED_ERROR_ROWS_PER_CLASS
        ),
        "maximum_similarity": maximum_similarity,
        "raw_prompt_text_included": False,
        "note": f"Within-{dataset_name} lexical near-duplicates may inflate effective sample size; not proof of shared origin.",
    }


def evaluate_holdout(texts, labels, *, threshold=0.5, train_data=None, categories=None,
                     reject_near_overlap=False, reject_training_duplicates=False,
                     reject_training_near_overlap=False):
    """Fit on training examples only, score each disjoint holdout item once."""
    if (type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not 0 <= threshold <= 1):
        raise ValueError("threshold must be a finite number in [0, 1]")
    if (len(texts) != len(labels) or not texts or
            any(type(label) is not int or label not in (0, 1) for label in labels)
            or set(labels) != {0, 1}):
        raise ValueError("holdout requires aligned examples from both classes")
    if type(reject_near_overlap) is not bool:
        raise ValueError("reject_near_overlap must be a boolean")
    if type(reject_training_duplicates) is not bool:
        raise ValueError("reject_training_duplicates must be a boolean")
    if type(reject_training_near_overlap) is not bool:
        raise ValueError("reject_training_near_overlap must be a boolean")
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
    # Repeated feature-equivalent training rows silently give some examples
    # extra weight in TF-IDF/logistic-regression fitting. Expose this in the
    # evaluation report without disclosing raw training prompts. Equal-label
    # duplicates are permitted for compatibility; conflicting labels above
    # remain a hard error.
    training_counts = Counter(training_keys)
    duplicate_groups = [count for count in training_counts.values() if count > 1]
    training_duplicate_audit = {
        "method": "normalized_tfidf_unigram_bigram_feature_multisets",
        "total_rows": len(training_keys),
        "unique_feature_rows": len(training_counts),
        "duplicate_groups": len(duplicate_groups),
        "extra_duplicate_rows": sum(count - 1 for count in duplicate_groups),
        "largest_group_size": max(training_counts.values()),
        "conflicting_labels_rejected": True,
        "raw_prompt_text_included": False,
        "note": "Same-label feature-equivalent training rows can overweight examples; this is not an independent-sampling assessment.",
    }
    # Strict frozen-evaluation mode rejects repeated training examples before
    # model fitting, so duplicate weighting cannot distort the baseline.
    # Report-only is retained for backwards-compatible exploratory runs.
    if reject_training_duplicates and duplicate_groups:
        raise ValueError(
            f"training has {len(duplicate_groups)} feature-equivalent duplicate "
            "group(s); remove repeated rows before evaluation"
        )
    training_duplicate_audit["policy"] = (
        "reject_duplicates" if reject_training_duplicates else "report_only"
    )
    # Near-duplicate training rows can overweight lexical patterns even when
    # no two rows are feature-equivalent. Report the evidence separately from
    # exact duplicates, and optionally fail closed before model fitting.
    # This is a lexical screening heuristic, not proof of label leakage.
    training_near_overlap_audit = _internal_near_overlap_audit(
        training_keys, dataset_name="training", labels=training_labels
    )
    if reject_training_near_overlap and training_near_overlap_audit["near_overlap_count"]:
        raise ValueError(
            f"training has {training_near_overlap_audit['near_overlap_count']} "
            "near-duplicate example(s) at Jaccard >= 0.8"
        )
    training_near_overlap_audit["policy"] = (
        "reject_at_threshold" if reject_training_near_overlap else "report_only"
    )
    holdout_keys = _feature_keys(texts)
    if len(set(holdout_keys)) != len(holdout_keys):
        raise ValueError("holdout contains feature-equivalent duplicate prompts")
    if set(training_keys).intersection(holdout_keys):
        raise ValueError("holdout overlaps training data under TF-IDF normalization")
    overlap_audit = _near_overlap_audit(
        training_keys, holdout_keys,
        training_labels=training_labels, holdout_labels=labels,
    )
    internal_overlap_audit = _internal_near_overlap_audit(
        holdout_keys, labels=labels
    )
    # Report-only remains the default for exploratory analysis. For a frozen
    # external evaluation, callers can fail closed before model fitting when
    # a holdout example is lexically too similar to training examples.
    # This is a conservative screening gate, not a data-independence proof.
    if reject_near_overlap and overlap_audit["near_overlap_count"]:
        raise ValueError(
            f"holdout has {overlap_audit['near_overlap_count']} near-overlap "
            f"example(s) at Jaccard >= {overlap_audit['threshold']}"
        )
    if reject_near_overlap and internal_overlap_audit["near_overlap_count"]:
        raise ValueError(
            f"holdout has {internal_overlap_audit['near_overlap_count']} "
            f"internally near-duplicate example(s) at Jaccard >= "
            f"{internal_overlap_audit['threshold']}"
        )
    overlap_audit["policy"] = (
        "reject_at_threshold" if reject_near_overlap else "report_only"
    )
    internal_overlap_audit["policy"] = overlap_audit["policy"]

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
                "false_positive_rate_ci95": wilson_interval(
                    counts["fp"], category_benign
                ),
                "false_negative_rate": (
                    counts["fn"] / category_attacks if category_attacks else None
                ),
                "false_negative_rate_ci95": wilson_interval(
                    counts["fn"], category_attacks
                ),
            }
        # Macro rates give each category with the relevant class support
        # equal weight. Unlike pooled rates, these expose poor performance
        # in small attack families without treating absent classes as zero.
        # They remain exploratory because category definitions and samples
        # are supplied by the evaluator, not independently validated.
        group_rows = list(category_breakdown["groups"].values())
        fpr_rates = [row["false_positive_rate"] for row in group_rows
                     if row["false_positive_rate"] is not None]
        fnr_rates = [row["false_negative_rate"] for row in group_rows
                     if row["false_negative_rate"] is not None]
        category_breakdown["macro_error_rates"] = {
            "scope": "unweighted_categories_with_class_support_exploratory",
            "false_positive_rate": (
                sum(fpr_rates) / len(fpr_rates) if fpr_rates else None
            ),
            "false_positive_rate_eligible_categories": len(fpr_rates),
            "false_negative_rate": (
                sum(fnr_rates) / len(fnr_rates) if fnr_rates else None
            ),
            "false_negative_rate_eligible_categories": len(fnr_rates),
            "total_categories": len(group_rows),
            "note": "Unweighted macro rates are sensitive to small category samples; use category Wilson intervals and pooled rates alongside them.",
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
        "training_duplicate_audit": training_duplicate_audit,
        "training_near_overlap_audit": training_near_overlap_audit,
        "holdout": dataset_provenance(texts, labels),
        "feature_overlap_audit": overlap_audit,
        "holdout_internal_overlap_audit": internal_overlap_audit,
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
        "recall_ci95": wilson_interval(tp, attacks),
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
    parser.add_argument(
        "--reject-near-overlap", action="store_true",
        help="reject holdout rows with normalized TF-IDF Jaccard >= 0.8 before fitting",
    )
    parser.add_argument(
        "--reject-training-duplicates", action="store_true",
        help="reject feature-equivalent repeated training rows before model fitting",
    )
    parser.add_argument(
        "--reject-training-near-overlap", action="store_true",
        help="reject near-duplicate training rows (normalized feature Jaccard >= 0.8) before fitting",
    )
    args = parser.parse_args(argv)
    try:
        texts, labels, categories = load_holdout(args.holdout, with_categories=True)
        report = evaluate_holdout(
            texts, labels, threshold=args.threshold, categories=categories,
            reject_near_overlap=args.reject_near_overlap,
            reject_training_duplicates=args.reject_training_duplicates,
            reject_training_near_overlap=args.reject_training_near_overlap,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
