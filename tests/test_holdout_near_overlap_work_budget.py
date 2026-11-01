"""Near-duplicate audits must fail closed instead of doing unbounded pairwise work."""

import pytest

from src import holdout


def test_cross_split_audit_rejects_quadratic_candidate_work(monkeypatch):
    monkeypatch.setattr(holdout, "MAX_NEAR_OVERLAP_PAIR_COMPARISONS", 2)
    training = [("shared", "alpha"), ("shared", "beta"), ("shared", "gamma")]
    with pytest.raises(ValueError, match="cross-split.*pair comparison budget"):
        holdout._near_overlap_audit(training, [("shared", "delta")])


def test_within_split_audit_rejects_quadratic_candidate_work(monkeypatch):
    monkeypatch.setattr(holdout, "MAX_NEAR_OVERLAP_PAIR_COMPARISONS", 2)
    keys = [("shared", "alpha"), ("shared", "beta"), ("shared", "gamma")]
    with pytest.raises(ValueError, match="within-training.*pair comparison budget"):
        holdout._internal_near_overlap_audit(keys, dataset_name="training")


def test_exact_budget_is_allowed_and_reports_complete_evidence(monkeypatch):
    monkeypatch.setattr(holdout, "MAX_NEAR_OVERLAP_PAIR_COMPARISONS", 3)
    keys = [("shared", "alpha"), ("shared", "beta"), ("shared", "gamma")]
    cross = holdout._near_overlap_audit(keys, [("shared", "delta")], threshold=0.2)
    within = holdout._internal_near_overlap_audit(keys, threshold=0.2)
    assert cross["near_overlap_count"] == 1
    assert within["near_overlap_count"] == 2
    assert cross["raw_prompt_text_included"] is False
    assert within["raw_prompt_text_included"] is False


def test_holdout_rejects_excess_work_before_model_fit(monkeypatch):
    monkeypatch.setattr(holdout, "MAX_NEAR_OVERLAP_PAIR_COMPARISONS", 0)

    class MustNotFit:
        def __init__(self, *args, **kwargs):
            pytest.fail("model must not fit after audit work budget is exceeded")

    monkeypatch.setattr(holdout, "PromptInjectionDetector", MustNotFit)
    training = (["shared benign phrase", "shared attack phrase"], [0, 1])
    holdout_data = (["unrelated benign sentence", "unrelated attack sentence"], [0, 1])
    with pytest.raises(ValueError, match="pair comparison budget"):
        holdout.evaluate_holdout(*holdout_data, train_data=training)
