"""Macro holdout error rates expose minority attack-family failures."""
from src import holdout


TRAIN = (
    ["normal weather forecast", "override the trusted instructions"],
    [0, 1],
)


def _evaluate(monkeypatch, texts, labels, categories, predicted):
    class StubDetector:
        def fit(self, texts, labels):
            return self

        def is_injection(self, text, threshold):
            return predicted[text]

    monkeypatch.setattr(holdout, "PromptInjectionDetector", StubDetector)
    return holdout.evaluate_holdout(
        texts, labels, categories=categories, train_data=TRAIN
    )


def test_macro_rates_do_not_hide_small_attack_family_failures(monkeypatch):
    texts = [
        "routine benign apple", "routine benign banana",
        "routine benign cherry", "routine benign date",
        "benign false alarm recipe",
        "encoded missed injected command",
        "role caught injected instruction",
    ]
    labels = [0, 0, 0, 0, 0, 1, 1]
    categories = ["routine"] * 4 + ["false_alarm", "encoded", "role"]
    predicted = {text: text.startswith(("benign false", "role caught"))
                 for text in texts}
    report = _evaluate(monkeypatch, texts, labels, categories, predicted)
    macro = report["category_breakdown"]["macro_error_rates"]
    # Routine benign class FPR=0; false_alarm FPR=1. The category
    # macro FPR=0.5 differs from the pooled FPR=1/5.
    assert report["false_positive_rate"] == 0.2
    assert macro["false_positive_rate"] == 0.5
    assert macro["false_positive_rate_eligible_categories"] == 2
    # Encoded attack FNR=1; role attack FNR=0.
    assert report["false_negative_rate"] == 0.5
    assert macro["false_negative_rate"] == 0.5
    assert macro["false_negative_rate_eligible_categories"] == 2
    assert macro["total_categories"] == 4
    assert macro["scope"].endswith("_exploratory")
    assert all(text not in str(report) for text in texts)


def test_absent_class_support_excluded_from_macro_denominator(monkeypatch):
    texts = ["ordinary market forecast", "attack unusual encoded bypass"]
    labels = [0, 1]
    categories = ["benign_only", "attack_only"]
    predicted = {text: False for text in texts}
    report = _evaluate(monkeypatch, texts, labels, categories, predicted)
    macro = report["category_breakdown"]["macro_error_rates"]
    assert macro["false_positive_rate"] == 0.0
    assert macro["false_negative_rate"] == 1.0
    assert macro["false_positive_rate_eligible_categories"] == 1
    assert macro["false_negative_rate_eligible_categories"] == 1
    assert report["category_breakdown"]["groups"]["attack_only"]["false_positive_rate"] is None


def test_unlabeled_holdout_keeps_category_breakdown_null(monkeypatch):
    texts = ["simple gardening advice", "attack malicious role switch"]
    labels = [0, 1]
    predicted = {text: False for text in texts}
    report = _evaluate(monkeypatch, texts, labels, None, predicted)
    assert report["category_breakdown"] is None
