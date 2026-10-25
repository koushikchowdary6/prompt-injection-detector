"""Hybrid prompt-injection detector: heuristic rules + ML classifier.

Two complementary layers:

  1. Heuristics -- fast, transparent regex/keyword rules for the best-known
     injection patterns ("ignore previous instructions", "reveal your system
     prompt", jailbreak persona names, etc.). Zero training, fully auditable.
  2. ML -- a TF-IDF + LogisticRegression model trained on labeled examples,
     which generalizes to phrasings the rules don't literally match.

`score()` returns a risk score (not a calibrated probability) by blending the two, so you get the rules'
precision on known attacks plus the model's recall on novel phrasings. This
is exactly the kind of guardrail an LLM application puts in front of user
input -- and it connects to my SigmaForge research on adversarial context.
"""
from __future__ import annotations

import math
import re
from itertools import product
import unicodedata
from numbers import Integral

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Transparent high-signal patterns. Each is a known injection tell.
HEURISTIC_PATTERNS = [
    r"\bignore\s+(?:(?:all|the)\s+)?(?:previous|prior|above)\b",
    r"\bdisregard\s+(?:(?:the|all)\s+)?(?:above|previous|prior)\b",
    r"\bforget\s+(?:everything|all|what\s+you)\b",
    r"(reveal|print|repeat|show).{0,30}(system|initial|hidden).{0,20}(prompt|instructions?|message|rules)",
    r"\bdo\s+anything\s+now\b|developer\s+mode|jailbreak|\bDAN\b.{0,24}\b(mode|persona|assistant|jailbreak)\b",
    r"\bno\s+(?:restrictions|rules|filter|guidelines|safety)\b",
    r"(bypass|override|disable).{0,20}(restrictions?|filters?|policy|guidelines|instructions?|safety)",
    r"pretend you are (a |an )?(different|unrestricted)",
    r"you are now.{0,40}\b(unrestricted|unfiltered|uncensored|developer mode|different assistant|dan)\b",
    r"\bend\s+of\s+prompt\b",
    r"do not follow.{0,20}(policy|guidelines|openai|anthropic)",
    r"what (were|are) the.{0,20}instructions (given|you)",
    # High-signal attempts to inject a privileged role boundary into
    # untrusted prompt text (ChatML and Llama 3 header conventions).
    r"(?:<\|im_start\|>|\[im_start\])\s*(?:system|developer)\b|\[start_header_id\]\s*(?:system|developer)\s*\[end_header_id\]",
]
_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in HEURISTIC_PATTERNS]

# Guardrail inputs are untrusted. In particular, up to 256 regex variants of
# a long prompt would otherwise multiply memory use and scanning time.
MAX_INPUT_CHARS = 100_000


def _bounded_nfkc(text: str) -> str:
    """Reject oversized inputs before and after Unicode compatibility expansion."""
    if not isinstance(text, str):
        raise ValueError("input must be a string")
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(f"input exceeds {MAX_INPUT_CHARS} characters")
    normalized = unicodedata.normalize("NFKC", text)
    if len(normalized) > MAX_INPUT_CHARS:
        raise ValueError(f"normalized input exceeds {MAX_INPUT_CHARS} characters")
    return normalized


def normalize_text(text: str) -> str:
    """Canonicalize common Unicode obfuscations for both detector layers."""
    normalized = _bounded_nfkc(text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf")


def heuristic_score(text: str) -> float:
    """Fraction-based score in [0,1] from how many injection patterns match."""
    # Invisible format characters can split keywords or replace word
    # boundaries, even within one phrase ("ig\\u200bnore\\u200bprevious").
    # Each character has two plausible interpretations: removed or space.
    # Explore a bounded number of combinations so mixed obfuscation cannot
    # evade both all-removed and all-spaced checks. Excessive format characters
    # are themselves a high-signal obfuscation indicator; avoid exponential
    # work on attacker-controlled input. ML normalization stays unchanged.
    normalized = _bounded_nfkc(text)
    segments = [[]]
    for char in normalized:
        if unicodedata.category(char) == "Cf":
            segments.append([])
        else:
            segments[-1].append(char)
    pieces = ["".join(chars) for chars in segments]
    format_count = len(pieces) - 1
    if format_count > 8:
        return 0.7
    candidates = (
        "".join(part + (separators[i] if i < format_count else "")
                for i, part in enumerate(pieces))
        for separators in product(("", " "), repeat=format_count)
    )
    # Materialize at most 256 variants; each rule is credited once even if
    # it matches several interpretations of the same untrusted input.
    variants = tuple(candidates)
    hits = sum(
        1 for pattern in _COMPILED if any(pattern.search(v) for v in variants)
    )
    if hits == 0:
        return 0.0
    # 1 hit -> 0.7, saturating toward 1.0 with more hits.
    return min(1.0, 0.7 + 0.15 * (hits - 1))


class PromptInjectionDetector:
    def __init__(self, ml_weight: float = 0.5):
        if not math.isfinite(ml_weight) or not 0.0 <= ml_weight <= 1.0:
            raise ValueError("ml_weight must be a finite number between 0 and 1")
        self.ml_weight = ml_weight
        self.model: Pipeline | None = None

    def fit(self, texts: list[str], labels: list[int]) -> "PromptInjectionDetector":
        """Fit a binary 0=benign, 1=injection classifier without corrupting an existing model.

        predict_proba()[0, 1] assumes class 1 is the injection label. Reject
        mislabeled, single-class, or mismatched training data before fitting.
        """
        if not texts or len(texts) != len(labels):
            raise ValueError("training texts and labels must be non-empty and aligned")
        if any(not isinstance(text, str) or not normalize_text(text).strip()
               for text in texts):
            raise ValueError("training texts must be non-empty strings")
        if any(isinstance(label, bool) or not isinstance(label, Integral)
               or label not in (0, 1) for label in labels):
            raise ValueError("training labels must be integers 0 (benign) or 1 (injection)")
        if set(labels) != {0, 1}:
            raise ValueError("training data must contain both benign and injection labels")
        candidate = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        normalized_texts = [normalize_text(text) for text in texts]
        # The TF-IDF model discards punctuation and one-character tokens.
        # Feature-identical prompts with contradictory labels make training
        # internally inconsistent, even when the raw strings differ.
        tokenizer = candidate.named_steps["tfidf"].build_tokenizer()
        observed_labels = {}
        for text, label in zip(normalized_texts, labels):
            tokens = tuple(tokenizer(text.casefold()))
            if not tokens:
                raise ValueError("training text has no usable TF-IDF tokens")
            if tokens in observed_labels and observed_labels[tokens] != label:
                raise ValueError("training labels conflict for feature-equivalent prompts")
            observed_labels[tokens] = label
        candidate.fit(normalized_texts, labels)
        self.model = candidate
        return self

    def ml_score(self, text: str) -> float:
        if self.model is None:
            raise RuntimeError("call fit() before scoring")
        return float(self.model.predict_proba([normalize_text(text)])[0, 1])

    def score(self, text: str) -> float:
        """Blend heuristic and ML scores. Heuristics can only *raise* suspicion."""
        h = heuristic_score(text)
        # A zero ML weight is an explicit opt-out, including the confidence floor.
        m = self.ml_score(text) if self.model is not None and self.ml_weight > 0 else 0.0
        blended = self.ml_weight * m + (1 - self.ml_weight) * h
        # Preserve high-confidence signals from either detector against blend dilution.
        return max(blended, h * 0.85, m * 0.85)

    def is_injection(self, text: str, threshold: float = 0.5) -> bool:
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a finite number between 0 and 1")
        return self.score(text) >= threshold
