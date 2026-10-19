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
]
_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in HEURISTIC_PATTERNS]


def heuristic_score(text: str) -> float:
    """Fraction-based score in [0,1] from how many injection patterns match."""
    hits = sum(1 for p in _COMPILED if p.search(text))
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
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        self.model.fit(texts, labels)
        return self

    def ml_score(self, text: str) -> float:
        if self.model is None:
            raise RuntimeError("call fit() before scoring")
        return float(self.model.predict_proba([text])[0, 1])

    def score(self, text: str) -> float:
        """Blend heuristic and ML scores. Heuristics can only *raise* suspicion."""
        h = heuristic_score(text)
        m = self.ml_score(text) if self.model is not None else 0.0
        blended = self.ml_weight * m + (1 - self.ml_weight) * h
        # A confident heuristic hit shouldn't be diluted below a floor.
        return max(blended, h * 0.85)

    def is_injection(self, text: str, threshold: float = 0.5) -> bool:
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a finite number between 0 and 1")
        return self.score(text) >= threshold
