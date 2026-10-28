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

import html
import math
import re
from itertools import product
import unicodedata
from numbers import Integral
from urllib.parse import unquote

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
# Worst-case total characters passed to regex scans across all separator
# interpretations. An input can be below MAX_INPUT_CHARS yet expand to
# 256 full-length variants, multiplying CPU work on untrusted prompts.
MAX_REGEX_SCAN_CHARS = 2_000_000


def _bounded_nfkc(text: str) -> str:
    """Reject oversized inputs before and after Unicode compatibility expansion."""
    if not isinstance(text, str):
        raise ValueError("input must be a string")
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(f"input exceeds {MAX_INPUT_CHARS} characters")
    # Untrusted retrieved HTML may encode attack instructions as character
    # references (e.g. &#73;gnore or &lt;|im_start|&gt;). Decode once,
    # consistently for heuristic and ML paths, before Unicode normalization.
    # Keep pre/post bounds so decoding and NFKC cannot expand inputs unchecked.
    # Retrieved URLs and form bodies may percent-encode instructions or
    # privileged role delimiters. Decode exactly once, after HTML entities,
    # for consistent heuristic/ML features. Reject malformed UTF-8 rather
    # than silently replacing bytes and changing the security decision.
    try:
        decoded = unquote(html.unescape(text), encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("input contains invalid UTF-8 percent encoding") from exc
    if len(decoded) > MAX_INPUT_CHARS:
        raise ValueError(f"decoded input exceeds {MAX_INPUT_CHARS} characters")
    normalized = unicodedata.normalize("NFKC", decoded)
    if len(normalized) > MAX_INPUT_CHARS:
        raise ValueError(f"normalized input exceeds {MAX_INPUT_CHARS} characters")
    return normalized


# Common Cyrillic/Greek glyphs that visually impersonate ASCII letters.
# Fold only within mixed-script words: native-language text stays intact.
_CONFUSABLES = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "х": "x", "у": "y", "і": "i", "ј": "j", "ѕ": "s",
    "ο": "o", "ι": "i", "ν": "v", "α": "a", "ε": "e",
    "ρ": "p", "τ": "t", "κ": "k", "χ": "x",
}
_CONFUSABLES.update({key.upper(): value.upper() for key, value in list(_CONFUSABLES.items())})


def _fold_mixed_script_confusables(text: str) -> str:
    """Canonicalize lookalikes only in words mixing ASCII and mapped glyphs.

    A native Cyrillic or Greek word should not be transliterated or treated
    as an attack; a spoofed ASCII keyword such as ignоre should be detected.
    This is deliberately a limited confusable map, not full UTS #39 support.
    """
    def fold(match):
        word = match.group()
        if not re.search(r"[A-Za-z]", word):
            return word
        return "".join(_CONFUSABLES.get(char, char) for char in word)

    return re.sub(r"\w+", fold, text)


def _is_variation_selector(char: str) -> bool:
    """Unicode variation selectors are invisible inside ordinary words."""
    return "\ufe00" <= char <= "\ufe0f" or "\U000e0100" <= char <= "\U000e01ef"


def _is_hidden_separator(char: str) -> bool:
    """Invisible format/control/variation bytes can split attack keywords.

    Preserve ordinary whitespace (including newlines and tabs), which is
    meaningful for both prompt content and multiline heuristic patterns.
    """
    category = unicodedata.category(char)
    return (
        category == "Cf"
        or (category == "Cc" and not char.isspace())
        or char == "\u034f"  # Combining grapheme joiner (category Mn)
        or _is_variation_selector(char)
    )


def normalize_text(text: str) -> str:
    """Canonicalize invisible Unicode and control-byte obfuscations for ML."""
    normalized = _bounded_nfkc(text)
    return _fold_mixed_script_confusables("".join(ch for ch in normalized if not _is_hidden_separator(ch)))


def heuristic_score(text: str) -> float:
    """Fraction-based score in [0,1] from how many injection patterns match."""
    # Invisible format characters and non-whitespace control bytes can
    # split keywords or replace boundaries ("ig\\u200bnore", "ig\\x00nore").
    # Each character has two plausible interpretations: removed or space.
    # Explore a bounded number of combinations so mixed obfuscation cannot
    # evade both all-removed and all-spaced checks. Excessive hidden separators
    # are themselves a high-signal obfuscation indicator; avoid exponential
    # work on attacker-controlled input. ML normalization stays unchanged.
    # Fold lookalike letters *after* joining candidate segments. A Cyrillic
    # glyph isolated by invisible separators is not yet in a mixed-script
    # word; folding before separator removal misses that combined evasion.
    normalized = _bounded_nfkc(text)
    segments = [[]]
    for index, char in enumerate(normalized):
        # U+200D also joins emoji pictographs (e.g. woman-technologist).
        # These ordinary glyph sequences should not consume the eight-byte
        # obfuscation budget or be labeled suspicious by themselves. Only
        # exempt a joiner directly between two Unicode symbols; a joiner
        # splitting letters in an attack still receives full scrutiny.
        emoji_joiner = (
            char == "\u200d"
            and 0 < index < len(normalized) - 1
            and unicodedata.category(normalized[index - 1]) in {"So", "Sk"}
            and unicodedata.category(normalized[index + 1]) in {"So", "Sk"}
        )
        # Variation selectors are also legitimate emoji modifiers (including
        # keycap emoji). They become suspicious only outside that context,
        # such as when inserted into "ig\ufe0fnore".
        emoji_variation = (
            _is_variation_selector(char)
            and index > 0
            and (
                unicodedata.category(normalized[index - 1]) in {"So", "Sk"}
                or normalized[index - 1] in "#*0123456789"
            )
        )
        if _is_hidden_separator(char) and not (emoji_joiner or emoji_variation):
            segments.append([])
        else:
            segments[-1].append(char)
    pieces = ["".join(chars) for chars in segments]
    format_count = len(pieces) - 1
    if format_count > 8:
        return 0.7
    # Bound aggregate regex work, not only the number of variants. Eight
    # hidden separators in a 100k-character prompt would otherwise trigger
    # roughly 25 million characters of repeated regex scans *per pattern*.
    # When inspection is too expensive, conservatively flag obfuscation
    # rather than silently treating the text as benign.
    variant_count = 1 << format_count
    if (len(normalized) + format_count) * variant_count > MAX_REGEX_SCAN_CHARS:
        return 0.7
    candidates = (
        "".join(part + (separators[i] if i < format_count else "")
                for i, part in enumerate(pieces))
        for separators in product(("", " "), repeat=format_count)
    )
    # Materialize at most 256 variants; each rule is credited once even if
    # it matches several interpretations of the same untrusted input.
    variants = tuple(_fold_mixed_script_confusables(candidate) for candidate in candidates)
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
        if texts is None or labels is None or len(texts) == 0 or len(texts) != len(labels):
            raise ValueError("training texts and labels must be non-empty and aligned")
        # Normalize once per example. The previous validation pass decoded
        # every untrusted prompt and then repeated the same work before fit.
        # A list or NumPy array of texts should follow the same validation.
        normalized_texts = []
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("training texts must be non-empty strings")
            normalized = normalize_text(text)
            if not normalized.strip():
                raise ValueError("training texts must be non-empty strings")
            normalized_texts.append(normalized)
        if any(isinstance(label, bool) or not isinstance(label, Integral)
               or label not in (0, 1) for label in labels):
            raise ValueError("training labels must be integers 0 (benign) or 1 (injection)")
        if set(labels) != {0, 1}:
            raise ValueError("training data must contain both benign and injection labels")
        candidate = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        # Compare the *actual* features the vectorizer will train on,
        # including its lowercase preprocessing and n-gram generation.
        # Using str.casefold() with build_tokenizer() is not equivalent to
        # sklearn's default lower(): e.g. "straße" and "STRASSE" become
        # identical under casefold but remain distinct TF-IDF features.
        # Reject truly feature-identical contradictory labels without
        # discarding legitimate Unicode distinctions.
        analyzer = candidate.named_steps["tfidf"].build_analyzer()
        observed_labels = {}
        for text, label in zip(normalized_texts, labels):
            # TF-IDF ignores feature order; only n-gram counts matter.
            # Permuted text can produce the same vector and must not carry
            # contradictory training labels.
            features = tuple(sorted(analyzer(text)))
            if not features:
                raise ValueError("training text has no usable TF-IDF tokens")
            if features in observed_labels and observed_labels[features] != label:
                raise ValueError("training labels conflict for feature-equivalent prompts")
            observed_labels[features] = label
        candidate.fit(normalized_texts, labels)
        self.model = candidate
        return self

    def ml_score(self, text: str) -> float:
        if self.model is None:
            raise RuntimeError("call fit() before scoring")
        # A broken or tampered classifier must not produce NaN/Inf risk
        # scores: comparing NaN against a threshold silently returns False,
        # which would classify untrusted text as benign (fail open).
        # Preserve explicit input-validation errors for oversized prompts.
        normalized = normalize_text(text)
        try:
            probabilities = np.asarray(
                self.model.predict_proba([normalized]), dtype=float
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("model returned invalid class probabilities") from exc
        if (
            probabilities.shape != (1, 2)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
            or not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-6)
        ):
            raise ValueError("model returned invalid class probabilities")
        return float(probabilities[0, 1])

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
