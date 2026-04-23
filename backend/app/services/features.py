"""Feature extraction and weak-label rules shared by training and inference."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


URGENT_TERMS = {
    "urgent",
    "asap",
    "immediately",
    "emergency",
    "down",
    "outage",
    "broken",
    "crash",
    "crashed",
    "cannot",
    "can't",
    "refund",
    "cancel",
    "cancellation",
    "charged",
    "payment",
    "locked",
    "fraud",
    "stolen",
    "help",
}

HIGH_IMPACT_TERMS = {
    "account",
    "login",
    "password",
    "internet",
    "flight",
    "order",
    "delivery",
    "billing",
    "service",
    "phone",
}

NEGATIVE_TERMS = {
    "angry",
    "awful",
    "bad",
    "disappointed",
    "hate",
    "horrible",
    "mad",
    "poor",
    "sucks",
    "terrible",
    "unacceptable",
    "useless",
    "wrong",
}

LABELS = ("low", "normal", "high", "urgent")


@dataclass(frozen=True)
class FeatureVector:
    """Human-readable engineered feature values for one ticket."""

    char_count: int
    word_count: int
    exclamation_count: int
    question_count: int
    uppercase_ratio: float
    urgent_keyword_count: int
    high_impact_keyword_count: int
    negative_keyword_count: int
    has_url: int
    has_mention: int

    def as_array(self) -> np.ndarray:
        """Return the features as a 2D NumPy array for sklearn."""

        return np.array(
            [
                [
                    self.char_count,
                    self.word_count,
                    self.exclamation_count,
                    self.question_count,
                    self.uppercase_ratio,
                    self.urgent_keyword_count,
                    self.high_impact_keyword_count,
                    self.negative_keyword_count,
                    self.has_url,
                    self.has_mention,
                ],
            ],
            dtype=float,
        )


def normalize_text(text: str) -> str:
    """Normalize whitespace without destroying useful ticket wording."""

    return re.sub(r"\s+", " ", text.strip())


def tokenize_words(text: str) -> list[str]:
    """Return lowercase word tokens used by feature rules."""

    return re.findall(r"[a-zA-Z']+", text.lower())


def calculate_uppercase_ratio(text: str) -> float:
    """Calculate the share of alphabetic characters that are uppercase."""

    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0
    uppercase_letters = [character for character in letters if character.isupper()]
    return len(uppercase_letters) / len(letters)


def extract_features(text: str) -> FeatureVector:
    """Compute deterministic engineered features for one support ticket."""

    normalized_text = normalize_text(text)
    tokens = tokenize_words(normalized_text)
    token_set = set(tokens)

    return FeatureVector(
        char_count=len(normalized_text),
        word_count=len(tokens),
        exclamation_count=normalized_text.count("!"),
        question_count=normalized_text.count("?"),
        uppercase_ratio=calculate_uppercase_ratio(normalized_text),
        urgent_keyword_count=sum(token in URGENT_TERMS for token in tokens),
        high_impact_keyword_count=sum(token in HIGH_IMPACT_TERMS for token in tokens),
        negative_keyword_count=sum(token in NEGATIVE_TERMS for token in tokens),
        has_url=int("http://" in normalized_text or "https://" in normalized_text),
        has_mention=int("@" in normalized_text),
    )


def weak_label_priority(text: str) -> str:
    """Assign a transparent weak priority label from rule-based signals."""

    features = extract_features(text)
    score = 0
    score += min(features.urgent_keyword_count, 3) * 2
    score += min(features.high_impact_keyword_count, 2)
    score += min(features.negative_keyword_count, 2)

    if features.exclamation_count >= 3:
        score += 2
    elif features.exclamation_count > 0:
        score += 1

    if features.uppercase_ratio >= 0.35 and features.word_count >= 3:
        score += 2

    if features.question_count >= 2:
        score += 1

    if score >= 7:
        return "urgent"
    if score >= 4:
        return "high"
    if score >= 2:
        return "normal"
    return "low"
