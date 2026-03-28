"""Shared utilities for feature extraction."""

import spacy

_nlp = None


def get_spacy_nlp():
    """Lazy-load spaCy model (singleton)."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def safe_ratio(count: int, total: int, eps: float = 1e-8) -> float:
    """Compute ratio safely avoiding division by zero."""
    if total == 0:
        return 0.0
    return count / (total + eps)


def word_list_ratio(tokens: list[str], word_set: set[str]) -> float:
    """Compute fraction of tokens that appear in word_set."""
    if not tokens:
        return 0.0
    matches = sum(1 for t in tokens if t in word_set)
    return matches / len(tokens)
