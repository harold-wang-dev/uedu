"""Feature registry: orchestrates extraction of all 40 psycholinguistic features.

Usage:
    from src.features.registry import extract_all_features
    features = extract_all_features("I feel so hopeless and alone")
    # Returns dict with 40 keys (A1-A8, C1-C7, T1-T6, S1-S8, SO1-SO6, LP1-LP5)
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from src.features.affective import extract_affective_features
from src.features.cognitive import extract_cognitive_features
from src.features.temporal import extract_temporal_features
from src.features.structural import extract_structural_features
from src.features.social import extract_social_features
from src.features.linguistic_process import extract_linguistic_process_features

logger = logging.getLogger(__name__)

# Canonical feature order
FEATURE_GROUPS = {
    "affective": [f"A{i}" for i in range(1, 9)],         # A1-A8
    "cognitive": [f"C{i}" for i in range(1, 8)],          # C1-C7
    "temporal": [f"T{i}" for i in range(1, 7)],           # T1-T6
    "structural": [f"S{i}" for i in range(1, 9)],         # S1-S8
    "social": [f"SO{i}" for i in range(1, 7)],            # SO1-SO6
    "linguistic_process": [f"LP{i}" for i in range(1, 6)],  # LP1-LP5
}

ALL_FEATURE_NAMES = [
    # Affective (8)
    "A1_negative_emotion_ratio", "A2_positive_emotion_ratio", "A3_emotion_polarity",
    "A4_absolutist_word_ratio", "A5_anger_word_ratio", "A6_sadness_word_ratio",
    "A7_anxiety_word_ratio", "A8_death_word_ratio",
    # Cognitive (7)
    "C1_cognitive_distortion_ratio", "C2_certainty_word_ratio", "C3_tentative_word_ratio",
    "C4_negation_ratio", "C5_causal_word_ratio", "C6_insight_word_ratio",
    "C7_discrepancy_word_ratio",
    # Temporal (6)
    "T1_past_tense_ratio", "T2_present_tense_ratio", "T3_future_tense_ratio",
    "T4_temporal_focus_score", "T5_time_word_density", "T6_urgency_score",
    # Structural (8)
    "S1_avg_sentence_length", "S2_lexical_diversity_ttr", "S3_flesch_reading_ease",
    "S4_punctuation_density", "S5_exclamation_question_ratio", "S6_word_count_log",
    "S7_emotional_variability", "S8_repetition_score",
    # Social (6)
    "SO1_first_person_singular_ratio", "SO2_first_person_plural_ratio",
    "SO3_second_person_ratio", "SO4_third_person_ratio",
    "SO5_social_reference_ratio", "SO6_self_vs_other_ratio",
    # Linguistic process (5)
    "LP1_function_word_ratio", "LP2_article_ratio", "LP3_swear_word_ratio",
    "LP4_health_body_word_ratio", "LP5_auxiliary_verb_ratio",
]


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with lowercasing and basic cleanup."""
    text = text.lower()
    # Split on whitespace, remove pure punctuation tokens
    tokens = re.findall(r"[a-z']+", text)
    return tokens


def extract_all_features(text: str, use_spacy: bool = True) -> dict[str, float]:
    """Extract all 40 psycholinguistic features from a text.

    Args:
        text: Raw or cleaned text string.
        use_spacy: Whether to use spaCy for POS-based features (temporal).

    Returns:
        Dict with 40 float values keyed by feature names.
    """
    tokens = tokenize(text)
    features = {}

    features.update(extract_affective_features(tokens))
    features.update(extract_cognitive_features(tokens))
    features.update(extract_temporal_features(tokens, text=text if use_spacy else None))
    features.update(extract_structural_features(tokens, text=text))
    features.update(extract_social_features(tokens))
    features.update(extract_linguistic_process_features(tokens))

    return features


def extract_features_batch(
    texts: list[str] | pd.Series,
    use_spacy: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Extract features for a batch of texts.

    Args:
        texts: Iterable of text strings.
        use_spacy: Whether to use spaCy for temporal features.
        show_progress: Log progress every 1000 texts.

    Returns:
        DataFrame with 40 feature columns, one row per text.
    """
    records = []
    total = len(texts)
    for i, text in enumerate(texts):
        if show_progress and i > 0 and i % 1000 == 0:
            logger.info(f"Extracting features: {i}/{total} ({i/total:.0%})")
        features = extract_all_features(str(text), use_spacy=use_spacy)
        records.append(features)

    df = pd.DataFrame(records, columns=ALL_FEATURE_NAMES)
    logger.info(f"Extracted {len(df)} x {len(df.columns)} feature matrix")
    return df
