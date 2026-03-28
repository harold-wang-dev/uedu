"""Social features (SO1-SO6): Pronoun usage and social references.

References:
- Edwards & Holtzman (2017): First-person singular pronoun use and depression (r=0.13)
- Pennebaker et al. (2003): Linguistic markers of psychological state
"""

from src.features.utils import safe_ratio, word_list_ratio

# ── Lexicons ──────────────────────────────────────────────────────────────────

FIRST_PERSON_SINGULAR = frozenset({
    "i", "me", "my", "mine", "myself",
})

FIRST_PERSON_PLURAL = frozenset({
    "we", "us", "our", "ours", "ourselves",
})

SECOND_PERSON = frozenset({
    "you", "your", "yours", "yourself", "yourselves",
})

THIRD_PERSON = frozenset({
    "he", "she", "him", "her", "his", "hers", "himself", "herself",
    "they", "them", "their", "theirs", "themselves",
    "it", "its", "itself",
})

SOCIAL_REFERENCE_WORDS = frozenset({
    "friend", "friends", "family", "mother", "father", "mom", "dad",
    "parent", "parents", "brother", "sister", "husband", "wife",
    "partner", "boyfriend", "girlfriend", "colleague", "teacher",
    "alone", "lonely", "isolated", "together", "relationship",
    "people", "someone", "everyone", "nobody", "group", "community",
    "support", "help", "talk", "listen", "share", "care",
})


def extract_social_features(tokens: list[str]) -> dict[str, float]:
    """Extract 6 social features from tokenized text.

    Args:
        tokens: Lowercased word tokens.

    Returns:
        Dict with keys SO1-SO6.
    """
    fps = word_list_ratio(tokens, FIRST_PERSON_SINGULAR)   # SO1
    fpp = word_list_ratio(tokens, FIRST_PERSON_PLURAL)     # SO2
    sp = word_list_ratio(tokens, SECOND_PERSON)            # SO3
    tp = word_list_ratio(tokens, THIRD_PERSON)             # SO4
    social = word_list_ratio(tokens, SOCIAL_REFERENCE_WORDS)  # SO5

    # SO6: self vs other ratio
    other_total = sp + tp + fpp + 0.01  # epsilon to avoid division by zero
    self_other = fps / other_total

    return {
        "SO1_first_person_singular_ratio": fps,
        "SO2_first_person_plural_ratio": fpp,
        "SO3_second_person_ratio": sp,
        "SO4_third_person_ratio": tp,
        "SO5_social_reference_ratio": social,
        "SO6_self_vs_other_ratio": self_other,
    }
