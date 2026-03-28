"""Linguistic process features (LP1-LP5): Function words and LIWC-style categories.

References:
- Pennebaker et al. (2003): LIWC function word categories
- Chung & Pennebaker (2007): Function words as markers of psychological state
- Hovy & Spruit (2016): Demographic bias in NLP
"""

from src.features.utils import word_list_ratio

# ── Lexicons ──────────────────────────────────────────────────────────────────

# LIWC "function words" — the most common closed-class words
FUNCTION_WORDS = frozenset({
    "the", "a", "an", "of", "in", "to", "for", "with", "on", "at",
    "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "can", "could", "must",
    "that", "which", "who", "whom", "this", "these", "those",
    "it", "its", "and", "but", "or", "nor", "so", "yet", "if",
    "then", "than", "as", "not", "no",
})

# Articles (subset of function words, tracked separately in LIWC)
ARTICLES = frozenset({"a", "an", "the"})

# Swear/taboo words — LIWC "swear" category (Pennebaker 2003)
# Elevated swearing correlates with emotional arousal and lower social filtering
SWEAR_WORDS = frozenset({
    "fuck", "fucking", "fucked", "shit", "damn", "hell", "ass",
    "bitch", "bastard", "crap", "bullshit", "asshole", "piss",
    "dick", "dammit", "goddamn", "wtf", "stfu", "suck", "sucks",
})

# Health and body words — LIWC "bio" category
# Somatic complaints often co-occur with depression/anxiety (De Choudhury 2013)
HEALTH_BODY_WORDS = frozenset({
    "body", "heart", "brain", "head", "stomach", "chest", "pain",
    "ache", "sick", "ill", "hospital", "doctor", "medicine",
    "medication", "drug", "pill", "therapy", "treatment", "sleep",
    "insomnia", "fatigue", "tired", "exhausted", "nausea", "dizzy",
    "weight", "eat", "eating", "appetite", "blood", "breath",
    "breathing", "skin", "sweat", "vomit", "clinic", "nurse",
})

# Auxiliary verbs — tracked separately in LIWC; higher rates in depression
AUXILIARY_VERBS = frozenset({
    "can", "could", "may", "might", "shall", "should", "will",
    "would", "must", "do", "does", "did", "have", "has", "had",
    "am", "is", "are", "was", "were", "be", "been", "being",
})


def extract_linguistic_process_features(tokens: list[str]) -> dict[str, float]:
    """Extract 5 linguistic process features from tokenized text.

    Args:
        tokens: Lowercased word tokens.

    Returns:
        Dict with keys LP1-LP5.
    """
    return {
        "LP1_function_word_ratio": word_list_ratio(tokens, FUNCTION_WORDS),
        "LP2_article_ratio": word_list_ratio(tokens, ARTICLES),
        "LP3_swear_word_ratio": word_list_ratio(tokens, SWEAR_WORDS),
        "LP4_health_body_word_ratio": word_list_ratio(tokens, HEALTH_BODY_WORDS),
        "LP5_auxiliary_verb_ratio": word_list_ratio(tokens, AUXILIARY_VERBS),
    }
