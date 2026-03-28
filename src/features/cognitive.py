"""Cognitive features (C1-C6): Thinking patterns and cognitive distortions.

References:
- Beck (1979): Cognitive distortions in depression
- Pennebaker et al. (2003): LIWC cognitive process words
"""

from src.features.utils import word_list_ratio

# ── Lexicons ──────────────────────────────────────────────────────────────────

COGNITIVE_DISTORTION_WORDS = frozenset({
    "hopeless", "worthless", "useless", "pointless", "meaningless",
    "helpless", "failure", "loser", "stupid", "idiot", "pathetic",
    "burden", "ruined", "doomed", "should", "shouldnt", "must",
    "cant", "wont", "impossible", "unbearable", "intolerable",
    "unforgivable", "unlovable", "defective", "broken", "damaged",
    "trapped", "stuck", "inferior",
})

CERTAINTY_WORDS = frozenset({
    "always", "never", "absolutely", "definitely", "certainly",
    "sure", "obvious", "clearly", "undoubtedly", "guaranteed",
    "positive", "convinced", "know", "fact", "proven", "exact",
    "truly", "without doubt",
})

TENTATIVE_WORDS = frozenset({
    "maybe", "perhaps", "might", "could", "possibly", "probably",
    "guess", "suppose", "unsure", "uncertain", "unclear", "wonder",
    "seem", "seems", "appears", "apparently", "roughly", "somewhat",
    "sort of", "kind of",
})

NEGATION_WORDS = frozenset({
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "nor", "none", "dont", "doesnt", "didnt", "wont", "wouldnt",
    "cant", "cannot", "couldnt", "shouldnt", "isnt", "arent",
    "wasnt", "werent", "havent", "hasnt", "hadnt",
})

CAUSAL_WORDS = frozenset({
    "because", "reason", "why", "cause", "caused", "since", "therefore",
    "thus", "hence", "result", "consequence", "due", "effect",
    "lead", "leads", "leading",
})

INSIGHT_WORDS = frozenset({
    "think", "know", "realize", "understand", "believe", "consider",
    "thought", "aware", "recognize", "discover", "found", "learn",
    "learned", "meaning", "sense", "figure", "figured", "conclude",
    "reflect", "insight",
})


# LIWC "discrepancy" category — gap between reality and desire (Pennebaker 2003)
DISCREPANCY_WORDS = frozenset({
    "should", "would", "could", "ought", "want", "need", "wish",
    "hope", "expect", "desire", "prefer", "lack", "miss", "without",
    "rather", "instead", "suppose", "if only",
})


def extract_cognitive_features(tokens: list[str]) -> dict[str, float]:
    """Extract 7 cognitive features from tokenized text.

    Args:
        tokens: Lowercased word tokens.

    Returns:
        Dict with keys C1-C7.
    """
    return {
        "C1_cognitive_distortion_ratio": word_list_ratio(tokens, COGNITIVE_DISTORTION_WORDS),
        "C2_certainty_word_ratio": word_list_ratio(tokens, CERTAINTY_WORDS),
        "C3_tentative_word_ratio": word_list_ratio(tokens, TENTATIVE_WORDS),
        "C4_negation_ratio": word_list_ratio(tokens, NEGATION_WORDS),
        "C5_causal_word_ratio": word_list_ratio(tokens, CAUSAL_WORDS),
        "C6_insight_word_ratio": word_list_ratio(tokens, INSIGHT_WORDS),
        "C7_discrepancy_word_ratio": word_list_ratio(tokens, DISCREPANCY_WORDS),
    }
