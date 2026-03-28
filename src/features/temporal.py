"""Temporal features (T1-T6): Time orientation and urgency markers.

References:
- Coppersmith et al. (2015): Temporal language and mental health
- Pennebaker (2011): Past tense and rumination
"""
from __future__ import annotations

from src.features.utils import safe_ratio, word_list_ratio, get_spacy_nlp

# ── Lexicons ──────────────────────────────────────────────────────────────────

TIME_WORDS = frozenset({
    "today", "tomorrow", "yesterday", "now", "later", "soon", "always",
    "never", "before", "after", "recently", "currently", "eventually",
    "already", "still", "yet", "once", "morning", "night", "evening",
    "week", "month", "year", "day", "hour", "minute", "moment",
    "time", "past", "present", "future", "forever", "ago",
})

URGENCY_WORDS = frozenset({
    "now", "immediately", "urgent", "emergency", "help", "please",
    "desperate", "asap", "hurry", "quick", "right now", "tonight",
    "today", "cant wait", "need", "dying", "killing", "end it",
    "crisis", "sos",
})

# spaCy POS tags for tense detection
PAST_TENSE_TAGS = {"VBD", "VBN"}      # past tense, past participle
PRESENT_TENSE_TAGS = {"VBP", "VBZ", "VBG"}  # present, 3rd person, gerund
FUTURE_MARKERS = {"will", "shall", "gonna", "going to", "'ll"}


def extract_temporal_features(
    tokens: list[str],
    text: str | None = None,
) -> dict[str, float]:
    """Extract 6 temporal features.

    Args:
        tokens: Lowercased word tokens.
        text: Original text for spaCy POS tagging (optional but recommended).

    Returns:
        Dict with keys T1-T6.
    """
    if text:
        nlp = get_spacy_nlp()
        doc = nlp(text[:5000])  # Limit for performance
        verb_tags = [token.tag_ for token in doc if token.pos_ == "VERB"]
        n_verbs = len(verb_tags) if verb_tags else 1

        past_count = sum(1 for t in verb_tags if t in PAST_TENSE_TAGS)
        present_count = sum(1 for t in verb_tags if t in PRESENT_TENSE_TAGS)

        # Future tense: detect "will/shall/gonna" + verb patterns
        future_count = sum(1 for token in doc if token.lower_ in FUTURE_MARKERS)

        past_ratio = safe_ratio(past_count, n_verbs)
        present_ratio = safe_ratio(present_count, n_verbs)
        future_ratio = safe_ratio(future_count, n_verbs)
    else:
        past_ratio = 0.0
        present_ratio = 0.0
        future_ratio = 0.0

    # T4: temporal focus (future - past, range roughly -1 to 1)
    temporal_focus = future_ratio - past_ratio

    # T5: time word density
    time_density = word_list_ratio(tokens, TIME_WORDS)

    # T6: urgency score
    urgency = word_list_ratio(tokens, URGENCY_WORDS)

    return {
        "T1_past_tense_ratio": past_ratio,
        "T2_present_tense_ratio": present_ratio,
        "T3_future_tense_ratio": future_ratio,
        "T4_temporal_focus_score": temporal_focus,
        "T5_time_word_density": time_density,
        "T6_urgency_score": urgency,
    }
