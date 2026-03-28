"""Affective features (A1-A6): Emotion-related linguistic markers.

References:
- Al-Mosaiwi & Johnstone (2018): Absolutist words in depression/anxiety/suicidal ideation
- NRC Emotion Lexicon (Mohammad & Turney, 2013)
"""

from src.features.utils import safe_ratio, word_list_ratio

# ── Lexicons ──────────────────────────────────────────────────────────────────

# NRC-derived positive/negative emotion words (core subset)
NEGATIVE_EMOTION_WORDS = frozenset({
    "angry", "sad", "fear", "disgust", "hate", "terrible", "horrible", "awful",
    "miserable", "depressed", "anxious", "worried", "upset", "frustrated",
    "hopeless", "lonely", "guilty", "ashamed", "worthless", "helpless",
    "desperate", "suffering", "pain", "hurt", "cry", "crying", "tears",
    "grief", "sorrow", "agony", "dread", "panic", "terrified", "devastated",
    "heartbroken", "rejected", "abandoned", "betrayed", "humiliated",
    "regret", "resentment", "bitter", "enraged", "furious", "irritated",
    "disgusted", "numb", "empty", "broken", "shattered", "tormented",
})

POSITIVE_EMOTION_WORDS = frozenset({
    "happy", "joy", "love", "wonderful", "great", "excellent", "amazing",
    "fantastic", "beautiful", "grateful", "thankful", "excited", "proud",
    "confident", "peaceful", "calm", "hopeful", "optimistic", "cheerful",
    "delighted", "pleased", "satisfied", "content", "blessed", "fortunate",
    "inspired", "motivated", "enthusiastic", "warm", "kind", "caring",
    "compassionate", "generous", "brave", "strong", "resilient",
    "accomplished", "successful", "thriving", "flourishing",
})

# Al-Mosaiwi 2018 absolutist words
ABSOLUTIST_WORDS = frozenset({
    "always", "never", "nothing", "everything", "completely", "totally",
    "absolutely", "entire", "entirely", "all", "every", "none", "must",
    "definitely", "certainly", "constantly", "forever", "impossible",
    "perfect", "perfectly", "whole", "wholly",
})

ANGER_WORDS = frozenset({
    "angry", "furious", "enraged", "rage", "mad", "hate", "hatred",
    "irritated", "annoyed", "frustrated", "hostile", "bitter",
    "resentment", "outraged", "livid", "infuriated", "aggravated",
    "pissed", "fuming",
})

SADNESS_WORDS = frozenset({
    "sad", "depressed", "unhappy", "miserable", "sorrowful", "grief",
    "mourning", "melancholy", "gloomy", "dismal", "heartbroken",
    "dejected", "despondent", "forlorn", "wretched", "crying",
    "tears", "weeping", "sobbing", "lonely", "loneliness",
})

# DSM-5 anxiety-related terms (Coppersmith et al. 2015; De Choudhury 2013)
ANXIETY_WORDS = frozenset({
    "anxious", "worried", "nervous", "uneasy", "restless", "tense",
    "panicked", "panic", "scared", "terrified", "frightened",
    "apprehensive", "dread", "phobia", "obsessive", "paranoid",
    "overwhelmed", "stressed", "hypervigilant", "jittery", "shaky",
    "fearful", "alarmed", "distressed", "agitated",
})

# Death/suicide-related terms (Coppersmith et al. 2015; key risk markers)
DEATH_WORDS = frozenset({
    "die", "dying", "dead", "death", "kill", "killed", "suicide",
    "suicidal", "funeral", "grave", "bury", "buried", "lethal",
    "fatal", "corpse", "murder", "lifeless", "deceased", "afterlife",
    "overdose", "hang", "drown", "bleed", "suffocate", "poison",
})


def extract_affective_features(tokens: list[str]) -> dict[str, float]:
    """Extract 8 affective features from tokenized text.

    Args:
        tokens: Lowercased word tokens.

    Returns:
        Dict with keys A1-A8.
    """
    neg_ratio = word_list_ratio(tokens, NEGATIVE_EMOTION_WORDS)     # A1
    pos_ratio = word_list_ratio(tokens, POSITIVE_EMOTION_WORDS)     # A2

    # A3: emotion polarity
    eps = 1e-8
    polarity = (pos_ratio - neg_ratio) / (pos_ratio + neg_ratio + eps)

    abs_ratio = word_list_ratio(tokens, ABSOLUTIST_WORDS)           # A4
    anger_ratio = word_list_ratio(tokens, ANGER_WORDS)              # A5
    sadness_ratio = word_list_ratio(tokens, SADNESS_WORDS)          # A6
    anxiety_ratio = word_list_ratio(tokens, ANXIETY_WORDS)          # A7
    death_ratio = word_list_ratio(tokens, DEATH_WORDS)              # A8

    return {
        "A1_negative_emotion_ratio": neg_ratio,
        "A2_positive_emotion_ratio": pos_ratio,
        "A3_emotion_polarity": polarity,
        "A4_absolutist_word_ratio": abs_ratio,
        "A5_anger_word_ratio": anger_ratio,
        "A6_sadness_word_ratio": sadness_ratio,
        "A7_anxiety_word_ratio": anxiety_ratio,
        "A8_death_word_ratio": death_ratio,
    }
