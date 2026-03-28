"""Structural features (S1-S8): Writing style and readability.

References:
- Pennebaker (2010): Type-Token Ratio and psychological state
- Flesch (1948): Readability formulas
- Stirman & Pennebaker (2001): Linguistic markers in suicidal vs non-suicidal poets
"""

import math
import re
import statistics

import textstat

from src.features.utils import safe_ratio, word_list_ratio

# Reuse negative emotion words for per-sentence variability calculation
_NEG_WORDS = frozenset({
    "angry", "sad", "fear", "disgust", "hate", "terrible", "horrible",
    "awful", "miserable", "depressed", "anxious", "worried", "upset",
    "frustrated", "hopeless", "lonely", "guilty", "ashamed", "worthless",
    "helpless", "desperate", "suffering", "pain", "hurt", "cry", "crying",
})


def extract_structural_features(
    tokens: list[str],
    text: str = "",
) -> dict[str, float]:
    """Extract 8 structural features.

    Args:
        tokens: Lowercased word tokens.
        text: Original text (needed for sentence splitting and readability).

    Returns:
        Dict with keys S1-S8.
    """
    # Sentence splitting (simple regex)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    n_sentences = max(len(sentences), 1)
    n_words = max(len(tokens), 1)

    # S1: average sentence length
    avg_sent_len = n_words / n_sentences

    # S2: lexical diversity (Type-Token Ratio)
    unique_tokens = set(tokens)
    ttr = len(unique_tokens) / n_words if n_words > 0 else 0.0

    # S3: Flesch Reading Ease (via textstat)
    if text and len(text) > 10:
        fre = textstat.flesch_reading_ease(text)
    else:
        fre = 0.0

    # S4: punctuation density
    punct_count = sum(1 for ch in text if ch in ".,!?;:'-\"()[]{}...")
    punct_density = safe_ratio(punct_count, len(text)) if text else 0.0

    # S5: exclamation + question mark ratio (over all punctuation)
    excl_q = sum(1 for ch in text if ch in "!?")
    excl_q_ratio = safe_ratio(excl_q, punct_count) if punct_count > 0 else 0.0

    # S6: log word count
    word_count_log = math.log1p(n_words)

    # S7: emotional variability — std dev of per-sentence negative emotion ratio
    # Higher variability may indicate emotional dysregulation (Stirman 2001)
    if n_sentences >= 2:
        sent_neg_ratios = []
        for sent in sentences:
            sent_tokens = re.findall(r"[a-z']+", sent.lower())
            if sent_tokens:
                sent_neg_ratios.append(word_list_ratio(sent_tokens, _NEG_WORDS))
            else:
                sent_neg_ratios.append(0.0)
        emotional_variability = statistics.stdev(sent_neg_ratios)
    else:
        emotional_variability = 0.0

    # S8: repetition score — fraction of bigrams that are repeated
    # Repetitive language is associated with rumination (Pennebaker 2010)
    if len(tokens) >= 2:
        bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        bigram_counts: dict[tuple[str, str], int] = {}
        for bg in bigrams:
            bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
        repeated = sum(1 for c in bigram_counts.values() if c > 1)
        repetition_score = repeated / len(bigram_counts) if bigram_counts else 0.0
    else:
        repetition_score = 0.0

    return {
        "S1_avg_sentence_length": avg_sent_len,
        "S2_lexical_diversity_ttr": ttr,
        "S3_flesch_reading_ease": fre,
        "S4_punctuation_density": punct_density,
        "S5_exclamation_question_ratio": excl_q_ratio,
        "S6_word_count_log": word_count_log,
        "S7_emotional_variability": emotional_variability,
        "S8_repetition_score": repetition_score,
    }
