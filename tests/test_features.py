"""Tests for all 30 psycholinguistic feature extractors."""

import pytest

from src.features.affective import extract_affective_features
from src.features.cognitive import extract_cognitive_features
from src.features.temporal import extract_temporal_features
from src.features.structural import extract_structural_features
from src.features.social import extract_social_features
from src.features.registry import extract_all_features, tokenize, ALL_FEATURE_NAMES


# ── Tokenizer ─────────────────────────────────────────────────────────────────


class TestTokenizer:
    def test_basic(self):
        tokens = tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_punctuation_removed(self):
        tokens = tokenize("I can't believe it!")
        assert "can't" in tokens
        assert "!" not in tokens

    def test_empty(self):
        assert tokenize("") == []


# ── Affective Features (A1-A6) ───────────────────────────────────────────────


class TestAffectiveFeatures:
    def test_negative_emotion(self):
        tokens = tokenize("I feel sad depressed and miserable")
        features = extract_affective_features(tokens)
        assert features["A1_negative_emotion_ratio"] > 0
        assert features["A6_sadness_word_ratio"] > 0

    def test_positive_emotion(self):
        tokens = tokenize("I am happy grateful and excited about life")
        features = extract_affective_features(tokens)
        assert features["A2_positive_emotion_ratio"] > 0

    def test_polarity_negative(self):
        tokens = tokenize("sad depressed miserable terrible horrible")
        features = extract_affective_features(tokens)
        assert features["A3_emotion_polarity"] < 0

    def test_polarity_positive(self):
        tokens = tokenize("happy joyful wonderful amazing great")
        features = extract_affective_features(tokens)
        assert features["A3_emotion_polarity"] > 0

    def test_absolutist_words(self):
        tokens = tokenize("I always fail at everything and nothing ever works")
        features = extract_affective_features(tokens)
        assert features["A4_absolutist_word_ratio"] > 0

    def test_anger(self):
        tokens = tokenize("I am so angry and furious about this")
        features = extract_affective_features(tokens)
        assert features["A5_anger_word_ratio"] > 0

    def test_empty_tokens(self):
        features = extract_affective_features([])
        assert all(v == 0.0 for v in features.values())

    def test_returns_six_features(self):
        features = extract_affective_features(["test"])
        assert len(features) == 6
        assert all(k.startswith("A") for k in features)


# ── Cognitive Features (C1-C6) ────────────────────────────────────────────────


class TestCognitiveFeatures:
    def test_distortion_words(self):
        tokens = tokenize("I am hopeless worthless and a total failure")
        features = extract_cognitive_features(tokens)
        assert features["C1_cognitive_distortion_ratio"] > 0

    def test_certainty(self):
        tokens = tokenize("I am absolutely sure and definitely know this")
        features = extract_cognitive_features(tokens)
        assert features["C2_certainty_word_ratio"] > 0

    def test_tentative(self):
        tokens = tokenize("maybe I could possibly try perhaps")
        features = extract_cognitive_features(tokens)
        assert features["C3_tentative_word_ratio"] > 0

    def test_negation(self):
        tokens = tokenize("I cant do nothing right and nobody cares")
        features = extract_cognitive_features(tokens)
        assert features["C4_negation_ratio"] > 0

    def test_causal(self):
        tokens = tokenize("because of this reason I think therefore")
        features = extract_cognitive_features(tokens)
        assert features["C5_causal_word_ratio"] > 0

    def test_insight(self):
        tokens = tokenize("I realize and understand now I think I know")
        features = extract_cognitive_features(tokens)
        assert features["C6_insight_word_ratio"] > 0

    def test_returns_six_features(self):
        features = extract_cognitive_features(["test"])
        assert len(features) == 6
        assert all(k.startswith("C") for k in features)


# ── Temporal Features (T1-T6) ─────────────────────────────────────────────────


class TestTemporalFeatures:
    def test_time_words(self):
        tokens = tokenize("today I feel different from yesterday morning")
        features = extract_temporal_features(tokens)
        assert features["T5_time_word_density"] > 0

    def test_urgency(self):
        tokens = tokenize("help me now please I need help immediately")
        features = extract_temporal_features(tokens)
        assert features["T6_urgency_score"] > 0

    def test_with_spacy(self):
        text = "I walked home yesterday. I was feeling terrible."
        tokens = tokenize(text)
        features = extract_temporal_features(tokens, text=text)
        assert features["T1_past_tense_ratio"] > 0

    def test_without_spacy(self):
        tokens = tokenize("some words here")
        features = extract_temporal_features(tokens, text=None)
        assert features["T1_past_tense_ratio"] == 0.0
        assert features["T2_present_tense_ratio"] == 0.0

    def test_returns_six_features(self):
        features = extract_temporal_features(["test"])
        assert len(features) == 6
        assert all(k.startswith("T") for k in features)


# ── Structural Features (S1-S6) ──────────────────────────────────────────────


class TestStructuralFeatures:
    def test_avg_sentence_length(self):
        text = "Short. Sentences. Here."
        tokens = tokenize(text)
        features = extract_structural_features(tokens, text)
        assert features["S1_avg_sentence_length"] < 5

    def test_lexical_diversity(self):
        # All unique words → high TTR
        tokens = ["one", "two", "three", "four", "five"]
        features = extract_structural_features(tokens, "one two three four five")
        assert features["S2_lexical_diversity_ttr"] == 1.0

        # Repeated words → low TTR
        tokens = ["the", "the", "the", "the", "the"]
        features = extract_structural_features(tokens, "the the the the the")
        assert features["S2_lexical_diversity_ttr"] == 0.2

    def test_punctuation_density(self):
        text = "Hello, world! How are you? I'm fine."
        tokens = tokenize(text)
        features = extract_structural_features(tokens, text)
        assert features["S4_punctuation_density"] > 0

    def test_exclamation_question(self):
        text = "Help! What should I do?! Why?!"
        tokens = tokenize(text)
        features = extract_structural_features(tokens, text)
        assert features["S5_exclamation_question_ratio"] > 0

    def test_word_count_log(self):
        tokens = tokenize("one two three")
        features = extract_structural_features(tokens, "one two three")
        assert features["S6_word_count_log"] > 0

    def test_returns_six_features(self):
        features = extract_structural_features(["test"], "test")
        assert len(features) == 6
        assert all(k.startswith("S") for k in features)


# ── Social Features (SO1-SO6) ────────────────────────────────────────────────


class TestSocialFeatures:
    def test_first_person_singular(self):
        tokens = tokenize("I feel like my life is falling apart and me alone")
        features = extract_social_features(tokens)
        assert features["SO1_first_person_singular_ratio"] > 0

    def test_first_person_plural(self):
        tokens = tokenize("we can do this together as our team")
        features = extract_social_features(tokens)
        assert features["SO2_first_person_plural_ratio"] > 0

    def test_second_person(self):
        tokens = tokenize("you should try your best for yourself")
        features = extract_social_features(tokens)
        assert features["SO3_second_person_ratio"] > 0

    def test_third_person(self):
        tokens = tokenize("he told her that they were leaving")
        features = extract_social_features(tokens)
        assert features["SO4_third_person_ratio"] > 0

    def test_social_references(self):
        tokens = tokenize("my family and friends are the best")
        features = extract_social_features(tokens)
        assert features["SO5_social_reference_ratio"] > 0

    def test_self_vs_other(self):
        # High self-focus
        tokens = tokenize("I me my mine myself")
        features = extract_social_features(tokens)
        assert features["SO6_self_vs_other_ratio"] > 1.0

    def test_returns_six_features(self):
        features = extract_social_features(["test"])
        assert len(features) == 6
        assert all(k.startswith("SO") for k in features)


# ── Registry (All 30 features) ───────────────────────────────────────────────


class TestRegistry:
    def test_extract_all_30_features(self):
        text = "I always feel so hopeless and sad. Nobody cares about me."
        features = extract_all_features(text, use_spacy=False)
        assert len(features) == 30
        for name in ALL_FEATURE_NAMES:
            assert name in features
            assert isinstance(features[name], float)

    def test_extract_all_with_spacy(self):
        text = "I walked home yesterday and felt terrible about everything."
        features = extract_all_features(text, use_spacy=True)
        assert len(features) == 30

    def test_distress_vs_control_signal(self):
        """Distress text should have higher negative emotion than neutral text."""
        distress = "I feel hopeless and worthless. Nothing will ever get better."
        control = "The weather is nice today. I went to the park with friends."

        f_distress = extract_all_features(distress, use_spacy=False)
        f_control = extract_all_features(control, use_spacy=False)

        assert f_distress["A1_negative_emotion_ratio"] > f_control["A1_negative_emotion_ratio"]
        assert f_distress["C1_cognitive_distortion_ratio"] > f_control["C1_cognitive_distortion_ratio"]
