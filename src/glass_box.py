"""UEDU Glass Box Engine -- 100% local deterministic teacher report generator.

Replaces cloud LLM Layer 3 with template-based interpretations grounded in
SHAP feature importance, NORMAL_RANGES from Exp1, and clinical protocols
(MHFA ALGEE + BC ERASE).

No external API calls. Same input always produces the same output.

All constants (NORMAL_RANGES, SHAP_RANK, FEATURE_DISPLAY, FEATURE_GROUP,
TEACHER_GUIDANCE) are embedded directly -- no imports from demo/ or
external modules required.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Normal ranges for flagging abnormal features.
# Derived from Exp1 control-group distributions (non-distress posts).
# ---------------------------------------------------------------------------
NORMAL_RANGES: dict[str, tuple[float, float]] = {
    "A1_negative_emotion_ratio": (0.0, 0.05),
    "A2_positive_emotion_ratio": (0.01, 0.15),
    "A3_emotion_polarity": (-0.3, 0.3),
    "A4_absolutist_word_ratio": (0.0, 0.02),
    "A5_anger_word_ratio": (0.0, 0.02),
    "A6_sadness_word_ratio": (0.0, 0.02),
    "A7_anxiety_word_ratio": (0.0, 0.02),
    "A8_death_word_ratio": (0.0, 0.001),
    "C1_cognitive_distortion_ratio": (0.0, 0.02),
    "C2_certainty_word_ratio": (0.0, 0.03),
    "C3_tentative_word_ratio": (0.0, 0.03),
    "C4_negation_ratio": (0.0, 0.04),
    "C5_causal_word_ratio": (0.0, 0.03),
    "C6_insight_word_ratio": (0.0, 0.03),
    "C7_discrepancy_word_ratio": (0.0, 0.03),
    "SO1_first_person_singular_ratio": (0.03, 0.12),
    "SO2_first_person_plural_ratio": (0.0, 0.04),
    "SO3_second_person_ratio": (0.0, 0.03),
    "SO4_third_person_ratio": (0.0, 0.04),
    "S2_lexical_diversity": (0.50, 0.95),
}

# ---------------------------------------------------------------------------
# Global SHAP rankings (from Exp4, M5 XGBoost, 30K samples)
# Lower rank = more important. Used to select top-3 features per prediction.
# ---------------------------------------------------------------------------
SHAP_RANK: dict[str, int] = {
    "A8_death_word_ratio": 1,
    "S6_word_count_log": 2,
    "SO6_self_vs_other_ratio": 3,
    "S1_avg_sentence_length": 4,
    "LP1_function_word_ratio": 5,
    "S4_punctuation_density": 6,
    "C7_discrepancy_word_ratio": 7,
    "S5_exclamation_question_ratio": 8,
    "LP4_health_body_word_ratio": 9,
    "LP2_article_ratio": 10,
    "A1_negative_emotion_ratio": 11,
    "SO3_second_person_ratio": 12,
    "A4_absolutist_word_ratio": 13,
    "S2_lexical_diversity_ttr": 14,
    "SO2_first_person_plural_ratio": 15,
    "S7_emotional_variability": 16,
    "C4_negation_ratio": 17,
    "SO5_social_reference_ratio": 18,
    "SO1_first_person_singular_ratio": 19,
    "S8_repetition_score": 20,
    "S3_flesch_reading_ease": 21,
    "C1_cognitive_distortion_ratio": 22,
    "T5_time_word_density": 23,
    "C2_certainty_word_ratio": 24,
    "SO4_third_person_ratio": 25,
    "T6_urgency_score": 26,
    "C6_insight_word_ratio": 27,
    "A5_anger_word_ratio": 28,
    "A2_positive_emotion_ratio": 29,
    "LP5_auxiliary_verb_ratio": 30,
    "C3_tentative_word_ratio": 31,
    "C5_causal_word_ratio": 32,
    "LP3_swear_word_ratio": 33,
    "A6_sadness_word_ratio": 34,
    "A3_emotion_polarity": 35,
    "A7_anxiety_word_ratio": 36,
    "T3_future_tense_ratio": 37,
    "T2_present_tense_ratio": 38,
    "T4_temporal_focus_score": 39,
    "T1_past_tense_ratio": 40,
}

# ---------------------------------------------------------------------------
# Human-readable display names for features
# ---------------------------------------------------------------------------
FEATURE_DISPLAY: dict[str, str] = {
    "A1_negative_emotion_ratio": "Negative Emotions",
    "A2_positive_emotion_ratio": "Positive Emotions",
    "A3_emotion_polarity": "Emotion Polarity",
    "A4_absolutist_word_ratio": "Absolutist Words",
    "A5_anger_word_ratio": "Anger Words",
    "A6_sadness_word_ratio": "Sadness Words",
    "A7_anxiety_word_ratio": "Anxiety Words",
    "A8_death_word_ratio": "Death/Crisis Words",
    "C1_cognitive_distortion_ratio": "Cognitive Distortions",
    "C2_certainty_word_ratio": "Certainty Words",
    "C3_tentative_word_ratio": "Tentative Words",
    "C4_negation_ratio": "Negation (no/not/never)",
    "C5_causal_word_ratio": "Causal Words",
    "C6_insight_word_ratio": "Insight Words",
    "C7_discrepancy_word_ratio": "Discrepancy Words",
    "T1_past_tense_ratio": "Past Tense",
    "T2_present_tense_ratio": "Present Tense",
    "T3_future_tense_ratio": "Future Tense",
    "T4_temporal_focus_score": "Temporal Focus",
    "T5_time_word_density": "Time Word Density",
    "T6_urgency_score": "Urgency Score",
    "S1_avg_sentence_length": "Avg Sentence Length",
    "S2_lexical_diversity_ttr": "Lexical Diversity",
    "S3_flesch_reading_ease": "Reading Ease",
    "S4_punctuation_density": "Punctuation Density",
    "S5_exclamation_question_ratio": "Exclamation/Questions",
    "S6_word_count_log": "Text Length",
    "S7_emotional_variability": "Emotional Variability",
    "S8_repetition_score": "Word Repetition",
    "SO1_first_person_singular_ratio": "Self-Focus (I/me/my)",
    "SO2_first_person_plural_ratio": "We/Us/Our",
    "SO3_second_person_ratio": "You/Your",
    "SO4_third_person_ratio": "He/She/They",
    "SO5_social_word_ratio": "Social Words",
    "SO6_self_vs_other_ratio": "Self vs Others",
    "LP1_function_word_ratio": "Function Words",
    "LP2_article_ratio": "Articles (a/an/the)",
    "LP3_swear_word_ratio": "Strong Language",
    "LP4_health_body_word_ratio": "Health/Body Words",
    "LP5_auxiliary_verb_ratio": "Auxiliary Verbs",
}

# ---------------------------------------------------------------------------
# Feature group classification
# ---------------------------------------------------------------------------
FEATURE_GROUP: dict[str, str] = {
    "A1_negative_emotion_ratio": "Affective",
    "A2_positive_emotion_ratio": "Affective",
    "A3_emotion_polarity": "Affective",
    "A4_absolutist_word_ratio": "Affective",
    "A5_anger_word_ratio": "Affective",
    "A6_sadness_word_ratio": "Affective",
    "A7_anxiety_word_ratio": "Affective",
    "A8_death_word_ratio": "Affective",
    "C1_cognitive_distortion_ratio": "Cognitive",
    "C2_certainty_word_ratio": "Cognitive",
    "C3_tentative_word_ratio": "Cognitive",
    "C4_negation_ratio": "Cognitive",
    "C5_causal_word_ratio": "Cognitive",
    "C6_insight_word_ratio": "Cognitive",
    "C7_discrepancy_word_ratio": "Cognitive",
    "T1_past_tense_ratio": "Temporal",
    "T2_present_tense_ratio": "Temporal",
    "T3_future_tense_ratio": "Temporal",
    "T4_temporal_focus_score": "Temporal",
    "T5_time_word_density": "Temporal",
    "T6_urgency_score": "Temporal",
    "S1_avg_sentence_length": "Structural",
    "S2_lexical_diversity_ttr": "Structural",
    "S3_flesch_reading_ease": "Structural",
    "S4_punctuation_density": "Structural",
    "S5_exclamation_question_ratio": "Structural",
    "S6_word_count_log": "Structural",
    "S7_emotional_variability": "Structural",
    "S8_repetition_score": "Structural",
    "SO1_first_person_singular_ratio": "Social",
    "SO2_first_person_plural_ratio": "Social",
    "SO3_second_person_ratio": "Social",
    "SO4_third_person_ratio": "Social",
    "SO5_social_word_ratio": "Social",
    "SO6_self_vs_other_ratio": "Social",
    "LP1_function_word_ratio": "Linguistic",
    "LP2_article_ratio": "Linguistic",
    "LP3_swear_word_ratio": "Linguistic",
    "LP4_health_body_word_ratio": "Linguistic",
    "LP5_auxiliary_verb_ratio": "Linguistic",
}

# ---------------------------------------------------------------------------
# Teacher guidance (offline, level-based)
# ---------------------------------------------------------------------------
TEACHER_GUIDANCE: dict[str, dict] = {
    "none": {
        "title": "No Action Required",
        "color": "#22c55e",
        "steps": [
            "Continue regular check-ins as part of your normal classroom routine.",
            "Encourage the student's positive writing and self-expression.",
            "Keep the door open -- let students know you're available to talk.",
        ],
        "watch_for": (
            "Changes in writing tone, withdrawal, or declining grades "
            "over the next few weeks."
        ),
    },
    "mild": {
        "title": "Informal Check-In Recommended",
        "color": "#f59e0b",
        "steps": [
            "Have a brief private conversation: 'I noticed your writing "
            "lately -- how are things going?'",
            "Listen without judgment. You don't need to diagnose or solve "
            "the problem.",
            "Mention available resources: school counselor, student services, "
            "or a trusted adult.",
            "Document the conversation briefly in case you need to refer later.",
        ],
        "watch_for": (
            "Escalating themes, social withdrawal, changes in academic "
            "performance, or the student seeking you out."
        ),
    },
    "moderate": {
        "title": "Counselor Referral Recommended -- Same Week",
        "color": "#ef4444",
        "steps": [
            "Speak privately with the student today: 'I care about how "
            "you're doing. Would you be open to talking to the school "
            "counselor?'",
            "Contact the school counselor or student support team to flag "
            "this student.",
            "Do NOT promise confidentiality -- explain that you may need to "
            "share if you're worried about safety.",
            "Inform school administration per your school's mental health "
            "protocol.",
            "Follow up with the student within 2-3 days to show continued "
            "support.",
        ],
        "watch_for": (
            "Any mention of self-harm, hopelessness, or giving things away "
            "-- escalate immediately if observed."
        ),
    },
    "severe": {
        "title": "URGENT -- Immediate Action Required",
        "color": "#7f1d1d",
        "steps": [
            "Do NOT leave the student alone. Remain calm and stay with them.",
            "Contact the school counselor or administrator immediately -- "
            "right now.",
            "If the counselor is unavailable, call the Crisis Line: "
            "1-800-784-2433.",
            "If there is immediate risk of harm, call 9-1-1.",
            "Notify parents/guardians as directed by your school's crisis "
            "protocol.",
            "Document everything: time, what was said, who was contacted.",
        ],
        "watch_for": (
            "This situation requires professional intervention. Your role "
            "is to stay calm, stay present, and connect the student to help."
        ),
    },
}
# Alias so Glass Box "high" level can look up guidance
TEACHER_GUIDANCE["high"] = TEACHER_GUIDANCE["severe"]


# ---------------------------------------------------------------------------
# Risk-level thresholds (3-level system)
# ---------------------------------------------------------------------------
_HIGH_THRESHOLD = 0.75
_MILD_THRESHOLD = 0.30


def _prob_to_risk_level(prob: float) -> str:
    if prob > _HIGH_THRESHOLD:
        return "high"
    if prob >= _MILD_THRESHOLD:
        return "mild"
    return "none"


# ---------------------------------------------------------------------------
# Clinical protocols
# ---------------------------------------------------------------------------
ALGEE_PROTOCOL = {
    "name": "ALGEE (Mental Health First Aid)",
    "urgency": "crisis",
    "steps": [
        "Approach the student privately and assess the situation calmly.",
        "Listen nonjudgmentally. Let the student speak without interruption.",
        "Give reassurance: 'I'm glad you told me. You are not alone.'",
        "Encourage appropriate professional help: school counselor, "
        "crisis line 1-800-784-2433.",
        "Encourage self-help and other support strategies "
        "(trusted adults, peer groups).",
    ],
    "references": [
        "Mental Health First Aid (MHFA) Canada",
        "BC ERASE: https://www2.gov.bc.ca/gov/content/erase",
    ],
}

BC_ERASE_HIGH_RISK = {
    "name": "BC ERASE High-Risk Response",
    "urgency": "urgent",
    "steps": [
        "Do NOT leave the student alone.",
        "Report the concern to school administration immediately.",
        "Contact the Safe School Coordinator or school counselor.",
        "Document specific language and behaviours observed.",
        "Follow your district's threat assessment protocol.",
    ],
    "references": [
        "BC ERASE Bullying Strategy",
        "VTRA (Violence Threat Risk Assessment) Protocol",
    ],
}

OBSERVE_AND_WAIT_PROTOCOL = {
    "name": "Observe and Monitor",
    "urgency": "monitor",
    "steps": [
        "Have a brief, private check-in: 'I noticed your writing lately. "
        "How are things going?'",
        "Listen without judgment. You do not need to diagnose or solve "
        "the problem.",
        "Mention available resources: school counselor, student services, "
        "or a trusted adult.",
        "Document the conversation briefly for future reference.",
        "Schedule a follow-up check-in within 1-2 weeks.",
    ],
    "references": [
        "BC School Counsellors Association guidelines",
    ],
}

NO_ACTION_PROTOCOL = {
    "name": "Routine Monitoring",
    "urgency": "routine",
    "steps": [
        "Continue regular check-ins as part of normal classroom routine.",
        "Encourage the student's positive writing and self-expression.",
        "Keep the door open. Let students know you are available to talk.",
    ],
    "references": [],
}


# ---------------------------------------------------------------------------
# Feature interpretation templates (top SHAP features)
# ---------------------------------------------------------------------------
FEATURE_INTERPRETATIONS: dict[str, dict] = {
    "A8_death_word_ratio": {
        "high_signal": (
            "Death/crisis vocabulary detected at {multiplier:.1f}x the normal baseline. "
            "This feature has 2.1x the predictive power of any other linguistic marker "
            "in the model (SHAP rank #1)."
        ),
        "low_signal": "Death/crisis vocabulary within normal range.",
        "clinical_note": (
            "Presence of death-related words is the single strongest predictor of "
            "distress in the UEDU model. Requires elevated monitoring regardless of "
            "overall score."
        ),
        "severity": "high",
    },
    "SO6_self_vs_other_ratio": {
        "high_signal": (
            "Self-vs-other pronoun ratio is elevated ({value:.2f}, normal: {lo:.2f}-{hi:.2f}), "
            "indicating rumination and potential social withdrawal."
        ),
        "low_signal": "Self-vs-other pronoun balance is within normal range.",
        "clinical_note": (
            "Elevated self-referential language relative to social pronouns is a "
            "validated marker of depressive rumination (Edwards & Holtzman, 2017)."
        ),
        "severity": "moderate",
    },
    "C1_cognitive_distortion_ratio": {
        "high_signal": (
            "Cognitive distortion language detected ({value:.3f}, normal: {lo:.3f}-{hi:.3f}), "
            "suggesting absolutist thinking patterns (always, never, worthless, hopeless)."
        ),
        "low_signal": "Cognitive distortion markers within normal range.",
        "clinical_note": (
            "Absolutist thinking (Al-Mosaiwi & Johnstone, 2018) is significantly "
            "elevated in depression, anxiety, and suicidal ideation."
        ),
        "severity": "moderate",
    },
    "A1_negative_emotion_ratio": {
        "high_signal": (
            "Negative emotion word density is elevated ({value:.3f}, normal: {lo:.3f}-{hi:.3f}), "
            "indicating a high proportion of distress-related language."
        ),
        "low_signal": "Negative emotion word density within normal range.",
        "clinical_note": (
            "Elevated negative emotion language correlates with mood disturbance "
            "and is one of the core LIWC indicators for psychological distress."
        ),
        "severity": "moderate",
    },
    "A4_absolutist_word_ratio": {
        "high_signal": (
            "Absolutist word usage is elevated ({value:.3f}, normal: {lo:.3f}-{hi:.3f}). "
            "Words like 'always', 'never', 'completely' suggest black-and-white thinking."
        ),
        "low_signal": "Absolutist word usage within normal range.",
        "clinical_note": (
            "Absolutist language is a transdiagnostic marker across anxiety, "
            "depression, and suicidal ideation forums (Al-Mosaiwi & Johnstone, 2018)."
        ),
        "severity": "moderate",
    },
    "S6_word_count_log": {
        "high_signal": (
            "Text length (log word count = {value:.2f}) is notably {direction} average, "
            "which may indicate {length_meaning}."
        ),
        "low_signal": "Text length is within typical range.",
        "clinical_note": (
            "Both unusually short and unusually long texts can signal distress. "
            "Very short texts may indicate withdrawal; very long texts may indicate "
            "emotional flooding or crisis disclosure."
        ),
        "severity": "info",
    },
    "SO1_first_person_singular_ratio": {
        "high_signal": (
            "First-person singular pronoun usage (I/me/my) is elevated ({value:.3f}, "
            "normal: {lo:.3f}-{hi:.3f}), indicating heightened self-focus."
        ),
        "low_signal": "First-person pronoun usage within normal range.",
        "clinical_note": (
            "Excessive self-referential language is associated with depression "
            "and social withdrawal (Rude et al., 2004)."
        ),
        "severity": "moderate",
    },
    "S1_avg_sentence_length": {
        "high_signal": (
            "Average sentence length ({value:.1f} words) deviates from normal range "
            "({lo:.1f}-{hi:.1f}), suggesting atypical writing patterns."
        ),
        "low_signal": "Sentence structure is within normal range.",
        "clinical_note": (
            "Sentence fragmentation or run-on patterns can reflect cognitive "
            "disruption associated with emotional distress."
        ),
        "severity": "info",
    },
    "LP1_function_word_ratio": {
        "high_signal": (
            "Function word ratio ({value:.3f}) is outside normal range ({lo:.3f}-{hi:.3f}). "
            "Function words reflect cognitive processing style."
        ),
        "low_signal": "Function word usage within normal range.",
        "clinical_note": (
            "Function word patterns (Pennebaker, 2011) reveal psychological state "
            "more reliably than content words."
        ),
        "severity": "info",
    },
    "C7_discrepancy_word_ratio": {
        "high_signal": (
            "Discrepancy language (should, would, could) elevated ({value:.3f}, "
            "normal: {lo:.3f}-{hi:.3f}), indicating unmet expectations or internal conflict."
        ),
        "low_signal": "Discrepancy word usage within normal range.",
        "clinical_note": (
            "Discrepancy words reflect a gap between current reality and desired "
            "state, common in hopelessness and frustration."
        ),
        "severity": "moderate",
    },
    "LP4_health_body_word_ratio": {
        "high_signal": (
            "Health/body-related word density ({value:.3f}) is elevated, potentially "
            "indicating somatic complaints or health-related distress."
        ),
        "low_signal": "Health/body word density within normal range.",
        "clinical_note": (
            "Somatic language can indicate physical manifestations of psychological "
            "distress, especially in younger populations."
        ),
        "severity": "info",
    },
    "S4_punctuation_density": {
        "high_signal": (
            "Punctuation density ({value:.3f}) deviates from normal range, "
            "indicating atypical writing style."
        ),
        "low_signal": "Punctuation usage within normal range.",
        "clinical_note": (
            "Unusual punctuation patterns (excessive ellipses, lack of periods) "
            "can reflect emotional state and cognitive processing."
        ),
        "severity": "info",
    },
    "C4_negation_ratio": {
        "high_signal": (
            "Negation word usage (no, not, never) is elevated ({value:.3f}, "
            "normal: {lo:.3f}-{hi:.3f}), suggesting a negative cognitive frame."
        ),
        "low_signal": "Negation word usage within normal range.",
        "clinical_note": (
            "Elevated negation is associated with negative cognitive bias, "
            "a core feature of depressive thinking."
        ),
        "severity": "moderate",
    },
    "A6_sadness_word_ratio": {
        "high_signal": (
            "Sadness word density is elevated ({value:.3f}, normal: {lo:.3f}-{hi:.3f}), "
            "indicating expressions of grief, loss, or low mood."
        ),
        "low_signal": "Sadness word density within normal range.",
        "clinical_note": (
            "Sadness language is a direct indicator of low mood and "
            "potential depressive symptoms."
        ),
        "severity": "moderate",
    },
    "A7_anxiety_word_ratio": {
        "high_signal": (
            "Anxiety word density is elevated ({value:.3f}, normal: {lo:.3f}-{hi:.3f}), "
            "indicating worry, fear, or nervousness."
        ),
        "low_signal": "Anxiety word density within normal range.",
        "clinical_note": (
            "Elevated anxiety language may reflect generalized worry, "
            "social anxiety, or acute stress responses."
        ),
        "severity": "moderate",
    },
}

# Fallback template for features without a specific interpretation
_DEFAULT_INTERPRETATION = {
    "high_signal": (
        "{display_name} ({value:.3f}) is outside the normal range ({lo}-{hi}), "
        "contributing to the overall distress signal."
    ),
    "low_signal": "{display_name} is within normal range.",
    "clinical_note": "This feature contributes to the model's overall assessment.",
    "severity": "info",
}


# ---------------------------------------------------------------------------
# Coping strategy banks (age-appropriate, school-actionable)
# ---------------------------------------------------------------------------
_COPING_HIGH = [
    "Connect the student with the school counselor immediately.",
    "If the student is in crisis, call the BC Crisis Line: 1-800-784-2433.",
    "Ensure the student knows they are not alone and that help is available.",
    "Do not promise confidentiality. Safety comes first.",
]

_COPING_MILD = [
    "Encourage the student to talk to a trusted adult "
    "(teacher, counselor, family member).",
    "Suggest journaling or creative expression as a healthy outlet.",
    "Share school wellness resources "
    "(counseling office hours, peer support groups).",
    "Check in again in 1-2 weeks to see how things are going.",
]

_COPING_NONE = [
    "Continue encouraging positive self-expression through writing.",
    "Reinforce that it is always okay to ask for help.",
    "Maintain an open, supportive classroom environment.",
]


# ---------------------------------------------------------------------------
# Abnormal feature flagging (self-contained)
# ---------------------------------------------------------------------------

def flag_abnormal_features(features: dict) -> list[dict]:
    """Identify features outside normal ranges.

    Returns list of {name, value, normal_lo, normal_hi} for flagged features.
    """
    flagged = []
    for name, (lo, hi) in NORMAL_RANGES.items():
        val = features.get(name, 0.0)
        if val < lo or val > hi:
            flagged.append({
                "name": name,
                "value": round(val, 4),
                "normal_lo": lo,
                "normal_hi": hi,
            })
    return flagged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_top3_shap_features(
    feature_data: dict[str, float],
    shap_rank: dict[str, int] | None = None,
    feature_display: dict[str, str] | None = None,
    feature_group: dict[str, str] | None = None,
) -> list[dict]:
    """Select top 3 non-zero features by global SHAP rank."""
    shap_rank = shap_rank or SHAP_RANK
    feature_display = feature_display or FEATURE_DISPLAY
    feature_group = feature_group or FEATURE_GROUP

    candidates = []
    for fname, val in feature_data.items():
        if val == 0.0:
            continue
        rank = shap_rank.get(fname, 999)
        candidates.append((rank, fname, val))
    candidates.sort(key=lambda x: x[0])

    top3 = []
    for rank, fname, val in candidates[:3]:
        normal = NORMAL_RANGES.get(fname)
        lo = normal[0] if normal else 0.0
        hi = normal[1] if normal else 1.0
        is_flagged = val < lo or val > hi if normal else False

        top3.append({
            "name": fname,
            "display_name": feature_display.get(fname, fname),
            "value": round(val, 4),
            "group": feature_group.get(fname, "Other"),
            "shap_rank": rank,
            "normal_lo": lo,
            "normal_hi": hi,
            "is_flagged": is_flagged,
        })
    return top3


def _interpret_feature(feat: dict) -> dict:
    """Generate deterministic interpretation for a single feature."""
    name = feat["name"]
    value = feat["value"]
    lo = feat["normal_lo"]
    hi = feat["normal_hi"]
    is_flagged = feat["is_flagged"]
    display_name = feat["display_name"]

    tmpl = FEATURE_INTERPRETATIONS.get(name, _DEFAULT_INTERPRETATION)

    # Special handling for A8: compute multiplier
    if name == "A8_death_word_ratio":
        upper = NORMAL_RANGES.get(name, (0.0, 0.001))[1]
        multiplier = value / upper if upper > 0 else 0.0
        if is_flagged and value > 0:
            interpretation = tmpl["high_signal"].format(multiplier=multiplier)
        else:
            interpretation = tmpl["low_signal"]
        return {
            "feature": display_name,
            "value": round(value, 4),
            "interpretation": interpretation,
            "clinical_note": tmpl["clinical_note"],
            "severity": tmpl["severity"] if is_flagged else "info",
        }

    # Special handling for S6 (word count): direction-aware
    if name == "S6_word_count_log":
        direction = "above" if value > 4.0 else "below"
        length_meaning = (
            "emotional flooding or crisis disclosure"
            if value > 4.0
            else "withdrawal or reluctance to express"
        )
        if is_flagged:
            interpretation = tmpl["high_signal"].format(
                value=value, direction=direction, length_meaning=length_meaning,
            )
        else:
            interpretation = tmpl["low_signal"]
        return {
            "feature": display_name,
            "value": round(value, 4),
            "interpretation": interpretation,
            "clinical_note": tmpl["clinical_note"],
            "severity": "info",
        }

    # General case
    if is_flagged and value > 0:
        try:
            interpretation = tmpl["high_signal"].format(
                value=value, lo=lo, hi=hi, display_name=display_name,
            )
        except KeyError:
            interpretation = _DEFAULT_INTERPRETATION["high_signal"].format(
                value=value, lo=lo, hi=hi, display_name=display_name,
            )
    else:
        interpretation = tmpl["low_signal"].format(display_name=display_name)

    return {
        "feature": display_name,
        "value": round(value, 4),
        "interpretation": interpretation,
        "clinical_note": tmpl["clinical_note"],
        "severity": tmpl["severity"] if is_flagged else "info",
    }


def _build_explanation(
    risk_level: str,
    prob: float,
    top3_interpreted: list[dict],
    flagged: list[dict],
    death_word_override: bool,
) -> str:
    """Build a deterministic 2-4 sentence explanation paragraph."""
    level_desc = {
        "high": "high-concern indicators",
        "mild": "mild or ambiguous signals",
        "none": "no significant distress indicators",
    }

    parts = [
        f"The model assigns a distress probability of {prob:.1%}, "
        f"corresponding to {level_desc.get(risk_level, 'unknown')}."
    ]

    if top3_interpreted:
        feature_strs = []
        for f in top3_interpreted:
            feature_strs.append(f["interpretation"])
        parts.append(" ".join(feature_strs))

    if death_word_override:
        parts.append(
            "Note: Death/crisis vocabulary was detected. Although the overall "
            "distress probability is low, clinical best practice requires elevated monitoring."
        )

    if flagged and risk_level in ("mild", "high"):
        n = len(flagged)
        parts.append(
            f"{n} psycholinguistic feature{'s' if n > 1 else ''} "
            f"{'are' if n > 1 else 'is'} outside the normal range established "
            f"from control-group distributions."
        )

    return " ".join(parts)


def _select_clinical_protocol(risk_level: str, has_death_words: bool) -> dict:
    """Select the appropriate clinical protocol."""
    if risk_level == "high":
        return {
            "name": f"{ALGEE_PROTOCOL['name']} + {BC_ERASE_HIGH_RISK['name']}",
            "urgency": "crisis",
            "steps": ALGEE_PROTOCOL["steps"] + BC_ERASE_HIGH_RISK["steps"],
            "references": list(
                dict.fromkeys(
                    ALGEE_PROTOCOL["references"] + BC_ERASE_HIGH_RISK["references"]
                )
            ),
        }
    if risk_level == "mild":
        protocol = {**OBSERVE_AND_WAIT_PROTOCOL}
        if has_death_words:
            protocol["steps"] = [
                "PRIORITY: Death/crisis language was detected. "
                "Contact the school counselor within 24 hours.",
            ] + protocol["steps"]
            protocol["urgency"] = "urgent"
        return protocol
    return {**NO_ACTION_PROTOCOL}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_teacher_report(
    risk_score: float,
    shap_values: dict | None,
    feature_data: dict,
    *,
    _shap_rank: dict | None = None,
    _feature_display: dict | None = None,
    _feature_group: dict | None = None,
) -> dict:
    """Generate a fully deterministic teacher report. No external API calls.

    Args:
        risk_score: M5 XGBoost probability (0.0 to 1.0).
        shap_values: Per-prediction SHAP values (optional, currently unused).
            When None, global SHAP rankings are used instead.
        feature_data: Dict of all 40 psycholinguistic feature values.
        _shap_rank: Override for SHAP rankings (for testing).
        _feature_display: Override for display name map (for testing).
        _feature_group: Override for group map (for testing).

    Returns:
        Dict with risk_level, explanation, top3_features, clinical_protocol,
        coping_strategies, teacher_guidance, and flagged_features.
    """
    shap_rank = _shap_rank or SHAP_RANK
    feature_display = _feature_display or FEATURE_DISPLAY
    feature_group = _feature_group or FEATURE_GROUP

    # 1. Risk level
    risk_level = _prob_to_risk_level(risk_score)

    # 2. Death-word override
    a8_value = feature_data.get("A8_death_word_ratio", 0.0)
    death_word_override = False
    if a8_value > 0.0 and risk_level == "none":
        risk_level = "mild"
        death_word_override = True

    # 3. Top-3 SHAP features
    top3_raw = _get_top3_shap_features(
        feature_data, shap_rank, feature_display, feature_group,
    )
    top3_interpreted = [_interpret_feature(f) for f in top3_raw]

    # 4. Flagged features (outside normal range)
    flagged = flag_abnormal_features(feature_data)

    # 5. Explanation
    explanation = _build_explanation(
        risk_level, risk_score, top3_interpreted, flagged, death_word_override,
    )

    # 6. Clinical protocol
    has_death_words = a8_value > 0.0
    clinical_protocol = _select_clinical_protocol(risk_level, has_death_words)

    # 7. Coping strategies
    if risk_level == "high":
        coping = _COPING_HIGH
    elif risk_level == "mild":
        coping = _COPING_MILD
    else:
        coping = _COPING_NONE

    # 8. Teacher guidance
    guidance_key = {"none": "none", "mild": "mild", "high": "severe"}.get(
        risk_level, "mild"
    )
    teacher_guidance = TEACHER_GUIDANCE.get(guidance_key, TEACHER_GUIDANCE["mild"])

    return {
        "risk_level": risk_level,
        "risk_score": round(risk_score, 4),
        "explanation": explanation,
        "top3_features": top3_interpreted,
        "flagged_features": flagged,
        "clinical_protocol": clinical_protocol,
        "coping_strategies": coping,
        "teacher_guidance": {
            "title": teacher_guidance["title"],
            "color": teacher_guidance["color"],
            "steps": teacher_guidance["steps"],
            "watch_for": teacher_guidance.get("watch_for", ""),
        },
    }
