"""LLM-based features (L1-L8): Gemini API structured output for nuanced assessment.

These features capture aspects that lexicon-based methods miss:
implicit hopelessness, social withdrawal signals, etc.

Used in models M7, M8, M9.
"""
from __future__ import annotations

import json
import logging

from google import genai
from google.genai import types

from src.config import settings

logger = logging.getLogger(__name__)

LLM_FEATURE_NAMES = [
    "L1_implicit_hopelessness",
    "L2_social_withdrawal",
    "L3_cognitive_distortion_severity",
    "L4_emotional_numbness",
    "L5_help_seeking_vs_giving_up",
    "L6_self_harm_risk",
    "L7_overall_distress_score",
    "L8_context_adjusted_risk",
]

ASSESSMENT_PROMPT = """\
You are a clinical NLP research assistant. Analyze the following text and rate each \
dimension on a scale from 0.0 (not present) to 1.0 (strongly present).

Text: "{text}"

Return ONLY a JSON object with these exact keys:
- implicit_hopelessness: Hopelessness expressed indirectly (not just explicit words)
- social_withdrawal: Signs of isolation, avoiding others, or feeling disconnected
- cognitive_distortion_severity: Degree of distorted thinking (catastrophizing, black-white, etc.)
- emotional_numbness: Signs of emotional flatness, detachment, or dissociation
- help_seeking_vs_giving_up: 0.0 = actively giving up, 0.5 = neutral, 1.0 = seeking help
- self_harm_risk: Indicators of self-harm or suicidal ideation (be conservative)
- overall_distress_score: Holistic assessment of psychological distress level
- context_adjusted_risk: Risk level considering this may be a student writing assignment

Respond with ONLY the JSON object, no other text."""

_DEFAULT_FEATURES = {name: 0.0 for name in LLM_FEATURE_NAMES}
_DEFAULT_FEATURES["L5_help_seeking_vs_giving_up"] = 0.5


def _make_client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def extract_llm_features(text: str, client: genai.Client | None = None) -> dict[str, float]:
    """Extract 8 LLM-based features for a single text.

    Args:
        text: Input text to analyze.
        client: Gemini client (created if not provided).

    Returns:
        Dict with 8 float features (L1-L8).
    """
    if client is None:
        client = _make_client()

    truncated = text[:3000] if len(text) > 3000 else text

    try:
        response = client.models.generate_content(
            model=settings.llm_model,
            contents=ASSESSMENT_PROMPT.format(text=truncated),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=300,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        scores = json.loads(response.text.strip())

        return {
            "L1_implicit_hopelessness": float(scores.get("implicit_hopelessness", 0)),
            "L2_social_withdrawal": float(scores.get("social_withdrawal", 0)),
            "L3_cognitive_distortion_severity": float(scores.get("cognitive_distortion_severity", 0)),
            "L4_emotional_numbness": float(scores.get("emotional_numbness", 0)),
            "L5_help_seeking_vs_giving_up": float(scores.get("help_seeking_vs_giving_up", 0.5)),
            "L6_self_harm_risk": float(scores.get("self_harm_risk", 0)),
            "L7_overall_distress_score": float(scores.get("overall_distress_score", 0)),
            "L8_context_adjusted_risk": float(scores.get("context_adjusted_risk", 0)),
        }
    except Exception as e:
        logger.warning(f"LLM feature extraction failed: {e}")
        return dict(_DEFAULT_FEATURES)
