"""UEDU Quick Start: Analyze any text for mental health distress signals.

This script demonstrates the full UEDU pipeline:
  1. Extract 40 psycholinguistic features (local, deterministic)
  2. Score with pre-trained XGBoost model (local)
  3. Generate a Glass Box teacher report (local, no API calls)

Usage:
    python examples/quick_start.py
    python examples/quick_start.py "Your text here"
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.features.registry import ALL_FEATURE_NAMES, extract_all_features
from src.glass_box import generate_teacher_report


def analyze_text(text: str) -> dict:
    """Run the full UEDU pipeline on a single text."""
    # Step 1: Extract 40 psycholinguistic features
    features = extract_all_features(text, use_spacy=False)

    # Step 2: Load pre-trained M5 model and predict
    model_path = ROOT / "models" / "M5.pkl"
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    clf = model_data["classifier"]
    X = np.array([[features.get(f, 0.0) for f in ALL_FEATURE_NAMES]])
    prob = float(clf.predict_proba(X)[0, 1])

    # Step 3: Generate Glass Box report (100% deterministic, no API)
    report = generate_teacher_report(prob, None, features)
    return report


def main():
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = (
            "I've failed three exams this semester and my parents don't know. "
            "I feel completely worthless and like there's no point in trying anymore. "
            "I've stopped talking to my friends and spend all my time alone in my room."
        )

    print("=" * 60)
    print("UEDU Glass Box Analysis")
    print("=" * 60)
    print(f"\nInput text ({len(text.split())} words):")
    print(f"  {text[:150]}{'...' if len(text) > 150 else ''}")

    report = analyze_text(text)

    print(f"\nRisk Level: {report['risk_level'].upper()}")
    print(f"Risk Score: {report['risk_score']:.1%}")
    print(f"\nExplanation:")
    print(f"  {report['explanation']}")

    print(f"\nTop Contributing Features:")
    for i, feat in enumerate(report["top3_features"], 1):
        print(f"  {i}. {feat['feature']} = {feat['value']:.4f}")
        print(f"     {feat['interpretation']}")

    print(f"\nClinical Protocol: {report['clinical_protocol']['name']}")
    print(f"Urgency: {report['clinical_protocol']['urgency']}")

    print(f"\nTeacher Guidance: {report['teacher_guidance']['title']}")
    for step in report["teacher_guidance"]["steps"]:
        print(f"  - {step}")

    if report["teacher_guidance"]["watch_for"]:
        print(f"\nWatch for: {report['teacher_guidance']['watch_for']}")


if __name__ == "__main__":
    main()
