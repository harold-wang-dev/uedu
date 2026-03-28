"""UEDU Feature Extraction: Extract 40 psycholinguistic features from text.

Demonstrates how to use the feature extraction pipeline independently.

Usage:
    python examples/extract_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.preprocessor import clean_text
from src.features.registry import ALL_FEATURE_NAMES, extract_all_features

FEATURE_GROUPS = {
    "Affective": [f for f in ALL_FEATURE_NAMES if f.startswith("A")],
    "Cognitive": [f for f in ALL_FEATURE_NAMES if f.startswith("C")],
    "Temporal": [f for f in ALL_FEATURE_NAMES if f.startswith("T")],
    "Structural": [f for f in ALL_FEATURE_NAMES if f.startswith("S") and not f.startswith("SO")],
    "Social": [f for f in ALL_FEATURE_NAMES if f.startswith("SO")],
    "Linguistic Process": [f for f in ALL_FEATURE_NAMES if f.startswith("LP")],
}


def main():
    samples = [
        ("High concern", "I feel completely worthless. Nothing matters anymore. I just want to disappear."),
        ("Mild concern", "Lately everything feels heavy. I get anxious before class and wonder if things will get better."),
        ("No concern", "Studying hard for the math competition! My team has been practicing every day and I think we're ready."),
    ]

    for label, text in samples:
        print(f"\n{'=' * 60}")
        print(f"Sample: {label}")
        print(f"Text: {text[:80]}...")
        print(f"{'=' * 60}")

        cleaned = clean_text(text)
        features = extract_all_features(cleaned, use_spacy=False)

        for group_name, group_features in FEATURE_GROUPS.items():
            print(f"\n  {group_name}:")
            for fname in group_features:
                val = features.get(fname, 0.0)
                if val > 0:
                    print(f"    {fname}: {val:.4f}")

    # Batch extraction example
    print(f"\n{'=' * 60}")
    print("Batch extraction to DataFrame:")
    print(f"{'=' * 60}")

    texts = [s[1] for s in samples]
    rows = []
    for text in texts:
        cleaned = clean_text(text)
        feats = extract_all_features(cleaned, use_spacy=False)
        rows.append(feats)

    df = pd.DataFrame(rows)
    print(f"\nShape: {df.shape} (texts x features)")
    print(f"Non-zero features per text: {(df > 0).sum(axis=1).tolist()}")
    print(f"\nTop features by variance:")
    top_var = df.var().sort_values(ascending=False).head(10)
    for fname, var in top_var.items():
        print(f"  {fname}: variance = {var:.6f}")


if __name__ == "__main__":
    main()
