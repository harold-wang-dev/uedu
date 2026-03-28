"""UEDU Model Training: Train an M5-equivalent model from scratch.

Requires the Kaggle Suicide Detection dataset. Download first:
    kaggle datasets download -d nikhileswarkomati/suicide-watch
    unzip suicide-watch.zip -d data/raw/kaggle/suicide-depression/

Usage:
    python examples/train_model.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.preprocessor import clean_text
from src.features.registry import ALL_FEATURE_NAMES, extract_all_features


def load_kaggle_data(data_path: Path) -> pd.DataFrame:
    """Load the Kaggle Suicide Detection dataset."""
    csv_path = data_path / "Suicide_Detection.csv"
    if not csv_path.exists():
        print(f"Dataset not found at {csv_path}")
        print("Download it first:")
        print("  kaggle datasets download -d nikhileswarkomati/suicide-watch")
        print(f"  unzip suicide-watch.zip -d {data_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df["label"] = (df["class"] == "suicide").astype(int)
    df = df[["text", "label"]].dropna(subset=["text"])
    print(f"Loaded {len(df)} texts ({df['label'].sum()} distress, {(1 - df['label']).sum()} control)")
    return df


def extract_features_batch(texts: list[str], batch_size: int = 1000) -> pd.DataFrame:
    """Extract psycholinguistic features for a batch of texts."""
    rows = []
    total = len(texts)
    for i, text in enumerate(texts):
        cleaned = clean_text(text)
        feats = extract_all_features(cleaned, use_spacy=False)
        rows.append(feats)
        if (i + 1) % batch_size == 0:
            print(f"  Extracted features: {i + 1}/{total}")

    print(f"  Extracted features: {total}/{total}")
    return pd.DataFrame(rows)


def train_m5(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> dict:
    """Train M5 (XGBoost on 40 psycholinguistic features) with cross-validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "f1": [], "auc": []}

    print(f"\nTraining M5 with {n_folds}-fold CV...")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = XGBClassifier(
            max_depth=6,
            n_estimators=200,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        metrics["accuracy"].append(acc)
        metrics["f1"].append(f1)
        metrics["auc"].append(auc)
        print(f"  Fold {fold}: Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    print(f"\nMean: Accuracy={np.mean(metrics['accuracy']):.4f}, "
          f"F1={np.mean(metrics['f1']):.4f}, AUC={np.mean(metrics['auc']):.4f}")

    # Train final model on all data
    final_clf = XGBClassifier(
        max_depth=6, n_estimators=200, learning_rate=0.1,
        random_state=42, eval_metric="logloss", use_label_encoder=False,
    )
    final_clf.fit(X, y)

    return {
        "classifier": final_clf,
        "feature_names": ALL_FEATURE_NAMES,
        "metrics": metrics,
    }


def main():
    data_path = ROOT / "data" / "raw" / "kaggle" / "suicide-depression"
    df = load_kaggle_data(data_path)

    # Optional: subsample for faster iteration
    sample_size = min(len(df), 20_000)
    if sample_size < len(df):
        print(f"\nSubsampling to {sample_size} texts for faster training...")
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(sample_size // 2, random_state=42)
        ).reset_index(drop=True)

    print("\nExtracting 40 psycholinguistic features...")
    feature_df = extract_features_batch(df["text"].tolist())

    X = feature_df[ALL_FEATURE_NAMES].values
    y = df["label"].values

    model_data = train_m5(X, y)

    # Save model
    output_path = ROOT / "models" / "M5_custom.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to {output_path}")


if __name__ == "__main__":
    main()
