"""Model training: 9 model variants for ablation study.

M1-M3: LogisticRegression baselines
M4-M9: XGBoost variants with different feature sets
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, issparse, csr_matrix
import xgboost as xgb

from src.config import settings
from src.features.registry import ALL_FEATURE_NAMES
from src.features.llm_features import LLM_FEATURE_NAMES

logger = logging.getLogger(__name__)

# ── Feature set definitions ───────────────────────────────────────────────────

PSYCHO_COLS = ALL_FEATURE_NAMES  # 40 features
LLM_COLS = LLM_FEATURE_NAMES    # 8 features

MODEL_CONFIGS = {
    "M1": {"classifier": "logreg", "features": "tfidf"},
    "M2": {"classifier": "logreg", "features": "psycho"},
    "M3": {"classifier": "logreg", "features": "tfidf+psycho"},
    "M4": {"classifier": "xgboost", "features": "tfidf"},
    "M5": {"classifier": "xgboost", "features": "psycho"},
    "M6": {"classifier": "xgboost", "features": "tfidf+psycho"},
    "M7": {"classifier": "xgboost", "features": "llm"},
    "M8": {"classifier": "xgboost", "features": "psycho+llm"},
    "M9": {"classifier": "xgboost", "features": "tfidf+psycho+llm"},
}


def _get_classifier(name: str):
    """Create a classifier instance."""
    if name == "logreg":
        return LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=settings.random_seed,
            solver="lbfgs",
        )
    elif name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=settings.random_seed,
            eval_metric="logloss",
            use_label_encoder=False,
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")


def build_feature_matrix(
    df: pd.DataFrame,
    feature_type: str,
    tfidf_vectorizer: TfidfVectorizer | None = None,
    fit: bool = True,
) -> tuple[np.ndarray | csr_matrix, TfidfVectorizer | None]:
    """Build the feature matrix for a given feature type.

    Args:
        df: DataFrame with 'clean_text' and feature columns.
        feature_type: One of 'tfidf', 'psycho', 'llm', or combinations with '+'.
        tfidf_vectorizer: Pre-fitted vectorizer (for val/test).
        fit: Whether to fit the vectorizer.

    Returns:
        (feature_matrix, tfidf_vectorizer)
    """
    parts = feature_type.split("+")
    matrices = []

    for part in parts:
        if part == "tfidf":
            if tfidf_vectorizer is None:
                tfidf_vectorizer = TfidfVectorizer(
                    max_features=settings.tfidf_max_features,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=5,
                )
            if fit:
                X_tfidf = tfidf_vectorizer.fit_transform(df["clean_text"])
            else:
                X_tfidf = tfidf_vectorizer.transform(df["clean_text"])
            matrices.append(X_tfidf)
        elif part == "psycho":
            X_psycho = df[PSYCHO_COLS].values
            matrices.append(csr_matrix(X_psycho))
        elif part == "llm":
            X_llm = df[LLM_COLS].values
            matrices.append(csr_matrix(X_llm))

    if len(matrices) == 1:
        X = matrices[0]
    else:
        X = hstack(matrices)

    return X, tfidf_vectorizer


def train_model(
    model_name: str,
    train_df: pd.DataFrame,
    n_folds: int | None = None,
) -> dict:
    """Train a single model variant with cross-validation.

    Args:
        model_name: One of M1-M9.
        train_df: Training DataFrame with features already extracted.
        n_folds: Number of CV folds.

    Returns:
        Dict with model, vectorizer, and CV scores.
    """
    config = MODEL_CONFIGS[model_name]
    n_folds = n_folds or settings.n_folds

    logger.info(f"Training {model_name}: {config['classifier']} + {config['features']}")

    X, tfidf_vec = build_feature_matrix(train_df, config["features"])
    y = train_df["label"].values

    if issparse(X):
        X = X.toarray()

    clf = _get_classifier(config["classifier"])

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=settings.random_seed)
    cv_results = cross_validate(
        clf, X, y, cv=cv,
        scoring=["accuracy", "f1", "roc_auc"],
        return_train_score=True,
    )

    # Fit final model on all training data
    clf.fit(X, y)

    results = {
        "model_name": model_name,
        "classifier": clf,
        "tfidf_vectorizer": tfidf_vec,
        "config": config,
        "cv_accuracy": cv_results["test_accuracy"].mean(),
        "cv_f1": cv_results["test_f1"].mean(),
        "cv_roc_auc": cv_results["test_roc_auc"].mean(),
        "cv_accuracy_std": cv_results["test_accuracy"].std(),
        "cv_f1_std": cv_results["test_f1"].std(),
        "cv_roc_auc_std": cv_results["test_roc_auc"].std(),
    }

    logger.info(
        f"  {model_name} CV: acc={results['cv_accuracy']:.4f}±{results['cv_accuracy_std']:.4f}, "
        f"F1={results['cv_f1']:.4f}±{results['cv_f1_std']:.4f}, "
        f"AUC={results['cv_roc_auc']:.4f}±{results['cv_roc_auc_std']:.4f}"
    )
    return results


def save_model(results: dict, output_dir: Path | None = None) -> Path:
    """Save trained model to disk."""
    output_dir = output_dir or settings.results_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{results['model_name']}.pkl"
    with open(path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved {results['model_name']} to {path}")
    return path
