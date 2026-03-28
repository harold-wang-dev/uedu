"""Model evaluation: metrics, confusion matrices, and comparison tables."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.models.trainer import build_feature_matrix

logger = logging.getLogger(__name__)


def evaluate_model(
    model_results: dict,
    test_df: pd.DataFrame,
) -> dict:
    """Evaluate a trained model on test data.

    Args:
        model_results: Output from trainer.train_model().
        test_df: Test DataFrame with features.

    Returns:
        Dict with test metrics.
    """
    clf = model_results["classifier"]
    config = model_results["config"]

    X_test, _ = build_feature_matrix(
        test_df,
        config["features"],
        tfidf_vectorizer=model_results.get("tfidf_vectorizer"),
        fit=False,
    )

    from scipy.sparse import issparse
    if issparse(X_test):
        X_test = X_test.toarray()

    y_true = test_df["label"].values
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_results["model_name"],
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_f1": f1_score(y_true, y_pred),
        "test_roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }

    logger.info(
        f"{metrics['model_name']} test: "
        f"acc={metrics['test_accuracy']:.4f}, "
        f"F1={metrics['test_f1']:.4f}, "
        f"AUC={metrics['test_roc_auc']:.4f}"
    )
    return metrics


def compare_models(all_results: list[dict]) -> pd.DataFrame:
    """Create comparison table for all model variants."""
    rows = []
    for r in all_results:
        rows.append({
            "Model": r["model_name"],
            "CV Accuracy": f"{r.get('cv_accuracy', 0):.4f}",
            "CV F1": f"{r.get('cv_f1', 0):.4f}",
            "CV AUC": f"{r.get('cv_roc_auc', 0):.4f}",
            "Test Accuracy": f"{r.get('test_accuracy', 0):.4f}",
            "Test F1": f"{r.get('test_f1', 0):.4f}",
            "Test AUC": f"{r.get('test_roc_auc', 0):.4f}",
        })
    return pd.DataFrame(rows)
