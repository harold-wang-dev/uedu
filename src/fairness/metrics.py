"""Fairness metrics for demographic parity and equalized odds analysis.

All functions assume:
  - label=0: non-distressed (control)
  - label=1: distressed (positive)
  - For ASAP 2.0 audit, ALL essays are label=0, so any prediction of 1 = false positive.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ── Per-group metrics ─────────────────────────────────────────────────────────


def group_metrics(
    df: pd.DataFrame,
    group_col: str,
    pred_col: str,
    label_col: str = "label",
) -> pd.DataFrame:
    """Compute FPR, FNR, accuracy, F1 for each value of a demographic column.

    Args:
        df: DataFrame with label and prediction columns.
        group_col: Column name of the demographic group (e.g. 'ell_status').
        pred_col: Column name of model predictions (binary 0/1).
        label_col: Column name of ground-truth labels.

    Returns:
        DataFrame with one row per group and columns:
        group, n, n_pos, n_neg, tp, fp, fn, tn, fpr, fnr, accuracy, f1,
        precision, recall, positive_rate.
    """
    rows = []
    for group_val in sorted(df[group_col].dropna().unique()):
        sub = df[df[group_col] == group_val]
        y_true = sub[label_col].values
        y_pred = sub[pred_col].values

        # Build confusion matrix (handles edge case where only one class present)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        pos_rate = y_pred.mean()

        rows.append({
            "group": str(group_val),
            "n": len(sub),
            "n_neg": int(tn + fp),
            "n_pos": int(fn + tp),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "positive_rate": round(pos_rate, 4),
        })

    result = pd.DataFrame(rows)

    # Compute FPR ratio vs reference group (first group alphabetically)
    if len(result) > 0:
        ref_fpr = result.iloc[0]["fpr"]
        result["fpr_ratio_vs_ref"] = result["fpr"].apply(
            lambda x: round(x / ref_fpr, 4) if ref_fpr > 0 else float("nan")
        )
        result["ref_group"] = result.iloc[0]["group"]

    return result


# ── Aggregate fairness metrics ────────────────────────────────────────────────


def demographic_parity_diff(group_df: pd.DataFrame) -> float:
    """Max - Min positive prediction rate across groups.

    A value of 0 = perfect parity. Generally acceptable if < 0.10.
    """
    rates = group_df["positive_rate"]
    return round(float(rates.max() - rates.min()), 4)


def equalized_odds_diff(group_df: pd.DataFrame) -> dict[str, float]:
    """Max - Min FPR and FNR across groups (equalized odds criterion).

    Returns:
        Dict with fpr_diff and fnr_diff. Target: both < 0.10.
    """
    return {
        "fpr_diff": round(float(group_df["fpr"].max() - group_df["fpr"].min()), 4),
        "fnr_diff": round(float(group_df["fnr"].max() - group_df["fnr"].min()), 4),
    }


def disparate_impact_ratio(group_df: pd.DataFrame) -> float:
    """Min / Max positive prediction rate (80% rule).

    EEOC standard: ratio < 0.80 signals adverse impact.
    """
    rates = group_df["positive_rate"]
    if rates.max() == 0:
        return float("nan")
    return round(float(rates.min() / rates.max()), 4)


# ── Statistical significance ──────────────────────────────────────────────────


def chi2_fpr_test(group_df: pd.DataFrame) -> dict[str, float]:
    """Chi-square test of independence for FPR differences across groups.

    Contingency table: rows = groups, cols = [false_positives, true_negatives].

    Returns:
        Dict with chi2, p_value, dof, significant (p < 0.05).
    """
    # Only use groups that have negative samples (FPR is meaningful)
    neg_groups = group_df[group_df["n_neg"] > 0]
    if len(neg_groups) < 2:
        return {"chi2": float("nan"), "p_value": float("nan"), "dof": 0, "significant": False}

    contingency = neg_groups[["fp", "tn"]].values
    chi2, p_value, dof, _ = chi2_contingency(contingency)
    return {
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "dof": int(dof),
        "significant": bool(p_value < 0.05),
    }


# ── Summary report ────────────────────────────────────────────────────────────


def fairness_summary(
    group_dfs: dict[str, pd.DataFrame],
    model_name: str,
) -> pd.DataFrame:
    """Aggregate fairness metrics across all demographic dimensions.

    Args:
        group_dfs: Dict mapping dimension name → per-group metrics DataFrame.
        model_name: Label for the model being audited.

    Returns:
        Summary DataFrame with one row per demographic dimension.
    """
    rows = []
    for dimension, gdf in group_dfs.items():
        chi2_result = chi2_fpr_test(gdf)
        eo = equalized_odds_diff(gdf)
        rows.append({
            "model": model_name,
            "dimension": dimension,
            "n_groups": len(gdf),
            "fpr_min": gdf["fpr"].min(),
            "fpr_max": gdf["fpr"].max(),
            "fpr_diff": eo["fpr_diff"],
            "fnr_diff": eo["fnr_diff"],
            "demographic_parity_diff": demographic_parity_diff(gdf),
            "disparate_impact_ratio": disparate_impact_ratio(gdf),
            "chi2_p_value": chi2_result["p_value"],
            "chi2_significant": chi2_result["significant"],
        })
    return pd.DataFrame(rows)
