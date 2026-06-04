import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

from transaction_analysis.models.factory import ModelType

logger = logging.getLogger(__name__)

METRICS = {
    "roc_auc": roc_auc_score,
    "avg_precision": average_precision_score,
}


def score_fold(result: dict) -> dict:
    scores = {"fold": result["fold"]}
    for name, fn in METRICS.items():
        scores[name] = fn(result["y_test"], result["y_proba"])
    return scores


def summarise_cv(fold_results: list[dict]) -> pd.DataFrame:
    """Returns a DataFrame with one row per fold + a summary row."""
    rows = [score_fold(r) for r in fold_results]
    df = pd.DataFrame(rows).set_index("fold")
    summary = df.agg(["mean", "std"])
    return pd.concat([df, summary])


def print_report(fold_results: list[dict], model_name: ModelType) -> None:
    y_test_all = np.concatenate([r["y_test"] for r in fold_results])
    y_pred_all = np.concatenate([r["y_pred"] for r in fold_results])
    y_proba_all = np.concatenate([r["y_proba"] for r in fold_results])
    logger.info(f"\n{model_name.name} — pooled CV classification report")
    logger.info(classification_report(y_test_all, y_pred_all, digits=4))
    logger.info(f"ROC-AUC (pooled): {roc_auc_score(y_test_all, y_proba_all):.4f}")
