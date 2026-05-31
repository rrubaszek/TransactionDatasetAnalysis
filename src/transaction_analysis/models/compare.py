import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

from transaction_analysis.models.evaluate import score_fold


def wilcoxon_test(scores_a: list[float], scores_b: list[float], metric: str = "roc_auc") -> dict:
    """Paired Wilcoxon signed-rank test on per-fold scores."""
    stat, p = wilcoxon(scores_a, scores_b)
    return {"metric": metric, "statistic": stat, "p_value": p}


def mcnemar_test(fold_results_a: list[dict], fold_results_b: list[dict]) -> dict:
    """
    McNemar's test on pooled predictions — tests whether the two models
    make *different* errors, not just different aggregate accuracy.
    """
    y_true = np.concatenate([r["y_test"] for r in fold_results_a])
    pred_a = np.concatenate([r["y_pred"] for r in fold_results_a])
    pred_b = np.concatenate([r["y_pred"] for r in fold_results_b])

    # Contingency table: correct/incorrect agreement matrix
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    table = np.array(
        [
            [(correct_a & correct_b).sum(), (~correct_a & correct_b).sum()],
            [(correct_a & ~correct_b).sum(), (~correct_a & ~correct_b).sum()],
        ]
    )
    result = mcnemar(table, exact=False, correction=True)
    return {"statistic": result.statistic, "p_value": result.pvalue}


def compare_models(
    results_a: list[dict],
    results_b: list[dict],
    name_a: str = "XGB",
    name_b: str = "RF",
) -> pd.DataFrame:
    rows = []
    scores_a = [score_fold(r)["roc_auc"] for r in results_a]
    scores_b = [score_fold(r)["roc_auc"] for r in results_b]

    rows.append({"test": "Wilcoxon (ROC-AUC)", **wilcoxon_test(scores_a, scores_b)})
    rows.append({"test": "McNemar (predictions)", **mcnemar_test(results_a, results_b)})
    return pd.DataFrame(rows)
