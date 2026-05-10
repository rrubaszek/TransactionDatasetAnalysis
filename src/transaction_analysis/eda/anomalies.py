from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from transaction_analysis.eda.utils import save_figure


def anomaly_analysis(
    user_agg: pd.DataFrame,
    output_dir: Path,
    contamination: float = 0.02,
) -> tuple[pd.DataFrame, int]:

    user_agg = user_agg.copy()
    feature_cols = [
        c for c in user_agg.select_dtypes(include=[np.number]).columns if c not in ("anomaly", "anomaly_score")
    ]

    X_scaled = StandardScaler().fit_transform(user_agg[feature_cols])
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    user_agg["anomaly"] = iso.fit_predict(X_scaled)
    user_agg["anomaly_score"] = iso.decision_function(X_scaled)

    n_anomalies = (user_agg["anomaly"] == -1).sum()
    print(f"Flagged {n_anomalies} anomalous users ({n_anomalies / len(user_agg) * 100:.1f}%)")

    # Anomaly score histogram
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.histplot(user_agg["anomaly_score"], bins=60, color="steelblue", ax=ax)
    ax.axvline(0, color="red", linestyle="--", label="Decision boundary")
    ax.set_title("Isolation Forest Anomaly Score Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Score  (lower = more anomalous)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure("anomaly_scores.png", output_dir)
    plt.close()

    # Violin: avg_amount by anomaly label
    avg_col = "avg_amount" if "avg_amount" in user_agg.columns else "amount_cents_mean"
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.violinplot(
        data=user_agg,
        x="anomaly",
        y=avg_col,
        order=[1, -1],
        ax=ax,
        palette={1: "#69b3a2", -1: "#ff6b6b"},
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_title("Avg Transaction Amount: Normal vs Anomalous Users", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    save_figure("anomaly_amount.png", output_dir)
    plt.close()

    return user_agg, n_anomalies
