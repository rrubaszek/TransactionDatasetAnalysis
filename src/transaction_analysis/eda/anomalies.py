import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def anomaly_analysis(
    user_agg: pd.DataFrame,
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

    return user_agg, n_anomalies
