import numpy as np
import pandas as pd
from scipy import stats

PSI_NO_DRIFT = 0.1
PSI_MODERATE = 0.25


def _to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind == "M":
        return series.astype("int64")
    if series.dtype == object:
        sample = series.dropna()
        if len(sample) and hasattr(sample.iloc[0], "toordinal"):  # datetime.date / datetime
            return pd.to_datetime(series).astype("int64")
    return series


def psi(old: pd.Series, new: pd.Series, bins: int = 10, *, categorical: bool = False) -> float:
    old = _to_numeric(old).dropna()
    new = _to_numeric(new).dropna()

    if categorical:
        categories = pd.Index(sorted(set(old.unique()) | set(new.unique()), key=str))
        old_pct = old.value_counts(normalize=True).reindex(categories).fillna(0.0).to_numpy()
        new_pct = new.value_counts(normalize=True).reindex(categories).fillna(0.0).to_numpy()
    else:
        edges = np.unique(np.quantile(old, np.linspace(0, 1, bins + 1))).astype(float)
        edges[0], edges[-1] = -np.inf, np.inf
        old_pct = np.histogram(old, edges)[0] / len(old)
        new_pct = np.histogram(new, edges)[0] / len(new)

    eps = 1e-6
    old_pct = np.clip(old_pct, eps, None)
    new_pct = np.clip(new_pct, eps, None)
    return float(np.sum((new_pct - old_pct) * np.log(new_pct / old_pct)))


def ks_drift(old: pd.Series, new: pd.Series) -> tuple[float, float]:
    result = stats.ks_2samp(_to_numeric(old).dropna(), _to_numeric(new).dropna())
    return float(result.statistic), float(result.pvalue)


def cramers_v(old: pd.Series, new: pd.Series) -> tuple[float, float, float]:
    contingency = (
        pd.DataFrame({"old": old.dropna().value_counts(), "new": new.dropna().value_counts()}).fillna(0.0).astype(float)
    )
    chi2, pvalue, _, _ = stats.chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    k = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    v = float(np.sqrt(chi2 / (n * k))) if k > 0 else float("nan")
    return float(chi2), float(pvalue), v


def _verdict(psi_value: float) -> str:
    if psi_value < PSI_NO_DRIFT:
        return "znikomy drift"
    if psi_value < PSI_MODERATE:
        return "średni drift"
    return "duży drift"


def drift_report(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    *,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> pd.DataFrame:
    numeric_cols = numeric_cols or []
    categorical_cols = categorical_cols or []
    rows = []

    for column in numeric_cols:
        ks_d, ks_p = ks_drift(df_old[column], df_new[column])
        psi_value = psi(df_old[column], df_new[column])
        rows.append(
            {
                "column": column,
                "type": "numeric",
                "ks_stat": ks_d,
                "ks_pvalue": ks_p,
                "chi2_stat": pd.NA,
                "chi2_pvalue": pd.NA,
                "cramers_v": pd.NA,
                "psi": psi_value,
                "verdict": _verdict(psi_value),
            }
        )

    for column in categorical_cols:
        chi2, chi2_p, v = cramers_v(df_old[column], df_new[column])
        psi_value = psi(df_old[column], df_new[column], categorical=True)
        rows.append(
            {
                "column": column,
                "type": "categorical",
                "ks_stat": pd.NA,
                "ks_pvalue": pd.NA,
                "chi2_stat": chi2,
                "chi2_pvalue": chi2_p,
                "cramers_v": v,
                "psi": psi_value,
                "verdict": _verdict(psi_value),
            }
        )

    return pd.DataFrame(rows)
