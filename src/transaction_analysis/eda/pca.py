import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


MONETARY_COLS = (
    "amount_sum",
    "amount_mean",
    "amount_median",
    "amount_std",
    "amount_max",
    "yearly_income_usd",
    "per_capita_income_usd",
    "total_debt_usd",
    "avg_credit_limit",
)


@dataclass
class PCAResult:
    scores: pd.DataFrame  # (n_users, n_components) PC scores, indexed by client_id
    loadings: pd.DataFrame  # (n_features, n_components)
    explained_variance: np.ndarray  # variance ratio per component
    feature_matrix: pd.DataFrame  # the standardized matrix used as PCA input (debugging/reuse)
    raw_features: pd.DataFrame  # pre-scaling, post-engineering features (indexed by client_id)
    fraud_rate: pd.Series | None  # per-user fraud rate, if labels provided
    clusters: pd.Series | None  # KMeans cluster labels on first k PCs
    pca: PCA
    scaler: StandardScaler


def build_user_pca_features(
    transactions: pd.DataFrame,
    users: pd.DataFrame,
    cards: pd.DataFrame,
) -> pd.DataFrame:
    """Engineer a wide numeric matrix indexed by client_id, ready for PCA.

    Combines spend intensity, channel mix, diversity, temporal rhythm, error
    behavior, demographics, and card-portfolio features. Designed to be robust
    to missing optional fields.
    """
    # TODO drop tx
    # TODO drop frequency column, first_txn, last_txn
    # TODO is entropy for mcc a real deal?
    tx = transactions
    # base spend aggregates ------------------------------------------------
    grouped = tx.groupby("client_id", observed=True)
    base = grouped.agg(
        txn_count=("transaction_id", "count"),
        amount_sum=("amount_usd", "sum"),
        amount_mean=("amount_usd", "mean"),
        amount_median=("amount_usd", "median"),
        amount_std=("amount_usd", "std"),
        amount_max=("amount_usd", "max"),
        unique_merchants=("merchant_id", "nunique"),
        unique_mcc=("mcc", "nunique"),
        first_txn=("date", "min"),
        last_txn=("date", "max"),
    )
    base["txn_span_days"] = (base["last_txn"] - base["first_txn"]).dt.days.clip(lower=1)
    base["txn_frequency"] = base["txn_count"] / base["txn_span_days"]
    base = base.drop(columns=["first_txn", "last_txn"])

    # channel mix -----------------------------------------------------------
    if "transaction_type" in tx.columns:
        ttype = tx.groupby(["client_id", "transaction_type"], observed=True).size().unstack(fill_value=0)
        print()
        ttype = ttype.div(ttype.sum(axis=1), axis=0).fillna(0.0)
        ttype.columns = [f"share_{str(c).split()[0].lower()}" for c in ttype.columns]
        base = base.join(ttype, how="left")

    # refund and error signals ---------------------------------------------
    base["refund_share"] = grouped.apply(lambda g: (g["amount_usd"] < 0).mean(), include_groups=False)
    if "errors" in tx.columns:
        base["error_rate"] = grouped.apply(lambda g: g["errors"].notna().mean(), include_groups=False)

    # temporal rhythm -------------------------------------------------------
    hours = tx["date"].dt.hour
    dow = tx["date"].dt.dayofweek
    base["night_share"] = (
        tx.assign(_n=((hours < 6) | (hours >= 22)).astype("int8")).groupby("client_id", observed=True)["_n"].mean()
    )
    base["weekend_share"] = tx.assign(_w=(dow >= 5).astype("int8")).groupby("client_id", observed=True)["_w"].mean()

    # mcc diversity (Shannon entropy of spend distribution across MCCs) ----
    mcc_counts = tx.groupby(["client_id", "mcc"], observed=True).size().unstack(fill_value=0)
    mcc_share = mcc_counts.div(mcc_counts.sum(axis=1), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(mcc_share * np.log(mcc_share.where(mcc_share > 0))).sum(axis=1)
    base["mcc_entropy"] = entropy

    # demographics ----------------------------------------------------------
    if users is not None and not users.empty:
        u = users.copy()
        if "birth_date" in u.columns:
            u["birth_date"] = pd.to_datetime(u["birth_date"], errors="coerce")
            ref = tx["date"].max()
            u["age"] = ((ref - u["birth_date"]).dt.days / 365.25).astype("float32")
        demo_cols = [
            c
            for c in (
                "yearly_income_usd",
                "per_capita_income_usd",
                "total_debt_usd",
                "credit_score",
                "num_credit_cards",
                "age",
                "retirement_age",
            )
            if c in u.columns
        ]
        base = base.join(u.set_index("id")[demo_cols], how="left")

    # card portfolio --------------------------------------------------------
    if cards is not None and not cards.empty:
        c = cards.copy()
        c["acct_open_date"] = pd.to_datetime(c["acct_open_date"], errors="coerce")
        ref = tx["date"].max()
        c["card_age_years"] = ((ref - c["acct_open_date"]).dt.days / 365.25).astype("float32")
        card_agg = c.groupby("client_id").agg(
            num_cards=("card_id", "count"),
            avg_credit_limit=("credit_limit_usd", "mean"),
            total_credit_limit=("credit_limit_usd", "sum"),
            chip_share=("has_chip", "mean"),
            avg_card_age_years=("card_age_years", "mean"),
        )
        base = base.join(card_agg, how="left")

    # derived ratios --------------------------------------------------------
    if {"total_debt_usd", "yearly_income_usd"}.issubset(base.columns):
        base["debt_to_income"] = base["total_debt_usd"] / base["yearly_income_usd"].replace(0, np.nan)

    base.index.name = "client_id"
    return base


def _prepare_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Log-transform monetary columns, fill NaN, drop zero-variance cols."""
    X = features.copy().astype(float)

    for col in MONETARY_COLS:
        if col in X.columns:
            X[col] = np.sign(X[col]) * np.log1p(X[col].abs())

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)

    variances = X.var(axis=0)
    keep = variances[variances > 1e-12].index
    dropped = sorted(set(X.columns) - set(keep))
    if dropped:
        logger.info("Dropping zero-variance columns from PCA input: %s", dropped)
    return X[keep]


def compute_user_fraud_rate(
    transactions: pd.DataFrame,
    fraud_labels: pd.DataFrame,
) -> pd.Series:
    """Per-client fraud rate from transaction-level labels.

    Joins fraud_labels (id, fraud) onto transactions and averages per client_id.
    """
    if fraud_labels is None or fraud_labels.empty:
        return pd.Series(dtype="float32", name="fraud_rate")
    merged = transactions[["transaction_id", "client_id"]].merge(
        fraud_labels.rename(columns={"id": "transaction_id"}),
        on="transaction_id",
        how="inner",
    )
    return merged.groupby("client_id")["fraud"].mean().astype("float32").rename("fraud_rate")


def run_user_pca(
    transactions: pd.DataFrame,
    users: pd.DataFrame,
    cards: pd.DataFrame,
    fraud_labels: pd.DataFrame | None = None,
    n_components: int = 6,
    n_clusters: int = 4,
    random_state: int = 42,
) -> PCAResult:
    """End-to-end user-level PCA: build features, scale, fit, cluster."""
    raw = build_user_pca_features(transactions, users, cards)
    prepared = _prepare_matrix(raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(prepared.values)
    X_scaled_df = pd.DataFrame(X_scaled, index=prepared.index, columns=prepared.columns)

    n_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=random_state)
    scores_arr = pca.fit_transform(X_scaled)

    pc_names = [f"PC{i + 1}" for i in range(n_components)]
    scores = pd.DataFrame(scores_arr, index=prepared.index, columns=pc_names)
    loadings = pd.DataFrame(pca.components_.T, index=prepared.columns, columns=pc_names)

    fraud_rate = None
    if fraud_labels is not None and not fraud_labels.empty:
        fraud_rate = compute_user_fraud_rate(transactions, fraud_labels).reindex(scores.index)

    k = min(n_clusters, len(scores))
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    clusters = pd.Series(
        km.fit_predict(scores.values),
        index=scores.index,
        name="cluster",
    )

    logger.info(
        "User PCA: n_users=%d, n_features=%d, components=%d, cum_var(2)=%.2f, cum_var(all)=%.2f",
        len(scores),
        prepared.shape[1],
        n_components,
        pca.explained_variance_ratio_[:2].sum(),
        pca.explained_variance_ratio_.sum(),
    )

    return PCAResult(
        scores=scores,
        loadings=loadings,
        explained_variance=pca.explained_variance_ratio_,
        feature_matrix=X_scaled_df,
        raw_features=raw,
        fraud_rate=fraud_rate,
        clusters=clusters,
        pca=pca,
        scaler=scaler,
    )


def load_fraud_labels(dataset_dir: Path) -> pd.DataFrame:
    """Load fraud_labels.parquet if present; otherwise return empty frame."""
    path = dataset_dir / "fraud_labels.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
