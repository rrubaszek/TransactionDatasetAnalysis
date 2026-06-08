import logging
from dataclasses import dataclass

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
    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: np.ndarray
    feature_matrix: pd.DataFrame
    raw_features: pd.DataFrame
    fraud_rate: pd.Series | None
    clusters: pd.Series | None
    pca: PCA
    scaler: StandardScaler


def build_user_pca_features(
    transactions: pd.DataFrame,
    users: pd.DataFrame,
    cards: pd.DataFrame,
) -> pd.DataFrame:
    client_transactions = transactions.groupby("client_id", observed=True)
    transactions_agg: pd.DataFrame = client_transactions.agg(
        txn_count=("transaction_id", "count"),
        amount_sum=("amount_usd", "sum"),
        amount_mean=("amount_usd", "mean"),
        amount_median=("amount_usd", "median"),
        amount_std=("amount_usd", "std"),
        amount_max=("amount_usd", "max"),
        unique_merchants=("merchant_id", "nunique"),
        unique_mcc=("mcc", "nunique"),
    )

    transaction_type_share = (
        transactions.groupby(["client_id", "transaction_type"], observed=True).size().unstack(fill_value=0)
    )
    client_transaction_count = transaction_type_share.sum(axis=1)
    transaction_type_share = transaction_type_share.div(client_transaction_count, axis=0).fillna(0.0)

    transaction_type_share.columns = [f"share_{str(c).split()[0].lower()}" for c in transaction_type_share.columns]
    transactions_agg = transactions_agg.join(transaction_type_share, how="left")

    transactions_agg["refund_share"] = client_transactions["amount_usd"].apply(lambda s: (s < 0).mean())
    transactions_agg["error_rate"] = client_transactions["errors"].apply(lambda s: s.notna().mean())

    hours = transactions["date"].dt.hour
    day_of_week = transactions["date"].dt.dayofweek
    transactions_agg["night_share"] = (
        ((hours < 6) | (hours >= 22)).astype("int8").groupby(transactions["client_id"], observed=True).mean()
    )
    transactions_agg["weekend_share"] = (
        (day_of_week >= 5).astype("int8").groupby(transactions["client_id"], observed=True).mean()
    )

    mcc_counts = transactions.groupby(["client_id", "mcc"], observed=True).size().unstack(fill_value=0)
    client_mcc_sum = mcc_counts.sum(axis=1)
    mcc_share = mcc_counts.div(client_mcc_sum, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(mcc_share * np.log(mcc_share.where(mcc_share > 0))).sum(axis=1)
    transactions_agg["mcc_entropy"] = entropy

    users["birth_date"] = pd.to_datetime(users["birth_date"], errors="coerce")
    today_date_ref = transactions["date"].max()
    users["age"] = ((today_date_ref - users["birth_date"]).dt.days / 365.25).astype("int8")
    users_demographic_cols = [
        "yearly_income_usd",
        "per_capita_income_usd",
        "total_debt_usd",
        "credit_score",
        "num_credit_cards",
        "age",
        "retirement_age",
    ]
    transactions_agg = transactions_agg.join(users.set_index("id")[users_demographic_cols], how="left")

    cards["acct_open_date"] = pd.to_datetime(cards["acct_open_date"], errors="coerce")
    today_date_ref = transactions["date"].max()
    cards["card_age_years"] = ((today_date_ref - cards["acct_open_date"]).dt.days / 365.25).astype("int8")
    card_agg = cards.groupby("client_id").agg(
        num_cards=("card_id", "count"),
        avg_credit_limit=("credit_limit_usd", "mean"),
        total_credit_limit=("credit_limit_usd", "sum"),
        chip_share=("has_chip", "mean"),
        avg_card_age_years=("card_age_years", "mean"),
    )
    transactions_agg = transactions_agg.join(card_agg, how="left")

    transactions_agg["debt_to_income"] = transactions_agg["total_debt_usd"].div(transactions_agg["yearly_income_usd"])
    return transactions_agg


def build_client_fraud_profile(
    transactions: pd.DataFrame,
    users: pd.DataFrame,
    cards: pd.DataFrame,
    min_fraud_txns: int = 2,
) -> pd.DataFrame:
    features = build_user_pca_features(transactions, users, cards).copy()

    fraud_flag = transactions["fraud"].fillna(False).astype(bool)
    fraud_count = fraud_flag.groupby(transactions["client_id"], observed=True).sum()

    features["fraud_count"] = fraud_count.reindex(features.index).fillna(0).astype(int)
    features["is_fraudster"] = features["fraud_count"] >= min_fraud_txns
    return features


def _prepare_matrix(features: pd.DataFrame) -> pd.DataFrame:
    X: pd.DataFrame = features.copy().astype(float)

    for col in MONETARY_COLS:
        if col in X.columns:
            X[col] = np.sign(X[col]) * np.log1p(X[col].abs())

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)

    variances = X.var(axis=0)
    non_zero_variance_cols = variances[variances > 1e-12].index
    dropped = sorted(set(X.columns) - set(non_zero_variance_cols))
    if dropped:
        logger.info("Dropping zero-variance columns from PCA input: %s", dropped)
    return X[non_zero_variance_cols]


def compute_user_fraud_rate(
    transactions: pd.DataFrame,
    fraud_labels: pd.DataFrame,
) -> pd.Series:
    if fraud_labels is None or fraud_labels.empty:
        return pd.Series(dtype="float32", name="fraud_rate")

    merged = transactions[["transaction_id", "client_id"]].merge(
        fraud_labels,
        left_on="transaction_id",
        right_on="id",
        how="inner",
    )
    user_fraud_rate: pd.Series = merged.groupby("client_id")["fraud"].mean().astype("float32")
    user_fraud_rate.name = "fraud_rate"
    return user_fraud_rate


def run_user_pca(
    transactions: pd.DataFrame,
    users: pd.DataFrame,
    cards: pd.DataFrame,
    fraud_labels: pd.DataFrame,
    n_components: int = 6,
    n_clusters: int = 4,
    random_state: int = 42,
) -> PCAResult:
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
