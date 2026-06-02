import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from transaction_analysis.config.paths import FRAUD_DATASET_DIR
from transaction_analysis.data import loader
from transaction_analysis.eda.aggregations import aggregate_transactions_by_user

logger = logging.getLogger(__name__)
CLEANED_DIR = FRAUD_DATASET_DIR / "cleaned"

DROP_COLS = {
    "transaction_id",
    "client_id",
    "card_id",
    "merchant_id",
    "merchant_city",
    "merchant_state",
    "zip",
    "date",
    "errors",
    "fraud",
}


def _add_velocity_features(transactions: pd.DataFrame) -> pd.DataFrame:
    dates = transactions.sort_values("date").groupby("client_id")["date"]
    inter_txn_hours = dates.apply(lambda x: x.diff().dt.total_seconds().div(3600).dropna())
    velocity_agg = (
        inter_txn_hours.groupby(level=0).agg(inter_txn_hours_mean="mean", inter_txn_hours_std="std").reset_index()
    )
    return velocity_agg


def _add_temporal_features(transactions: pd.DataFrame) -> pd.DataFrame:
    txns = transactions.copy()
    if not pd.api.types.is_datetime64_any_dtype(txns["date"]):
        txns["date"] = pd.to_datetime(txns["date"])

    txns["_hour"] = txns["date"].dt.hour
    txns["_dayofweek"] = txns["date"].dt.dayofweek

    night_ratio = (
        txns.groupby("client_id")["_hour"]
        .apply(lambda x: ((x >= 23) | (x <= 5)).mean(), include_groups=False)
        .reset_index(name="night_txn_ratio")
    )
    weekend_ratio = (
        txns.groupby("client_id")["_dayofweek"]
        .apply(lambda x: x.isin([5, 6]).mean(), include_groups=False)
        .reset_index(name="weekend_txn_ratio")
    )
    return night_ratio.merge(weekend_ratio, on="client_id", how="left")


def _add_geographic_features(transactions: pd.DataFrame) -> pd.DataFrame:
    aggs = []

    if "merchant_state" in transactions.columns:
        top_state_ratio = (
            transactions.groupby("client_id")["merchant_state"]
            .apply(
                lambda x: x.value_counts(normalize=True).iloc[0] if len(x) else 0,
                include_groups=False,
            )
            .reset_index(name="top_state_ratio")
        )
        aggs.append(top_state_ratio)

    if "zip" in transactions.columns:
        unique_zips = transactions.groupby("client_id")["zip"].nunique().reset_index(name="unique_zips")
        aggs.append(unique_zips)

    if not aggs:
        return pd.DataFrame(columns=["client_id"])

    result = aggs[0]
    for df in aggs[1:]:
        result = result.merge(df, on="client_id", how="left")
    return result


def _add_amount_tail_features(transactions: pd.DataFrame) -> pd.DataFrame:
    tail_agg = (
        transactions.groupby("client_id")["amount_usd"]
        .agg(
            amount_p25=lambda x: x.quantile(0.25),
            amount_p75=lambda x: x.quantile(0.75),
            amount_p95=lambda x: x.quantile(0.95),
            large_txn_ratio=lambda x: (x > x.quantile(0.95)).mean(),
        )
        .reset_index()
    )
    tail_agg["amount_iqr"] = tail_agg["amount_p75"] - tail_agg["amount_p25"]
    return tail_agg


def _add_card_utilization_features(transactions: pd.DataFrame, cards: pd.DataFrame) -> pd.DataFrame:
    if cards is None or cards.empty or "credit_limit_usd" not in cards.columns:
        return pd.DataFrame(columns=["client_id"])

    txns_with_limit = transactions.merge(cards[["card_id", "credit_limit_usd"]], on="card_id", how="left")
    utilization = (
        txns_with_limit.groupby("client_id")
        .apply(
            lambda x: (x["amount_usd"] / (x["credit_limit_usd"] + 1e-6)).mean(),
            include_groups=False,
        )
        .reset_index(name="avg_limit_utilization")
    )
    return utilization


def _compute_user_agg(
    transactions: pd.DataFrame,
    users: pd.DataFrame,
    cards: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Computing base user-level aggregates...")
    user_agg = aggregate_transactions_by_user(transactions, users, cards)

    logger.info("Adding velocity features...")
    user_agg = user_agg.merge(_add_velocity_features(transactions), on="client_id", how="left")

    logger.info("Adding temporal features...")
    user_agg = user_agg.merge(_add_temporal_features(transactions), on="client_id", how="left")

    logger.info("Adding geographic features...")
    geo = _add_geographic_features(transactions)
    if "client_id" in geo.columns and len(geo.columns) > 1:
        user_agg = user_agg.merge(geo, on="client_id", how="left")

    logger.info("Adding amount tail features...")
    user_agg = user_agg.merge(_add_amount_tail_features(transactions), on="client_id", how="left")

    logger.info("Adding card utilization features...")
    util = _add_card_utilization_features(transactions, cards)
    if "client_id" in util.columns and len(util.columns) > 1:
        user_agg = user_agg.merge(util, on="client_id", how="left")

    return user_agg


def _prepare_risk_features(user_agg: pd.DataFrame) -> pd.DataFrame:
    cols = set(user_agg.columns)

    # Original ratios
    if {"amount_std", "amount_mean"} <= cols:
        user_agg["amount_volatility"] = user_agg["amount_std"] / (user_agg["amount_mean"] + 1e-6)
    if {"total_debt_usd", "yearly_income_usd"} <= cols:
        user_agg["debt_to_income"] = user_agg["total_debt_usd"] / (user_agg["yearly_income_usd"] + 1e-6)
    if {"amount_sum", "txn_count"} <= cols:
        user_agg["avg_txn_size"] = user_agg["amount_sum"] / (user_agg["txn_count"] + 1)
    if {"unique_merchants", "txn_count"} <= cols:
        user_agg["merchant_diversity"] = user_agg["unique_merchants"] / (user_agg["txn_count"] + 1)

    # New derived ratios
    if {"unique_zips", "txn_count"} <= cols:
        user_agg["zip_diversity"] = user_agg["unique_zips"] / (user_agg["txn_count"] + 1)
    if {"inter_txn_hours_std", "inter_txn_hours_mean"} <= cols:
        user_agg["timing_volatility"] = user_agg["inter_txn_hours_std"] / (user_agg["inter_txn_hours_mean"] + 1e-6)
    if {"amount_p95", "amount_mean"} <= cols:
        user_agg["p95_to_mean_ratio"] = user_agg["amount_p95"] / (user_agg["amount_mean"] + 1e-6)
    if {"amount_iqr", "amount_mean"} <= cols:
        user_agg["iqr_to_mean_ratio"] = user_agg["amount_iqr"] / (user_agg["amount_mean"] + 1e-6)

    return user_agg


def _join_user_agg(df: pd.DataFrame, user_agg: pd.DataFrame) -> pd.DataFrame:
    logger.info("Joining user-level features onto transactions...")
    return df.merge(
        user_agg.drop(columns=["txn_count", "amount_sum"], errors="ignore"),
        on="client_id",
        how="left",
    )


def _prepare_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    # Z-score of this transaction vs user's own distribution
    if {"amount_mean", "amount_std"} <= set(df.columns):
        df["amount_zscore"] = (df["amount_usd"] - df["amount_mean"]) / (df["amount_std"] + 1e-6)

    # How large is this transaction relative to the user's p95?
    if "amount_p95" in df.columns:
        df["amount_vs_p95"] = df["amount_usd"] / (df["amount_p95"] + 1e-6)

    # Hour and day of week of the individual transaction
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"])
        df["txn_hour"] = dt.dt.hour
        df["txn_dayofweek"] = dt.dt.dayofweek
        df["txn_is_night"] = ((dt.dt.hour >= 23) | (dt.dt.hour <= 5)).astype(int)
        df["txn_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)

    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["transaction_type", "mcc", "gender"]:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    return df.drop(columns=date_cols)


def build_features() -> tuple[pd.DataFrame, pd.Series]:
    """
    Full feature engineering pipeline.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix ready for XGBoost / Random Forest (no NaNs, no object cols).
    y : pd.Series
        Binary fraud label (0 / 1).
    """
    logger.info("Loading data...")
    transactions = loader.load_all_transactions()
    users = loader.load_users()
    cards = loader.load_cards()
    fraud_labels = loader.load_fraud_labels()

    df = transactions.merge(
        fraud_labels.rename(columns={"id": "transaction_id"})[["transaction_id", "fraud"]],
        on="transaction_id",
        how="left",
    )
    df["fraud"] = df["fraud"].fillna(False).astype(np.int8)
    logger.info(f"Transactions: {len(df):,}  |  Fraud: {df['fraud'].sum():,}  |  Fraud rate: {df['fraud'].mean():.4%}")

    user_agg = _compute_user_agg(transactions, users, cards)
    user_agg = _prepare_risk_features(user_agg)

    df = _join_user_agg(df, user_agg)
    df = _prepare_transaction_features(df)
    df = _encode_categoricals(df)

    feature_cols = [
        c
        for c in df.columns
        if c not in DROP_COLS
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and not pd.api.types.is_object_dtype(df[c])
    ]

    logger.info(f"Final feature set: {len(feature_cols)} columns — {feature_cols}")

    X = df[feature_cols].fillna(0)
    # Downcast integers
    int_cols = X.select_dtypes("int64").columns
    X[int_cols] = X[int_cols].apply(pd.to_numeric, downcast="integer")

    # Downcast floats to float32
    float_cols = X.select_dtypes("float64").columns
    X[float_cols] = X[float_cols].astype(np.float16)

    y = df["fraud"].astype(np.int8)

    return X, y
