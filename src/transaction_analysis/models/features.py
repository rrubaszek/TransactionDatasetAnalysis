import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from transaction_analysis.config.paths import FRAUD_DATASET_DIR
from transaction_analysis.eda.aggregations import aggregate_transactions_by_user
from transaction_analysis.eda.utils import load_cleaned_data

logger = logging.getLogger(__name__)
CLEANED_DIR = FRAUD_DATASET_DIR / "cleaned"


def _load_model_data():
    logger.info("Loading data...")
    transactions, users, _ = load_cleaned_data(dataset_dir=CLEANED_DIR)
    fraud_labels = pd.read_parquet(CLEANED_DIR / "fraud_labels.parquet")

    df = transactions.merge(
        fraud_labels.rename(columns={"id": "transaction_id"})[["transaction_id", "fraud"]],
        on="transaction_id",
        how="left",
    )
    df["fraud"] = df["fraud"].fillna(False).astype(int)
    logger.info(f"Transactions: {len(df):,}  |  Fraud: {df['fraud'].sum():,}  |  Fraud rate: {df['fraud'].mean():.4%}")

    return df, transactions, users


def _compute_user_agg(transactions: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing user-level aggregates...")
    user_agg = aggregate_transactions_by_user(transactions, users)
    return user_agg


def _prepare_risk_features(user_agg: pd.DataFrame):
    # Engineer derived risk features on top of the aggregates
    if "amount_std" in user_agg and "amount_mean" in user_agg:
        user_agg["amount_volatility"] = user_agg["amount_std"] / (user_agg["amount_mean"] + 1e-6)
    if "total_debt_usd" in user_agg and "yearly_income_usd" in user_agg:
        user_agg["debt_to_income"] = user_agg["total_debt_usd"] / (user_agg["yearly_income_usd"] + 1e-6)
    if "amount_sum" in user_agg and "txn_count" in user_agg:
        user_agg["avg_txn_size"] = user_agg["amount_sum"] / (user_agg["txn_count"] + 1)
    if "unique_merchants" in user_agg and "txn_count" in user_agg:
        user_agg["merchant_diversity"] = user_agg["unique_merchants"] / (user_agg["txn_count"] + 1)

    return user_agg


def _prepare_eng_features(df: pd.DataFrame) -> pd.DataFrame:
    if "amount_mean" in df.columns and "amount_std" in df.columns:
        df["amount_zscore"] = (df["amount_usd"] - df["amount_mean"]) / (df["amount_std"] + 1e-6)
    return df


def _prepare_columns(user_agg: pd.DataFrame) -> pd.DataFrame:
    # Encode gender if present
    if "gender" in user_agg.columns:
        user_agg["gender"] = LabelEncoder().fit_transform(user_agg["gender"].astype(str))

    # Drop datetime columns — not consumable by XGBoost
    date_cols = [c for c in user_agg.columns if pd.api.types.is_datetime64_any_dtype(user_agg[c])]
    user_agg = user_agg.drop(columns=date_cols)
    return user_agg


def _join_user_agg(df: pd.DataFrame, user_agg: pd.DataFrame) -> pd.DataFrame:
    logger.info("Joining features onto transactions...")
    df = df.merge(user_agg.drop(columns=["txn_count", "amount_sum"], errors="ignore"), on="client_id", how="left")
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["transaction_type", "mcc"]:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def build_features() -> tuple[pd.DataFrame, pd.Series]:
    df, transactions, users = _load_model_data()

    user_agg = _compute_user_agg(transactions, users)

    user_agg = _prepare_risk_features(user_agg)

    user_agg = _prepare_columns(user_agg)

    df = _join_user_agg(df, user_agg)

    df = _prepare_eng_features(df)

    df = _encode_categoricals(df)

    drop_cols = {
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
    feature_cols = [
        c
        for c in df.columns
        if c not in drop_cols
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and not pd.api.types.is_object_dtype(df[c])
    ]

    X = df[feature_cols].fillna(0)
    y = df["fraud"]

    return X, y
