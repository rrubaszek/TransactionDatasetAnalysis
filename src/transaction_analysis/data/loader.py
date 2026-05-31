import pandas as pd

from transaction_analysis.config.paths import FRAUD_DATASET_DIR


def load_cards(subdirectory: str = "cleaned") -> pd.DataFrame:
    return pd.read_parquet(FRAUD_DATASET_DIR / subdirectory / "cards.parquet")


def load_fraud_labels(subdirectory: str = "cleaned") -> pd.DataFrame:
    return pd.read_parquet(FRAUD_DATASET_DIR / subdirectory / "fraud_labels.parquet")


def load_mcc_codes(subdirectory: str = "cleaned") -> pd.DataFrame:
    return pd.read_parquet(FRAUD_DATASET_DIR / subdirectory / "mcc_codes.parquet")


def load_all_transactions(subdirectory: str = "cleaned") -> pd.DataFrame:
    return pd.read_parquet(FRAUD_DATASET_DIR / subdirectory / "transactions_with_fraud.parquet")


def load_users(subdirectory: str = "cleaned") -> pd.DataFrame:
    return pd.read_parquet(FRAUD_DATASET_DIR / subdirectory / "users.parquet")


def load_fraud_transactions(subdirectory: str = "cleaned") -> pd.DataFrame:
    transactions = load_all_transactions(subdirectory)
    return transactions[transactions["fraud"].fillna(False)]


def load_legit_transactions(subdirectory: str = "cleaned") -> pd.DataFrame:
    transactions = load_all_transactions(subdirectory)
    return transactions[(~transactions["fraud"]).fillna(False)]
