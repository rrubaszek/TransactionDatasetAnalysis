import os
from pathlib import Path
from typing import Literal

import pandas as pd

from transaction_analysis.paths import FRAUD_DATASET_DIR


def currency_to_float32(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].replace(r"[\$,]", "", regex=True).str.strip().astype("float32")
    return df


def str_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def try_downcast(
    df: pd.DataFrame, col: str, downcast: Literal["integer", "signed", "unsigned", "float"]
) -> pd.DataFrame:
    df[col] = pd.to_numeric(df[col], downcast=downcast)
    return df


def str_to_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].astype("category")
    return df


def zip_to_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].astype("string").replace(r"\.\d+$", "", regex=True).str.zfill(5).astype("category")
    return df


def month_year_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], format="%m/%Y", errors="coerce")
    return df


def yes_no_to_bool(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].str.strip().str.upper().map({"YES": True, "NO": False}).astype("boolean")
    return df


def run(dataset_in_dir: Path, dataset_out_dir: Path, force: bool = False) -> None:
    os.makedirs(dataset_out_dir, exist_ok=True)

    def process_transactions(in_file: Path, out_file: Path) -> None:
        if out_file.exists() and not force:
            print("Transactions already preprocessed, skipping. Use `force=True` to re-run.")
            return

        (
            pd.read_csv(in_file)
            .pipe(try_downcast, "id", "unsigned")
            .pipe(str_to_datetime, "date")
            .pipe(try_downcast, "client_id", "unsigned")
            .pipe(try_downcast, "card_id", "unsigned")
            .pipe(currency_to_float32, "amount")
            .pipe(str_to_category, "use_chip")
            .pipe(try_downcast, "merchant_id", "unsigned")
            .pipe(str_to_category, "merchant_city")
            .pipe(str_to_category, "merchant_state")
            .pipe(zip_to_category, "zip")
            .pipe(try_downcast, "mcc", "unsigned")
            .pipe(str_to_category, "errors")
            .to_parquet(out_file)
        )

    def process_users(in_file: Path, out_file: Path) -> None:
        if out_file.exists() and not force:
            print("Users already preprocessed, skipping. Use `force=True` to re-run.")
            return

        (
            pd.read_csv(in_file)
            .pipe(try_downcast, "id", "unsigned")
            .pipe(try_downcast, "current_age", "unsigned")
            .pipe(try_downcast, "retirement_age", "unsigned")
            .pipe(try_downcast, "birth_year", "unsigned")
            .pipe(try_downcast, "birth_month", "unsigned")
            .pipe(str_to_category, "gender")
            .pipe(currency_to_float32, "per_capita_income")
            .pipe(currency_to_float32, "yearly_income")
            .pipe(currency_to_float32, "total_debt")
            .pipe(try_downcast, "credit_score", "signed")
            .pipe(try_downcast, "num_credit_cards", "unsigned")
            .to_parquet(out_file)
        )

    def process_cards(in_file: Path, out_file: Path) -> None:
        if out_file.exists() and not force:
            print("Cards already preprocessed, skipping. Use `force=True` to re-run.")
            return

        (
            pd.read_csv(in_file)
            .pipe(try_downcast, "id", "unsigned")
            .pipe(try_downcast, "client_id", "unsigned")
            .pipe(str_to_category, "card_brand")
            .pipe(str_to_category, "card_type")
            .pipe(month_year_to_datetime, "expires")
            .pipe(try_downcast, "cvv", "unsigned")
            .pipe(yes_no_to_bool, "has_chip")
            .pipe(try_downcast, "num_cards_issued", "unsigned")
            .pipe(currency_to_float32, "credit_limit")
            .pipe(month_year_to_datetime, "acct_open_date")
            .pipe(try_downcast, "year_pin_last_changed", "unsigned")
            .pipe(yes_no_to_bool, "card_on_dark_web")
            .to_parquet(out_file)
        )

    def process_mcc_codes(in_file: Path, out_file: Path) -> None:
        if out_file.exists() and not force:
            print("MCC codes already preprocessed, skipping. Use `force=True` to re-run.")
            return

        (
            pd.read_json(in_file, typ="series")
            .rename_axis("id")
            .rename("description")
            .reset_index()
            .pipe(try_downcast, "id", "unsigned")
            .to_parquet(out_file)
        )

    def process_fraud_labels(in_file: Path, out_file: Path) -> None:
        if out_file.exists() and not force:
            print("Fraud labels already preprocessed, skipping. Use `force=True` to re-run.")
            return

        # fmt: off
        (
            pd.read_json(in_file)
            .rename_axis("id")
            .rename(columns={"target": "fraud"})
            .reset_index()
            .pipe(try_downcast, "id", "unsigned")
            .pipe(yes_no_to_bool, "fraud")
            .to_parquet(out_file)
        )
        # fmt: on

    # fmt: off
    process_transactions(
        dataset_in_dir / "transactions_data.csv",
        dataset_out_dir / "transactions.parquet",
    )
    process_users(
        dataset_in_dir / "users_data.csv",
        dataset_out_dir / "users.parquet",
    )
    process_cards(
        dataset_in_dir / "cards_data.csv",
        dataset_out_dir / "cards.parquet",
    )
    process_mcc_codes(
        dataset_in_dir / "mcc_codes.json",
        dataset_out_dir / "mcc_codes.parquet",
    )
    process_fraud_labels(
        dataset_in_dir / "train_fraud_labels.json",
        dataset_out_dir / "fraud_labels.parquet",
    )
    # fmt: on


if __name__ == "__main__":
    run(
        dataset_in_dir=FRAUD_DATASET_DIR / "raw",
        dataset_out_dir=FRAUD_DATASET_DIR / "preprocessed",
        force=True,
    )
