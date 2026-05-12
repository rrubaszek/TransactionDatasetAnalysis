"""Utility functions for EDA."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_cleaned_data(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transactions_path = dataset_dir / "transactions.parquet"

    transactions = pd.read_parquet(transactions_path)

    users_path = dataset_dir / "users.parquet"
    cards_path = dataset_dir / "cards.parquet"

    users = pd.read_parquet(users_path) if users_path.exists() else pd.DataFrame()
    cards = pd.read_parquet(cards_path) if cards_path.exists() else pd.DataFrame()

    return transactions, users, cards


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", palette="husl")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10


def save_figure(filename: str, output_dir: Path = Path("plots")) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Saved: {filepath}")
