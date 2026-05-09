"""Visualization functions for EDA."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from transaction_analysis.eda.utils import save_figure


def plot_amount_distribution(
    transactions: pd.DataFrame,
    output_dir: Path,
    bins: int = 50,
) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw distribution
    amounts = pd.to_numeric(transactions["amount"], errors="coerce").dropna()
    axes[0].hist(amounts, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].set_title("Transaction Amount Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Amount ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(alpha=0.3)

    # Log-transformed distribution
    log_amounts = np.log1p(amounts)
    axes[1].hist(log_amounts, bins=bins, color="teal", edgecolor="black", alpha=0.7)
    axes[1].set_title("Log-Transformed Amount Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("log(1 + Amount)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_figure("amount_distribution.png", output_dir)
    plt.close()


def plot_transactions_over_time(
    transactions: pd.DataFrame,
    output_dir: Path,
) -> None:

    transactions["date"] = pd.to_datetime(transactions["date"])

    daily_stats = (
        transactions.groupby(transactions["date"].dt.date)
        .agg(
            {
                "id": "count",
                "amount": "mean",
            }
        )
        .reset_index()
    )
    daily_stats.columns = ["date", "count", "avg_amount"]
    daily_stats["date"] = pd.to_datetime(daily_stats["date"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Transaction count over time
    axes[0].plot(daily_stats["date"], daily_stats["count"], marker="o", linewidth=1.5, color="steelblue")
    axes[0].set_title("Daily Transaction Volume", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.3)

    # Average amount over time
    axes[1].plot(daily_stats["date"], daily_stats["avg_amount"], marker="o", linewidth=1.5, color="coral")
    axes[1].set_title("Daily Average Transaction Amount", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Average Amount ($)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_figure("transactions_over_time.png", output_dir)
    plt.close()


def plot_top_merchants(
    merchant_agg: pd.DataFrame,
    output_dir: Path,
    top_n: int = 15,
) -> None:

    top_merchants = merchant_agg.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(top_merchants)), top_merchants["txn_count"], color="steelblue")
    ax.set_yticks(range(len(top_merchants)))
    ax.set_yticklabels([f"Merchant {m}" for m in top_merchants["merchant_id"]])
    ax.set_xlabel("Transaction Count")
    ax.set_title(f"Top {top_n} Merchants by Transaction Volume", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    save_figure(f"top_{top_n}_merchants.png", output_dir)
    plt.close()


def plot_top_mcc(
    mcc_agg: pd.DataFrame,
    output_dir: Path,
    top_n: int = 15,
) -> None:

    top_mcc = mcc_agg.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(top_mcc)), top_mcc["txn_count"], color="teal", edgecolor="black", alpha=0.7)
    ax.set_xticks(range(len(top_mcc)))
    ax.set_xticklabels(top_mcc["mcc"], rotation=45, ha="right")
    ax.set_ylabel("Transaction Count")
    ax.set_title(f"Top {top_n} MCC Codes by Transaction Volume", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(f"top_{top_n}_mcc.png", output_dir)
    plt.close()


def plot_user_transaction_distribution(
    user_agg: pd.DataFrame,
    output_dir: Path,
) -> None:

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Transaction count distribution
    axes[0, 0].hist(user_agg["txn_count"], bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Distribution of Transaction Count per User", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Transaction Count")
    axes[0, 0].set_ylabel("Number of Users")
    axes[0, 0].grid(alpha=0.3)

    # Total amount distribution
    if "amount_sum" in user_agg.columns:
        axes[0, 1].hist(user_agg["amount_sum"], bins=50, color="coral", edgecolor="black", alpha=0.7)
        axes[0, 1].set_title("Distribution of Total Amount per User", fontsize=11, fontweight="bold")
        axes[0, 1].set_xlabel("Total Amount ($)")
        axes[0, 1].set_ylabel("Number of Users")
        axes[0, 1].grid(alpha=0.3)

    # Average amount distribution
    if "amount_mean" in user_agg.columns:
        axes[1, 0].hist(user_agg["amount_mean"], bins=50, color="teal", edgecolor="black", alpha=0.7)
        axes[1, 0].set_title("Distribution of Average Amount per User", fontsize=11, fontweight="bold")
        axes[1, 0].set_xlabel("Average Amount ($)")
        axes[1, 0].set_ylabel("Number of Users")
        axes[1, 0].grid(alpha=0.3)

    # Transaction frequency
    if "txn_frequency" in user_agg.columns:
        axes[1, 1].hist(user_agg["txn_frequency"], bins=50, color="green", edgecolor="black", alpha=0.7)
        axes[1, 1].set_title("Distribution of Transaction Frequency per User", fontsize=11, fontweight="bold")
        axes[1, 1].set_xlabel("Transactions per Day")
        axes[1, 1].set_ylabel("Number of Users")
        axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    save_figure("user_transaction_distribution.png", output_dir)
    plt.close()


def plot_error_analysis(
    transactions: pd.DataFrame,
    output_dir: Path,
) -> None:

    if "errors" not in transactions.columns:
        return

    # Error rate over time
    transactions["date"] = pd.to_datetime(transactions["date"])
    daily_errors = (
        transactions.groupby(transactions["date"].dt.date)
        .apply(lambda x: (x["errors"].notna().sum() / len(x)) * 100, include_groups=False)
        .reset_index()
    )
    daily_errors.columns = ["date", "error_rate_pct"]
    daily_errors["date"] = pd.to_datetime(daily_errors["date"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error rate over time
    axes[0].plot(daily_errors["date"], daily_errors["error_rate_pct"], marker="o", linewidth=1.5, color="red")
    axes[0].set_title("Daily Error Rate", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Error Rate (%)")
    axes[0].grid(alpha=0.3)

    # Top error types
    error_counts = transactions[transactions["errors"].notna()]["errors"].value_counts().head(10)
    axes[1].barh(range(len(error_counts)), error_counts.values, color="salmon", edgecolor="black", alpha=0.7)
    axes[1].set_yticks(range(len(error_counts)))
    axes[1].set_yticklabels(error_counts.index, fontsize=9)
    axes[1].set_xlabel("Count")
    axes[1].set_title("Top 10 Error Types", fontsize=12, fontweight="bold")
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    save_figure("error_analysis.png", output_dir)
    plt.close()


def plot_correlation_heatmap(
    user_agg: pd.DataFrame,
    output_dir: Path,
) -> None:

    # Select numeric columns
    numeric_cols = user_agg.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return

    corr_matrix = user_agg[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Correlation"},
    )
    ax.set_title("Correlation Matrix - User Aggregated Features", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_figure("correlation_heatmap.png", output_dir)
    plt.close()


def plot_demographic_patterns(
    user_agg: pd.DataFrame,
    output_dir: Path,
) -> None:

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Transaction count by gender (if available)
    if "gender" in user_agg.columns and "txn_count" in user_agg.columns:
        gender_stats = user_agg.groupby("gender")["txn_count"].describe()
        axes[0, 0].bar(gender_stats.index, gender_stats["mean"], color="steelblue", alpha=0.7, edgecolor="black")
        axes[0, 0].set_title("Average Transaction Count by Gender", fontsize=11, fontweight="bold")
        axes[0, 0].set_ylabel("Average Count")
        axes[0, 0].grid(alpha=0.3, axis="y")

    # Average amount by credit score (if available)
    if "credit_score" in user_agg.columns and "amount_mean" in user_agg.columns:
        # Create score bins
        user_agg_copy = user_agg.copy()
        user_agg_copy["score_bin"] = pd.cut(user_agg_copy["credit_score"], bins=5)
        score_stats = user_agg_copy.groupby("score_bin")["amount_mean"].mean()
        axes[0, 1].bar(range(len(score_stats)), score_stats.values, color="coral", alpha=0.7, edgecolor="black")
        axes[0, 1].set_xticks(range(len(score_stats)))
        axes[0, 1].set_xticklabels([f"Bin {i + 1}" for i in range(len(score_stats))], rotation=45)
        axes[0, 1].set_title("Average Amount by Credit Score Bins", fontsize=11, fontweight="bold")
        axes[0, 1].set_ylabel("Average Amount ($)")
        axes[0, 1].grid(alpha=0.3, axis="y")

    # Number of credit cards
    if "num_credit_cards" in user_agg.columns and "amount_sum" in user_agg.columns:
        card_stats = user_agg.groupby("num_credit_cards")["amount_sum"].mean().head(10)
        axes[1, 0].bar(range(len(card_stats)), card_stats.values, color="teal", alpha=0.7, edgecolor="black")
        axes[1, 0].set_xticks(range(len(card_stats)))
        axes[1, 0].set_xticklabels(card_stats.index, rotation=45)
        axes[1, 0].set_title("Total Spending by Number of Credit Cards", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Total Amount ($)")
        axes[1, 0].grid(alpha=0.3, axis="y")

    # Age-based patterns (if available)
    if "current_age" in user_agg.columns and "amount_mean" in user_agg.columns:
        age_bins = pd.cut(user_agg["current_age"], bins=6)
        age_stats = user_agg.groupby(age_bins)["amount_mean"].mean()
        axes[1, 1].bar(range(len(age_stats)), age_stats.values, color="green", alpha=0.7, edgecolor="black")
        axes[1, 1].set_xticks(range(len(age_stats)))
        axes[1, 1].set_xticklabels([f"Bin {i + 1}" for i in range(len(age_stats))], rotation=45)
        axes[1, 1].set_title("Average Amount by Age Bins", fontsize=11, fontweight="bold")
        axes[1, 1].set_ylabel("Average Amount ($)")
        axes[1, 1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure("demographic_patterns.png", output_dir)
    plt.close()
