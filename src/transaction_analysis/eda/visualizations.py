from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from transaction_analysis.eda.geoanalysis import (
    load_us_geometry,
    prepare_us_transaction_geo_data,
    scale_bubbles,
)
from transaction_analysis.eda.utils import save_figure


def plot_anomalies(user_agg: pd.DataFrame, output_dir: Path) -> None:
    # Anomaly score histogram
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.histplot(user_agg["anomaly_score"], bins=60, color="steelblue", ax=ax)
    ax.axvline(0, color="red", linestyle="--", label="Decision boundary")
    ax.set_title("Isolation Forest Anomaly Score Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Score  (lower = more anomalous)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure("anomaly_scores.png", output_dir)
    plt.close()

    # Violin: avg_amount by anomaly label
    avg_col = "avg_amount" if "avg_amount" in user_agg.columns else "amount_mean"
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.violinplot(
        data=user_agg,
        x="anomaly",
        y=avg_col,
        order=[1, -1],
        ax=ax,
        palette={1: "#69b3a2", -1: "#ff6b6b"},
        hue="anomaly",
        legend=False,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_title("Avg Transaction Amount: Normal vs Anomalous Users", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    save_figure("anomaly_amount.png", output_dir)
    plt.close()


def plot_amount_distribution(
    transactions: pd.DataFrame,
    output_dir: Path,
    bins: int = 80,
) -> None:
    """Raw and log-transformed transaction amount distributions."""
    amounts = pd.to_numeric(transactions["amount_usd"], errors="coerce").dropna()
    amounts = amounts[amounts > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(amounts, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].set_title("Transaction Amount Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Amount ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(alpha=0.3)

    axes[1].hist(np.log1p(amounts), bins=bins, color="teal", edgecolor="black", alpha=0.7)
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
    """Monthly transaction volume over time (line chart)."""
    transactions = transactions.copy()
    transactions["date"] = pd.to_datetime(transactions["date"])
    transactions["period"] = transactions["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        transactions.groupby("period")
        .agg(txn_count=("transaction_id", "count"), avg_amount=("amount_usd", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly["period"], monthly["txn_count"], marker="o", linewidth=1.5, color="steelblue")
    ax.set_title("Monthly Transaction Volume", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Transactions")
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_figure("transactions_over_time.png", output_dir)
    plt.close()


def plot_time_patterns(
    transactions: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Transaction counts by hour of day and day of week."""
    transactions = transactions.copy()
    transactions["date"] = pd.to_datetime(transactions["date"])
    transactions["txn_hour"] = transactions["date"].dt.hour
    transactions["txn_dow"] = transactions["date"].dt.dayofweek  # 0=Mon, 6=Sun

    hourly = transactions.groupby("txn_hour").size()
    daily = transactions.groupby("txn_dow").size()
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    hourly.plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Transactions by Hour of Day", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.3, axis="y")

    daily.rename(index=dict(enumerate(dow_labels))).plot(kind="bar", ax=axes[1], color="coral")
    axes[1].set_title("Transactions by Day of Week", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Day of Week")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure("time_patterns.png", output_dir)
    plt.close()


def plot_errors_and_darkweb(
    transactions: pd.DataFrame,
    cards: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Top transaction error types (barh) and card-on-dark-web prevalence (pie)."""
    error_counts = transactions[transactions["errors"].notna()]["errors"].value_counts().head(10)

    dark_web = cards["card_on_dark_web"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(range(len(error_counts)), error_counts.values, color="salmon", edgecolor="black", alpha=0.7)
    axes[0].set_yticks(range(len(error_counts)))
    axes[0].set_yticklabels(error_counts.index, fontsize=9)
    axes[0].set_xlabel("Count")
    axes[0].set_title("Top Transaction Error Types", fontsize=12, fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].grid(alpha=0.3, axis="x")

    dark_web.plot(
        kind="pie",
        ax=axes[1],
        autopct="%1.1f%%",
        colors=["#69b3a2", "#ff6b6b"],
        startangle=90,
    )
    axes[1].set_title("Card on Dark Web", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("")

    plt.tight_layout()
    save_figure("errors_and_darkweb.png", output_dir)
    plt.close()


def plot_credit_score_by_gender(
    users: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Credit score KDE distributions split by gender."""
    user_scores = users[users["credit_score"].notna()]

    fig, ax = plt.subplots(figsize=(14, 5))
    for gender, grp in user_scores.groupby("gender"):
        sns.kdeplot(grp["credit_score"], ax=ax, label=gender, fill=True, alpha=0.4)
    ax.set_title("Credit Score Distribution by Gender", fontsize=12, fontweight="bold")
    ax.set_xlabel("Credit Score")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure("credit_score_by_gender.png", output_dir)
    plt.close()


def plot_top_merchants(
    merchant_agg: pd.DataFrame,
    output_dir: Path,
    top_n: int = 15,
) -> None:
    """Top merchants by transaction volume."""
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
    """Top MCC codes by transaction volume."""
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
    """Per-user distributions: txn count, total amount, avg amount, frequency."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(user_agg["txn_count"], bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Distribution of Transaction Count per User", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Transaction Count")
    axes[0, 0].set_ylabel("Number of Users")
    axes[0, 0].grid(alpha=0.3)

    if "amount_sum" in user_agg.columns:
        axes[0, 1].hist(user_agg["amount_sum"], bins=50, color="coral", edgecolor="black", alpha=0.7)
        axes[0, 1].set_title("Distribution of Total Amount per User", fontsize=11, fontweight="bold")
        axes[0, 1].set_xlabel("Total Amount ($)")
        axes[0, 1].set_ylabel("Number of Users")
        axes[0, 1].grid(alpha=0.3)

    if "amount_mean" in user_agg.columns:
        axes[1, 0].hist(user_agg["amount_mean"], bins=50, color="teal", edgecolor="black", alpha=0.7)
        axes[1, 0].set_title("Distribution of Average Amount per User", fontsize=11, fontweight="bold")
        axes[1, 0].set_xlabel("Average Amount ($)")
        axes[1, 0].set_ylabel("Number of Users")
        axes[1, 0].grid(alpha=0.3)

    if "txn_frequency" in user_agg.columns:
        axes[1, 1].hist(user_agg["txn_frequency"], bins=50, color="green", edgecolor="black", alpha=0.7)
        axes[1, 1].set_title("Distribution of Transaction Frequency per User", fontsize=11, fontweight="bold")
        axes[1, 1].set_xlabel("Transactions per Day")
        axes[1, 1].set_ylabel("Number of Users")
        axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    save_figure("user_transaction_distribution.png", output_dir)
    plt.close()


def plot_correlation_heatmap(
    user_agg: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Correlation heatmap of per-user aggregate features."""
    numeric_cols = user_agg.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        user_agg[numeric_cols].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Correlation"},
    )
    ax.set_title("Correlation Matrix — User Aggregated Features", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_figure("correlation_heatmap.png", output_dir)
    plt.close()


def plot_demographic_patterns(
    user_agg: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Transaction patterns broken down by gender, credit score, card count, and age."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if "gender" in user_agg.columns and "txn_count" in user_agg.columns:
        gender_stats = user_agg.groupby("gender")["txn_count"].mean()
        axes[0, 0].bar(gender_stats.index, gender_stats.values, color="steelblue", alpha=0.7, edgecolor="black")
        axes[0, 0].set_title("Average Transaction Count by Gender", fontsize=11, fontweight="bold")
        axes[0, 0].set_ylabel("Average Count")
        axes[0, 0].grid(alpha=0.3, axis="y")

    if "credit_score" in user_agg.columns and "amount_mean" in user_agg.columns:
        user_agg_copy = user_agg.copy()
        user_agg_copy["score_bin"] = pd.cut(user_agg_copy["credit_score"], bins=5)
        score_stats = user_agg_copy.groupby("score_bin", observed=True)["amount_mean"].mean()
        axes[0, 1].bar(range(len(score_stats)), score_stats.values, color="coral", alpha=0.7, edgecolor="black")
        axes[0, 1].set_xticks(range(len(score_stats)))
        axes[0, 1].set_xticklabels([str(b) for b in score_stats.index], rotation=45, ha="right", fontsize=8)
        axes[0, 1].set_title("Average Amount by Credit Score Bins", fontsize=11, fontweight="bold")
        axes[0, 1].set_ylabel("Average Amount ($)")
        axes[0, 1].grid(alpha=0.3, axis="y")

    if "num_credit_cards" in user_agg.columns and "amount_sum" in user_agg.columns:
        card_stats = user_agg.groupby("num_credit_cards")["amount_sum"].mean().head(10)
        axes[1, 0].bar(range(len(card_stats)), card_stats.values, color="teal", alpha=0.7, edgecolor="black")
        axes[1, 0].set_xticks(range(len(card_stats)))
        axes[1, 0].set_xticklabels(card_stats.index, rotation=45)
        axes[1, 0].set_title("Total Spending by Number of Credit Cards", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Total Amount ($)")
        axes[1, 0].grid(alpha=0.3, axis="y")

    if "current_age" in user_agg.columns and "amount_mean" in user_agg.columns:
        age_bins = pd.cut(user_agg["current_age"], bins=6)
        age_stats = user_agg.groupby(age_bins, observed=True)["amount_mean"].mean()
        axes[1, 1].bar(range(len(age_stats)), age_stats.values, color="green", alpha=0.7, edgecolor="black")
        axes[1, 1].set_xticks(range(len(age_stats)))
        axes[1, 1].set_xticklabels([str(b) for b in age_stats.index], rotation=45, ha="right", fontsize=8)
        axes[1, 1].set_title("Average Amount by Age Bins", fontsize=11, fontweight="bold")
        axes[1, 1].set_ylabel("Average Amount ($)")
        axes[1, 1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure("demographic_patterns.png", output_dir)
    plt.close()


def plot_us_transaction_map(
    transactions: pd.DataFrame,
    output_dir: Path,
    amount_col: str = "amount_usd",
    city_col: str = "merchant_city",
    state_col: str = "merchant_state",
    top_n: int = 300,
    force: bool = False,
) -> None:

    geo_df = prepare_us_transaction_geo_data(
        transactions=transactions,
        amount_col=amount_col,
        city_col=city_col,
        state_col=state_col,
        top_n=top_n,
    )

    us = load_us_geometry(force_download=force)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax in axes:
        us.plot(
            ax=ax,
            color="#e8e8e8",
            edgecolor="#aaaaaa",
            linewidth=0.6,
            zorder=1,
        )

        ax.set_xlim(-125, -66)
        ax.set_ylim(24, 50)

        ax.grid(alpha=0.2)

    sizes = scale_bubbles(
        geo_df["txn_count"],
    )

    sc0 = axes[0].scatter(
        geo_df["lon"],
        geo_df["lat"],
        s=sizes,
        c=geo_df["txn_count"],
        cmap="YlOrRd",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )

    fig.colorbar(
        sc0,
        ax=axes[0],
        label="Transaction Count",
    )

    axes[0].set_title("Transaction Volume by Merchant Location")

    sizes = scale_bubbles(
        geo_df["avg_amount"],
    )

    sc1 = axes[1].scatter(
        geo_df["lon"],
        geo_df["lat"],
        s=sizes,
        c=geo_df["avg_amount"],
        cmap="Blues",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )

    fig.colorbar(
        sc1,
        ax=axes[1],
        label="Average Transaction Amount ($)",
    )

    axes[1].set_title("Average Transaction Amount by Merchant Location")

    plt.tight_layout()
    save_figure("us_merchant_map.png", output_dir)
    plt.close()
