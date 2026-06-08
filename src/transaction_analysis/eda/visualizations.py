from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

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
    ax.axvline(0, color="red", linestyle="--", label="Granica decyzyjna")
    ax.set_title("Rozkład wyników detekcji anomalii Isolation Forest", fontsize=12, fontweight="bold")
    ax.set_xlabel("Wynik  (niższy = bardziej anomalny)")
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
    ax.set_xticklabels(["Normalny", "Anomalny"])
    ax.set_title("Średnia kwota transakcji: Normalny vs Anomalny", fontsize=12, fontweight="bold")
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
    axes[0].set_title("Ilość transakcji w zależności od kwoty", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Kwota transakcji (USD)")
    axes[0].set_ylabel("Ilość transakcji")
    axes[0].grid(alpha=0.3)

    axes[1].hist(np.log10(amounts.add(1)), bins=bins, color="teal", edgecolor="black", alpha=0.7)
    axes[1].set_title("Rozkład logarytmiczny kwoty transakcji", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("log10(1 + Kwota transakcji)")
    axes[1].set_ylabel("Ilość transakcji")
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
    ax.set_title("Ilość transakcji w zależności od miesiąca", fontsize=12, fontweight="bold")
    ax.set_xlabel("Miesiąc")
    ax.set_ylabel("Ilość transakcji")
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
    axes[0].set_title("Ilość transakcji w zależności od godziny", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Godzina")
    axes[0].set_ylabel("Ilość transakcji")
    axes[0].grid(alpha=0.3, axis="y")

    daily.rename(index=dict(enumerate(dow_labels))).plot(kind="bar", ax=axes[1], color="coral")
    axes[1].set_title("Ilość transakcji w zależności od dnia tygodnia", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Dzień tygodnia")
    axes[1].set_ylabel("Ilość transakcji")
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
    axes[0].set_xlabel("Ilość transakcji")
    axes[0].set_title("Najczęściej występujące błędy transakcji", fontsize=12, fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].grid(alpha=0.3, axis="x")

    dark_web.plot(
        kind="pie",
        ax=axes[1],
        autopct="%1.1f%%",
        colors=["#69b3a2", "#ff6b6b"],
        startangle=90,
    )
    axes[1].set_title("Karty na Dark Web", fontsize=12, fontweight="bold")
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
    ax.set_title("Rozkłady zdolności kredytowej dla mężczyzn i kobiet", fontsize=12, fontweight="bold")
    ax.set_xlabel("Zdolność kredytowa")
    ax.set_ylabel("Częstość występowania")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure("credit_score_by_gender.png", output_dir)
    plt.close()


def plot_top_states_by_amount(
    transactions: pd.DataFrame,
    output_dir: Path,
    top_n: int = 15,
) -> None:
    """Top US states by total transaction amount, with online transactions as their own bucket."""
    df = transactions[["merchant_state", "transaction_type", "amount_usd"]].copy()
    df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")
    df = df.dropna(subset=["amount_usd"])

    bucket = df["merchant_state"].astype("object")
    online_mask = df["transaction_type"].astype(str).str.contains("Online", case=False, na=False)
    bucket = bucket.mask(online_mask, "Online")
    df = df.assign(bucket=bucket).dropna(subset=["bucket"])

    top_states = df.groupby("bucket", observed=True)["amount_usd"].sum().sort_values(ascending=False).head(top_n)

    colors = ["coral" if label == "Online" else "steelblue" for label in top_states.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(top_states)), top_states.values / 1e6, color=colors)
    ax.set_yticks(range(len(top_states)))
    ax.set_yticklabels(top_states.index)
    ax.set_xlabel("Suma kwot transakcji (mln USD)")
    ax.set_title(f"Top {top_n} stanów / krajów w zależności od sumy kwot transakcji", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    save_figure(f"top_{top_n}_states_by_amount.png", output_dir)
    plt.close()


def plot_state_fraud_rate(
    transactions: pd.DataFrame,
    output_dir: Path,
    top_n: int = 5,
    min_transactions: int = 100,
) -> None:
    df = transactions.loc[transactions["fraud"].notna(), ["merchant_state", "transaction_type", "fraud"]].copy()
    df["fraud"] = df["fraud"].astype(bool)

    bucket = df["merchant_state"].astype("object")
    online_mask = df["transaction_type"].astype(str).str.contains("Online", case=False, na=False)
    bucket = bucket.mask(online_mask, "Online")
    df = df.assign(bucket=bucket).dropna(subset=["bucket"])

    per_state = df.groupby("bucket", observed=True)["fraud"].agg(txn_count="size", fraud_pct="mean")
    per_state["fraud_pct"] *= 100

    eligible = per_state[per_state["txn_count"] >= min_transactions]
    top = eligible.sort_values("fraud_pct", ascending=False).head(top_n)

    if top.empty:
        return

    colors = ["coral" if label == "Online" else "steelblue" for label in top.index]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(top))
    bars = ax.bar(x, top["fraud_pct"], color=colors, edgecolor="black", alpha=0.85)

    for bar, pct in zip(bars, top["fraud_pct"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, pct, f"{pct:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(top.index, rotation=60, ha="right")
    ax.set_xlabel("Stan / kraj sprzedawcy (merchant_state)")
    ax.set_ylabel("Procent transakcji oszukańczych (%)")
    ax.set_title(
        f"Top {top_n} stanów i krajów wg procentu transakcji oszukańczych",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure("state_fraud_rate.png", output_dir)
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
    ax.set_ylabel("Ilość transakcji")
    ax.set_title("Kody MCC o największych przychodach", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(f"top_{top_n}_mcc.png", output_dir)
    plt.close()


def plot_merchant_fraud_rate(
    transactions: pd.DataFrame,
    output_dir: Path,
    mcc_codes: pd.DataFrame | None = None,
    top_n: int = 25,
    min_transactions: int = 50,
) -> None:
    df = transactions.loc[transactions["fraud"].notna(), ["merchant_id", "mcc", "fraud"]].copy()
    df["fraud"] = df["fraud"].astype(bool)

    per_merchant = df.groupby("merchant_id", observed=True)["fraud"].agg(txn_count="size", fraud_pct="mean")
    per_merchant["fraud_pct"] *= 100

    dominant_mcc = (
        df.groupby(["merchant_id", "mcc"], observed=True)
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n")
        .drop_duplicates("merchant_id", keep="last")
        .set_index("merchant_id")["mcc"]
        .astype(int)
    )
    per_merchant["mcc"] = dominant_mcc

    eligible = per_merchant[per_merchant["txn_count"] >= min_transactions]
    top = eligible.sort_values("fraud_pct", ascending=False).head(top_n)
    top = top.sort_values(["mcc", "fraud_pct"], ascending=[True, False])

    if top.empty:
        return

    mccs = top["mcc"].unique()
    palette = sns.color_palette("tab20", n_colors=len(mccs))
    mcc_to_color = dict(zip(mccs, palette, strict=False))
    bar_colors = [mcc_to_color[m] for m in top["mcc"]]

    mcc_label: dict[int, str] = {}
    if mcc_codes is not None:
        mcc_label = {int(i): str(d) for i, d in zip(mcc_codes["id"], mcc_codes["description"], strict=False)}

    fig, ax = plt.subplots(figsize=(16, 7))
    x = range(len(top))
    bars = ax.bar(x, top["fraud_pct"], color=bar_colors, edgecolor="black", alpha=0.85)

    for bar, pct in zip(bars, top["fraud_pct"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, pct, f"{pct:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(top.index.astype(int), rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("ID sprzedawcy (merchant_id)")
    ax.set_ylabel("Procent transakcji oszukańczych (%)")
    ax.set_title(
        f"Top {top_n} sprzedawców wg procentu transakcji oszukańczych",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.3, axis="y")

    handles = [
        Patch(
            facecolor=mcc_to_color[m],
            edgecolor="black",
            label=f"{m} - {mcc_label[m]}" if m in mcc_label else f"MCC {m}",
        )
        for m in mccs
    ]
    ax.legend(handles=handles, title="MCC", fontsize=8, title_fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    plt.tight_layout()
    save_figure("merchant_fraud_rate_by_mcc.png", output_dir)
    plt.close()


def plot_user_transaction_distribution(
    user_agg: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(user_agg["txn_count"], bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Rozkład ilości transakcji spośród klientów", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Ilość transakcji")
    axes[0, 0].set_ylabel("Ilość klientów")
    axes[0, 0].grid(alpha=0.3)

    if "amount_sum" in user_agg.columns:
        axes[0, 1].hist(user_agg["amount_sum"], bins=50, color="coral", edgecolor="black", alpha=0.7)
        axes[0, 1].set_title("Rozkład sumy kwot transakcji spośród klientów", fontsize=11, fontweight="bold")
        axes[0, 1].set_xlabel("Suma kwot transakcji ($)")
        axes[0, 1].set_ylabel("Ilość klientów")
        axes[0, 1].grid(alpha=0.3)

    if "amount_mean" in user_agg.columns:
        axes[1, 0].hist(user_agg["amount_mean"], bins=50, color="teal", edgecolor="black", alpha=0.7)
        axes[1, 0].set_title("Rozkład średniej kwoty transakcji spośród klientów", fontsize=11, fontweight="bold")
        axes[1, 0].set_xlabel("Średnia kwota transakcji ($)")
        axes[1, 0].set_ylabel("Ilość klientów")
        axes[1, 0].grid(alpha=0.3)

    if "txn_frequency" in user_agg.columns:
        axes[1, 1].hist(user_agg["txn_frequency"], bins=50, color="green", edgecolor="black", alpha=0.7)
        axes[1, 1].set_title("Rozkład częstotliwości transakcji spośród klientów", fontsize=11, fontweight="bold")
        axes[1, 1].set_xlabel("Ilość transakcji na dzień")
        axes[1, 1].set_ylabel("Ilość klientów")
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
        cbar_kws={"label": "Korelacja"},
    )
    ax.set_title("Macierz korelacji - agregacja cech klientów", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_figure("correlation_heatmap.png", output_dir)
    plt.close()


def plot_demographic_patterns(
    user_agg: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Transaction patterns broken down by gender, credit score, card count, and age."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    if "gender" in user_agg.columns and "txn_count" in user_agg.columns:
        gender_stats = user_agg.groupby("gender")["txn_count"].mean()
        axes[0].bar(gender_stats.index, gender_stats.values, color="steelblue", alpha=0.7, edgecolor="black")
        axes[0].set_title("Średnia ilość transakcji w zależności od płci", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Średnia ilość transakcji")
        axes[0].grid(alpha=0.3, axis="y")

    if "credit_score" in user_agg.columns and "amount_mean" in user_agg.columns:
        user_agg_copy = user_agg.copy()
        user_agg_copy["score_bin"] = pd.cut(user_agg_copy["credit_score"], bins=5)
        score_stats = user_agg_copy.groupby("score_bin", observed=True)["amount_mean"].mean()
        axes[1].bar(range(len(score_stats)), score_stats.values, color="coral", alpha=0.7, edgecolor="black")
        axes[1].set_xticks(range(len(score_stats)))
        axes[1].set_xticklabels([str(b) for b in score_stats.index], rotation=45, ha="right", fontsize=8)
        axes[1].set_title(
            "Średnia kwota transakcji w zależności od zdolności kredytowej",
            fontsize=11,
            fontweight="bold",
        )
        axes[1].set_ylabel("Średnia kwota transakcji ($)")
        axes[1].grid(alpha=0.3, axis="y")

    if "num_credit_cards" in user_agg.columns and "amount_sum" in user_agg.columns:
        card_stats = user_agg.groupby("num_credit_cards")["amount_sum"].mean().head(10)
        axes[2].bar(range(len(card_stats)), card_stats.values, color="teal", alpha=0.7, edgecolor="black")
        axes[2].set_xticks(range(len(card_stats)))
        axes[2].set_xticklabels(card_stats.index, rotation=45)
        axes[2].set_title(
            "Suma kwot transakcji w zależności od ilości kart kredytowych",
            fontsize=11,
            fontweight="bold",
        )
        axes[2].set_ylabel("Suma kwot transakcji ($)")
        axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure("demographic_patterns.png", output_dir)
    plt.close()


_FRAUDSTER_PROFILE_FEATURES: list[str] = [
    "txn_count",
    "amount_mean",
    "amount_max",
    "unique_merchants",
    "mcc_entropy",
    "share_online",
    "share_swipe",
    "night_share",
    "error_rate",
    "age",
    "credit_score",
    "yearly_income_usd",
    "debt_to_income",
    "num_credit_cards",
    "avg_credit_limit",
]


def plot_fraudster_profile(
    profile: pd.DataFrame,
    output_dir: Path,
    features: dict[str, str] | None = None,
) -> None:
    features = features or _FRAUDSTER_PROFILE_FEATURES
    cols = [c for c in features if c in profile.columns]
    fraud = profile[profile["is_fraudster"]]
    rest = profile[~profile["is_fraudster"]]
    if fraud.empty or rest.empty or not cols:
        return

    rows = []
    for col in cols:
        a = pd.to_numeric(fraud[col], errors="coerce").dropna()
        b = pd.to_numeric(rest[col], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        if pooled <= 0:
            continue
        rows.append((col, (a.mean() - b.mean()) / pooled))

    if not rows:
        return
    prof = pd.DataFrame(rows, columns=["feature", "d"]).sort_values("d")
    colors = ["#ff6b6b" if v > 0 else "#4c72b0" for v in prof["d"]]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(prof)), prof["d"], color=colors, edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(prof)))
    ax.set_yticklabels(prof["feature"])
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("Standaryzowana różnica średnich:  oszust − pozostali")
    ax.set_title(
        "Profil klientów oszukańczych",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.3, axis="x")
    for i, v in enumerate(prof["d"]):
        ax.text(
            v + (0.01 if v >= 0 else -0.01),
            i,
            f"{v:+.2f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
        )
    plt.tight_layout()
    save_figure("fraudster_profile.png", output_dir)
    plt.close()


def plot_fraudster_channel_mix(
    profile: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Channel/timing behaviour: fraudsters vs the rest, as mean per-client shares."""
    feature_labels = {
        "share_online": "Online",
        "share_swipe": "Swipe",
        "share_chip": "Chip",
        "night_share": "Nocne (22-6)",
        "weekend_share": "Weekend",
    }
    cols = [c for c in feature_labels if c in profile.columns]
    fraud = profile[profile["is_fraudster"]]
    rest = profile[~profile["is_fraudster"]]
    if fraud.empty or rest.empty or not cols:
        return

    fraud_means = [fraud[c].mean() * 100 for c in cols]
    rest_means = [rest[c].mean() * 100 for c in cols]
    labels = [feature_labels[c] for c in cols]

    x = np.arange(len(cols))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_groups = [
        ax.bar(
            x - width / 2,
            fraud_means,
            width,
            label="Oszuści",
            color="#ff6b6b",
            edgecolor="black",
            alpha=0.85,
        ),
        ax.bar(
            x + width / 2,
            rest_means,
            width,
            label="Pozostali",
            color="#4c72b0",
            edgecolor="black",
            alpha=0.85,
        ),
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Średni udział transakcji klienta (%)")
    ax.set_title("Kanały i pory transakcji: oszuści vs pozostali", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    for bars in bar_groups:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    plt.tight_layout()
    save_figure("fraudster_channel_mix.png", output_dir)
    plt.close()


def plot_fraud_frequent_age(
    profile: pd.DataFrame,
    output_dir: Path,
    top_frac: float = 0.1,
) -> None:
    df = profile[["age", "fraud_count"]].copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.dropna(subset=["age"])
    df = df[(df["age"] >= 18) & (df["age"] <= 100)]
    if df.empty:
        return

    cutoff = df["fraud_count"].quantile(1 - top_frac)
    top = df[df["fraud_count"] >= cutoff]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bins = np.arange(df["age"].min(), df["age"].max() + 5, 5)
    axes[0].hist(
        df["age"],
        bins=bins,
        color="#cccccc",
        edgecolor="white",
        alpha=0.7,
        density=True,
        label=f"Wszyscy klienci (n={len(df):,})",
    )
    axes[0].hist(
        top["age"],
        bins=bins,
        color="#ff6b6b",
        edgecolor="black",
        alpha=0.6,
        density=True,
        label=f"Najczęstsi oszuści — top {top_frac:.0%} (n={len(top):,})",
    )
    axes[0].axvline(
        top["age"].median(),
        color="crimson",
        linestyle="--",
        label=f"Mediana wieku: {top['age'].median():.0f}",
    )
    axes[0].set_xlabel("Wiek klienta")
    axes[0].set_ylabel("Gęstość")
    axes[0].set_title("Rozkład wieku najczęstszych oszustów", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    age_bins = [18, 25, 35, 45, 55, 65, 75, 101]
    age_labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)
    grp = df.groupby("age_group", observed=True)["fraud_count"].mean()
    axes[1].bar(range(len(grp)), grp.values, color="steelblue", edgecolor="black", alpha=0.8)
    axes[1].set_xticks(range(len(grp)))
    axes[1].set_xticklabels(grp.index, rotation=45, ha="right")
    axes[1].set_xlabel("Grupa wiekowa")
    axes[1].set_ylabel("Śr. liczba transakcji oszukańczych")
    axes[1].set_title("Średnia liczba oszustw wg grupy wiekowej", fontsize=12, fontweight="bold")
    axes[1].grid(alpha=0.3, axis="y")
    for i, v in enumerate(grp.values):
        axes[1].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save_figure("fraud_frequent_age.png", output_dir)
    plt.close()


def plot_merchant_fraud_rate_volume(
    transactions: pd.DataFrame,
    output_dir: Path,
    mcc_codes: pd.DataFrame | None = None,
    min_transactions: int = 50,
    annotate_top: int = 8,
    max_mcc_colors: int = 10,
) -> None:
    df = transactions.loc[transactions["fraud"].notna(), ["merchant_id", "mcc", "amount_usd", "fraud"]].copy()
    df["fraud"] = df["fraud"].astype(bool)
    df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")
    df["fraud_amount"] = df["amount_usd"].where(df["fraud"], 0.0).clip(lower=0)

    per = df.groupby("merchant_id", observed=True).agg(
        txn_count=("fraud", "size"),
        fraud_pct=("fraud", "mean"),
        fraud_amount=("fraud_amount", "sum"),
    )
    per["fraud_pct"] *= 100

    dominant_mcc = (
        df.groupby(["merchant_id", "mcc"], observed=True)
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n")
        .drop_duplicates("merchant_id", keep="last")
        .set_index("merchant_id")["mcc"]
        .astype(int)
    )
    per["mcc"] = dominant_mcc

    eligible = per[per["txn_count"] >= min_transactions].copy()
    if eligible.empty:
        return

    top_mccs = eligible["mcc"].value_counts().head(max_mcc_colors).index.tolist()
    palette = sns.color_palette("tab10", n_colors=len(top_mccs))
    mcc_to_color = dict(zip(top_mccs, palette, strict=False))
    bar_colors = [mcc_to_color.get(m, "#cccccc") for m in eligible["mcc"]]

    fa = eligible["fraud_amount"]
    sizes = 20 + 580 * (fa / fa.max()) if fa.max() > 0 else pd.Series(40.0, index=eligible.index)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(
        eligible["txn_count"],
        eligible["fraud_pct"],
        s=sizes,
        c=bar_colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Liczba transakcji sprzedawcy")
    ax.set_ylabel("Procent transakcji oszukańczych (%)")
    ax.set_title(
        "Sprzedawcy: wolumen vs odsetek oszustw (rozmiar bąbla = kwota oszustw $)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)

    for mid, row in eligible.sort_values("fraud_pct", ascending=False).head(annotate_top).iterrows():
        ax.annotate(
            str(int(mid)),
            (row["txn_count"], row["fraud_pct"]),
            fontsize=8,
            fontweight="bold",
            xytext=(4, 2),
            textcoords="offset points",
        )

    mcc_label: dict[int, str] = {}
    if mcc_codes is not None:
        mcc_label = {int(i): str(d) for i, d in zip(mcc_codes["id"], mcc_codes["description"], strict=False)}
    handles = [
        Patch(
            facecolor=mcc_to_color[m],
            edgecolor="black",
            label=f"{m} - {mcc_label[m]}" if m in mcc_label else f"MCC {m}",
        )
        for m in top_mccs
    ]
    handles.append(Patch(facecolor="#cccccc", edgecolor="black", label="Inne MCC"))
    ax.legend(
        handles=handles,
        title="Dominujące MCC",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )

    plt.tight_layout()
    save_figure("merchant_fraud_rate_volume.png", output_dir)
    plt.close()


def plot_mcc_fraud_rate(
    transactions: pd.DataFrame,
    output_dir: Path,
    mcc_codes: pd.DataFrame | None = None,
    top_n: int = 15,
    min_transactions: int = 200,
) -> None:
    df = transactions.loc[transactions["fraud"].notna(), ["mcc", "fraud"]].copy()
    df["fraud"] = df["fraud"].astype(bool)

    per = df.groupby("mcc", observed=True)["fraud"].agg(txn_count="size", fraud_pct="mean")
    per["fraud_pct"] *= 100

    eligible = per[per["txn_count"] >= min_transactions]
    top = eligible.sort_values("fraud_pct", ascending=False).head(top_n)
    if top.empty:
        return

    mcc_label: dict[int, str] = {}
    if mcc_codes is not None:
        mcc_label = {int(i): str(d) for i, d in zip(mcc_codes["id"], mcc_codes["description"], strict=False)}
    labels = [f"{int(m)} - {mcc_label[int(m)]}" if int(m) in mcc_label else f"MCC {int(m)}" for m in top.index]

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.barh(range(len(top)), top["fraud_pct"], color="indianred", edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Procent transakcji oszukańczych (%)")
    ax.set_title(
        f"Top {top_n} kategorii MCC wg odsetka oszustw",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.3, axis="x")
    for i, (pct, n) in enumerate(zip(top["fraud_pct"], top["txn_count"], strict=False)):
        ax.text(pct, i, f"  {pct:.2f}%  (n={int(n):,})", va="center", fontsize=8)

    plt.tight_layout()
    save_figure("mcc_fraud_rate.png", output_dir)
    plt.close()


def plot_pca_scree(
    explained_variance: np.ndarray,
    output_dir: Path,
) -> None:
    """Scree (bars) + cumulative variance (line) for a fitted PCA."""
    n = len(explained_variance)
    pcs = np.arange(1, n + 1)
    cum = np.cumsum(explained_variance)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(pcs, explained_variance, color="steelblue", edgecolor="black", alpha=0.7, label="Per-component")
    ax.set_xlabel("Główna składowa")
    ax.set_ylabel("Wariancja wyjaśniona")
    ax.set_title("PCA Scree oraz Wariancja skumulowana", fontsize=12, fontweight="bold")
    ax.set_xticks(pcs)
    ax.grid(alpha=0.3, axis="y")

    for bar, v in zip(bars, explained_variance, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)

    ax2 = ax.twinx()
    ax2.plot(pcs, cum, marker="o", color="coral", linewidth=2, label="Kumulacja")
    ax2.set_ylabel("Wariancja skumulowana")
    ax2.set_ylim(0, 1.02)
    ax2.axhline(0.8, color="gray", linestyle="--", alpha=0.6)
    ax2.text(n, 0.81, "80%", color="gray", fontsize=9, ha="right")

    plt.tight_layout()
    save_figure("pca_screen.png", output_dir)
    plt.close()


def plot_pca_biplot(
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    explained_variance: np.ndarray,
    output_dir: Path,
    top_features: int = 8,
) -> None:
    """PC1-PC2 score scatter with top-loading feature arrows."""
    pc_x, pc_y = "PC1", "PC2"

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(scores[pc_x], scores[pc_y], s=14, alpha=0.35, color="steelblue", edgecolor="none")

    # Pick top features by magnitude in the PC1-PC2 plane.
    mag = np.sqrt(loadings[pc_x] ** 2 + loadings[pc_y] ** 2)
    top = mag.nlargest(top_features).index

    score_extent = max(scores[pc_x].abs().max(), scores[pc_y].abs().max())
    arrow_scale = 0.85 * score_extent / mag.loc[top].max()

    for feat in top:
        dx, dy = loadings.loc[feat, pc_x] * arrow_scale, loadings.loc[feat, pc_y] * arrow_scale
        ax.arrow(0, 0, dx, dy, color="crimson", alpha=0.85, width=score_extent * 0.003, head_width=score_extent * 0.02)
        ax.text(dx * 1.08, dy * 1.08, feat, color="darkred", fontsize=9, ha="center", va="center", fontweight="bold")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(f"{pc_x} ({explained_variance[0]:.1%})")
    ax.set_ylabel(f"{pc_y} ({explained_variance[1]:.1%})")
    ax.set_title("PCA klienta - najważniejsze cechy", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure("pca_biplot.png", output_dir)
    plt.close()


def plot_pca_fraud_overlay(
    scores: pd.DataFrame,
    fraud_rate: pd.Series,
    output_dir: Path,
) -> None:
    """PC1-PC2 scatter colored by per-user fraud rate."""
    fr = fraud_rate.reindex(scores.index).fillna(0.0)

    fig, ax = plt.subplots(figsize=(12, 8))
    # Plot zero-fraud users first as a faint background, then non-zero on top.
    zero = fr == 0
    ax.scatter(scores.loc[zero, "PC1"], scores.loc[zero, "PC2"], s=10, alpha=0.2, color="lightgray")
    sc = ax.scatter(
        scores.loc[~zero, "PC1"],
        scores.loc[~zero, "PC2"],
        c=fr.loc[~zero] * 100,
        cmap="YlOrRd",
        s=28,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Procent transakcji oszukańczych")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Klienta - Mapa oszukańczych transakcji", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure("pca_fraud_overlay.png", output_dir)
    plt.close()


def plot_pca_clusters(
    scores: pd.DataFrame,
    clusters: pd.Series,
    loadings: pd.DataFrame,
    output_dir: Path,
) -> None:
    """PC1-PC2 scatter colored by KMeans cluster, with top loadings as labels."""
    clusters = clusters.reindex(scores.index)
    palette = sns.color_palette("tab10", n_colors=clusters.nunique())

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, c in enumerate(sorted(clusters.unique())):
        mask = clusters == c
        ax.scatter(
            scores.loc[mask, "PC1"],
            scores.loc[mask, "PC2"],
            s=20,
            alpha=0.6,
            color=palette[i % len(palette)],
            label=f"Klaster {c} (n={int(mask.sum())})",
            edgecolor="none",
        )

    # Label each cluster with its top-magnitude loading feature (so the axes "read").
    top_pc1 = loadings["PC1"].abs().idxmax()
    top_pc2 = loadings["PC2"].abs().idxmax()
    pc1_sign = "+" if loadings.loc[top_pc1, "PC1"] > 0 else "-"
    pc2_sign = "+" if loadings.loc[top_pc2, "PC2"] > 0 else "-"

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(f"PC1  ({pc1_sign}{top_pc1})")
    ax.set_ylabel(f"PC2  ({pc2_sign}{top_pc2})")
    ax.set_title("PCA klienta - KMeans dla pierwszych dwóch głównych składowych", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure("pca_clusters.png", output_dir)
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
        label="Ilość transakcji",
    )

    axes[0].set_title("Ilość transakcji w zależności od lokalizacji sprzedawcy")

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
        label="Średnia kwota transakcji ($)",
    )

    axes[1].set_title("Średnia kwota transakcji w zależności od lokalizacji sprzedawcy")

    plt.tight_layout()
    save_figure("us_merchant_map.png", output_dir)
    plt.close()
