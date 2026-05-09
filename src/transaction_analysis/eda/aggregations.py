"""Data aggregation functions for EDA."""

import pandas as pd


def aggregate_transactions_by_user(
    transactions: pd.DataFrame,
    users: pd.DataFrame | None = None,
    cards: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate transaction-level data to user-level features.

    Args:
        transactions: Transaction dataframe
        users: Optional users dataframe for enrichment (id as index)
        cards: Optional cards dataframe for enrichment (id as index)

    Returns:
        User-level aggregated dataframe
    """
    # Use size() for count — avoids relying on any column named "id"
    txn_counts = (
        transactions.groupby("client_id")
        .size()
        .rename("txn_count")
        .reset_index()
        .rename(columns={"client_id": "user_id"})
    )

    user_agg = (
        transactions.groupby("client_id")
        .agg(
            {
                "amount_usd": ["sum", "mean", "median", "std", "min", "max"],
                "date": ["min", "max"],
            }
        )
        .reset_index()
    )

    user_agg.columns = [
        "user_id",
        "amount_sum",
        "amount_mean",
        "amount_median",
        "amount_std",
        "amount_min",
        "amount_max",
        "first_txn_date",
        "last_txn_date",
    ]

    user_agg = user_agg.merge(txn_counts, on="user_id", how="left")

    # Calculate additional features
    user_agg["txn_span_days"] = (
        pd.to_datetime(user_agg["last_txn_date"]) - pd.to_datetime(user_agg["first_txn_date"])
    ).dt.days

    user_agg["txn_frequency"] = user_agg["txn_count"] / (user_agg["txn_span_days"] + 1)

    # Error statistics
    if "errors" in transactions.columns:
        error_agg = (
            transactions.groupby("client_id")
            .apply(
                lambda x: (x["errors"].notna().sum() / len(x)) * 100,
                include_groups=False,
            )
            .reset_index(name="error_rate_pct")
            .rename(columns={"client_id": "user_id"})
        )
        user_agg = user_agg.merge(error_agg, on="user_id", how="left")

    # Chip usage (column is "trasaction_type" in source data — typo preserved intentionally)
    chip_col = next(
        (c for c in ["use_chip", "trasaction_type", "transaction_type"] if c in transactions.columns),
        None,
    )
    if chip_col is not None:
        chip_agg = (
            transactions.groupby("client_id")
            .apply(
                lambda x: (x[chip_col].str.contains("chip", case=False, na=False).sum() / len(x)) * 100,
                include_groups=False,
            )
            .reset_index(name="chip_usage_pct")
            .rename(columns={"client_id": "user_id"})
        )
        user_agg = user_agg.merge(chip_agg, on="user_id", how="left")

    # Merchant diversity
    merchant_diversity = (
        transactions.groupby("client_id")["merchant_id"]
        .nunique()
        .reset_index(name="unique_merchants")
        .rename(columns={"client_id": "user_id"})
    )
    user_agg = user_agg.merge(merchant_diversity, on="user_id", how="left")

    if users is not None and not users.empty:
        user_cols = [
            "current_age",
            "gender",
            "credit_score",
            "per_capita_income",
            "yearly_income",
            "total_debt",
            "num_credit_cards",
        ]
        available_cols = [c for c in user_cols if c in users.columns]
        if available_cols:
            user_info = users[available_cols].reset_index()
            # Rename whatever the index came back as to "user_id"
            user_info = user_info.rename(columns={user_info.columns[0]: "user_id"})
            user_agg = user_agg.merge(user_info, on="user_id", how="left")

    if cards is not None and not cards.empty:
        card_agg = (
            cards.reset_index()
            .groupby("client_id")
            .agg({"credit_limit_usd": "mean"})  # no "id" column — use size() instead
            .reset_index()
        )
        card_counts = cards.groupby("client_id").size().reset_index().rename(columns={0: "num_cards"})
        card_agg = card_agg.merge(card_counts, on="client_id")
        card_agg = card_agg.rename(columns={"client_id": "user_id", "credit_limit_usd": "avg_credit_limit"})
        user_agg = user_agg.merge(card_agg, on="user_id", how="left")

    return user_agg.sort_values("txn_count", ascending=False).reset_index(drop=True)


def aggregate_transactions_by_time(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions by time periods (daily).

    Args:
        transactions: Transaction dataframe with 'date' column

    Returns:
        Time-aggregated dataframe
    """
    txns = transactions.copy()
    txns["date"] = pd.to_datetime(txns["date"])
    date_groups = txns.groupby(txns["date"].dt.date)

    txn_counts = date_groups.size().reset_index()
    txn_counts.columns = ["date", "txn_count"]

    amount_agg = date_groups["amount_usd"].agg(["sum", "mean", "std"]).reset_index()
    amount_agg.columns = ["date", "amount_sum", "amount_mean", "amount_std"]

    time_agg = txn_counts.merge(amount_agg, on="date", how="left")

    if "errors" in txns.columns:
        error_agg = date_groups["errors"].apply(lambda x: (x.notna().sum() / len(x)) * 100).reset_index()
        error_agg.columns = ["date", "error_rate_pct"]
        time_agg = time_agg.merge(error_agg, on="date", how="left")
    else:
        time_agg["error_rate_pct"] = 0.0

    return time_agg


def aggregate_by_merchant(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions by merchant.

    Args:
        transactions: Transaction dataframe

    Returns:
        Merchant-level aggregated dataframe
    """
    txn_counts = transactions.groupby("merchant_id").size().reset_index().rename(columns={0: "txn_count"})

    merchant_agg = (
        transactions.groupby("merchant_id")
        .agg(
            {
                "amount_usd": ["sum", "mean", "std"],
                "client_id": "nunique",
                "card_id": "nunique",
            }
        )
        .reset_index()
    )

    merchant_agg.columns = [
        "merchant_id",
        "amount_sum",
        "amount_mean",
        "amount_std",
        "unique_customers",
        "unique_cards",
    ]

    merchant_agg = merchant_agg.merge(txn_counts, on="merchant_id", how="left")

    if "merchant_city" in transactions.columns:
        cities = (
            transactions.groupby("merchant_id")["merchant_city"]
            .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else None)
            .reset_index(name="merchant_city")
        )
        merchant_agg = merchant_agg.merge(cities, on="merchant_id", how="left")

    return merchant_agg.sort_values("txn_count", ascending=False).reset_index(drop=True)


def aggregate_by_mcc(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions by MCC (Merchant Category Code).

    Args:
        transactions: Transaction dataframe with 'mcc' column

    Returns:
        MCC-level aggregated dataframe
    """
    txn_counts = transactions.groupby("mcc").size().reset_index().rename(columns={0: "txn_count"})

    mcc_agg = (
        transactions.groupby("mcc")
        .agg(
            {
                "amount_usd": ["sum", "mean", "std"],
                "client_id": "nunique",
            }
        )
        .reset_index()
    )

    mcc_agg.columns = [
        "mcc",
        "amount_sum",
        "amount_mean",
        "amount_std",
        "unique_customers",
    ]

    mcc_agg = mcc_agg.merge(txn_counts, on="mcc", how="left")

    return mcc_agg.sort_values("txn_count", ascending=False).reset_index(drop=True)


def calculate_risk_metrics(
    user_agg: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate risk-related metrics for users.

    Args:
        user_agg: User-aggregated dataframe
        transactions: Optional transaction-level data for additional metrics

    Returns:
        User aggregation with added risk metrics
    """
    result = user_agg.copy()

    # Transaction volatility (coefficient of variation of amounts)
    if "amount_std" in result.columns and "amount_mean" in result.columns:
        result["amount_volatility"] = (result["amount_std"] / (result["amount_mean"] + 1e-6)).fillna(0)

    # Debt to income ratio
    if "total_debt" in result.columns and "yearly_income" in result.columns:
        result["debt_to_income"] = (result["total_debt"] / (result["yearly_income"] + 1e-6)).fillna(0)

    # Number of credit cards used
    if "num_cards" in result.columns:
        result["cards_per_user"] = result["num_cards"].fillna(0)

    # Error rate as risk indicator
    if "error_rate_pct" in result.columns:
        result["high_error_rate"] = result["error_rate_pct"] > result["error_rate_pct"].quantile(0.75)

    # Age-based risk (if available)
    if "current_age" in result.columns:
        result["is_senior"] = result["current_age"] > 65

    return result
