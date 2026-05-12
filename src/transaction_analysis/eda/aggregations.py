import pandas as pd


def aggregate_transactions_by_user(
    transactions: pd.DataFrame,
    users: pd.DataFrame | None = None,
    cards: pd.DataFrame | None = None,
) -> pd.DataFrame:
    agg = (
        transactions.groupby("client_id")
        .agg(
            {
                "transaction_id": "count",
                "amount_usd": ["sum", "mean", "median", "std", "min", "max"],
                "date": ["min", "max"],
            }
        )
        .reset_index()
    )
    agg.columns = [
        "client_id",
        "txn_count",
        "amount_sum",
        "amount_mean",
        "amount_median",
        "amount_std",
        "amount_min",
        "amount_max",
        "first_txn_date",
        "last_txn_date",
    ]

    agg["txn_span_days"] = (pd.to_datetime(agg["last_txn_date"]) - pd.to_datetime(agg["first_txn_date"])).dt.days
    agg["txn_frequency"] = agg["txn_count"] / (agg["txn_span_days"] + 1)

    optional_aggs = [
        transactions.groupby("client_id")["merchant_id"].nunique().reset_index(name="unique_merchants"),
    ]

    if "errors" in transactions.columns:
        optional_aggs.append(
            transactions.groupby("client_id")
            .apply(lambda x: (x["errors"].notna().sum() / len(x)) * 100, include_groups=False)
            .reset_index(name="error_rate_pct")
        )

    if "transaction_type" in transactions.columns:
        optional_aggs.append(
            transactions.groupby("client_id")
            .apply(
                lambda x: (x["transaction_type"].str.contains("chip", case=False, na=False).sum() / len(x)) * 100,
                include_groups=False,
            )
            .reset_index(name="chip_usage_pct")
        )

    for optional_agg in optional_aggs:
        agg = agg.merge(optional_agg, on="client_id", how="left")

    if users is not None and not users.empty:
        user_cols = [
            "id",
            "gender",
            "credit_score",
            "per_capita_income_usd",
            "yearly_income_usd",
            "total_debt_usd",
            "num_credit_cards",
        ]
        available_cols = [c for c in user_cols if c in users.columns]
        if available_cols:
            agg = agg.merge(
                users[available_cols].rename(columns={"id": "client_id"}),
                on="client_id",
                how="left",
            )

    if cards is not None and not cards.empty:
        card_agg = (
            cards.groupby("client_id")
            .agg(num_cards=("card_id", "count"), avg_credit_limit=("credit_limit_usd", "mean"))
            .reset_index()
        )
        agg = agg.merge(card_agg, on="client_id", how="left")

    return agg.sort_values("txn_count", ascending=False).reset_index(drop=True)


def aggregate_transactions_by_time(transactions: pd.DataFrame) -> pd.DataFrame:
    txns = transactions.copy()
    txns["date"] = pd.to_datetime(txns["date"])

    agg = (
        txns.groupby(txns["date"].dt.date)
        .agg(
            {
                "transaction_id": "count",
                "amount_usd": ["sum", "mean", "std"],
                "errors": lambda x: (x.notna().sum() / len(x)) * 100,
            }
        )
        .reset_index()
    )
    agg.columns = ["date", "txn_count", "amount_sum", "amount_mean", "amount_std", "error_rate_pct"]

    return agg


def aggregate_by_merchant(transactions: pd.DataFrame) -> pd.DataFrame:
    agg = (
        transactions.groupby("merchant_id")
        .agg(
            {
                "transaction_id": "count",
                "amount_usd": ["sum", "mean", "std"],
                "client_id": "nunique",
                "card_id": "nunique",
            }
        )
        .reset_index()
    )
    agg.columns = [
        "merchant_id",
        "txn_count",
        "amount_sum",
        "amount_mean",
        "amount_std",
        "unique_customers",
        "unique_cards",
    ]

    if "merchant_city" in transactions.columns:
        cities = (
            transactions.groupby("merchant_id")["merchant_city"]
            .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else None)
            .reset_index(name="merchant_city")
        )
        agg = agg.merge(cities, on="merchant_id", how="left")

    return agg.sort_values("txn_count", ascending=False).reset_index(drop=True)


def aggregate_by_mcc(transactions: pd.DataFrame) -> pd.DataFrame:
    agg = (
        transactions.groupby("mcc")
        .agg(
            {
                "transaction_id": "count",
                "amount_usd": ["sum", "mean", "std"],
                "client_id": "nunique",
            }
        )
        .reset_index()
    )
    agg.columns = ["mcc", "txn_count", "amount_sum", "amount_mean", "amount_std", "unique_customers"]

    return agg.sort_values("txn_count", ascending=False).reset_index(drop=True)


def calculate_risk_metrics(user_agg: pd.DataFrame) -> pd.DataFrame:
    result = user_agg.copy()

    if "amount_std" in result.columns and "amount_mean" in result.columns:
        result["amount_volatility"] = (result["amount_std"] / (result["amount_mean"] + 1e-6)).fillna(0)

    if "total_debt_usd" in result.columns and "yearly_income_usd" in result.columns:
        result["debt_to_income"] = (result["total_debt_usd"] / (result["yearly_income_usd"] + 1e-6)).fillna(0)

    if "num_cards" in result.columns:
        result["cards_per_user"] = result["num_cards"].fillna(0)

    if "error_rate_pct" in result.columns:
        result["high_error_rate"] = result["error_rate_pct"] > result["error_rate_pct"].quantile(0.75)

    return result
