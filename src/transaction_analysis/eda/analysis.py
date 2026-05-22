from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from transaction_analysis.eda.aggregations import (
    aggregate_by_mcc,
    aggregate_by_merchant,
    aggregate_transactions_by_time,
    aggregate_transactions_by_user,
    calculate_risk_metrics,
)
from transaction_analysis.eda.anomalies import anomaly_analysis
from transaction_analysis.eda.geoanalysis import plot_us_map
from transaction_analysis.eda.utils import configure_plotting, load_cleaned_data
from transaction_analysis.eda.visualizations import (
    plot_amount_distribution,
    plot_anomalies,
    plot_correlation_heatmap,
    plot_credit_score_by_gender,
    plot_demographic_patterns,
    plot_errors_and_darkweb,
    plot_time_patterns,
    plot_top_mcc,
    plot_top_merchants,
    plot_transactions_over_time,
    plot_user_transaction_distribution,
)
from transaction_analysis.paths import FRAUD_DATASET_DIR, PLOTS_DIR

"""Entry point for EDA analysis."""


class TransactionAnalysis:
    def __init__(self, dataset_dir: Path, output_dir: Path):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)

        configure_plotting()

        self.transactions, self.users, self.cards = load_cleaned_data(self.dataset_dir)

        self._user_agg = None
        self._time_agg = None
        self._merchant_agg = None
        self._mcc_agg = None
        self._risk_metrics = None

    @property
    def user_agg(self) -> pd.DataFrame:
        if self._user_agg is None:
            self._user_agg = aggregate_transactions_by_user(self.transactions, self.users, self.cards)
        return self._user_agg

    @property
    def time_agg(self) -> pd.DataFrame:
        if self._time_agg is None:
            self._time_agg = aggregate_transactions_by_time(self.transactions)
        return self._time_agg

    @property
    def merchant_agg(self) -> pd.DataFrame:
        if self._merchant_agg is None:
            self._merchant_agg = aggregate_by_merchant(self.transactions)
        return self._merchant_agg

    @property
    def mcc_agg(self) -> pd.DataFrame:
        if self._mcc_agg is None:
            self._mcc_agg = aggregate_by_mcc(self.transactions)
        return self._mcc_agg

    @property
    def risk_metrics(self) -> pd.DataFrame:
        if self._risk_metrics is None:
            self._risk_metrics = calculate_risk_metrics(self.user_agg, self.transactions)
        return self._risk_metrics

    def print_summary_statistics(self) -> None:
        print("\nTRANSACTION STATISTICS:")
        print(f"  Total transactions: {len(self.transactions):,}")
        print(f"  Unique users: {self.transactions['client_id'].nunique():,}")
        print(f"  Unique cards: {self.transactions['card_id'].nunique():,}")

        amounts = pd.to_numeric(self.transactions["amount_usd"], errors="coerce").dropna()
        print("\n  Amount statistics:")
        print(f"    Mean: ${amounts.mean():,.2f}")
        print(f"    Median: ${amounts.median():,.2f}")
        print(f"    Std Dev: ${amounts.std():,.2f}")
        print(f"    Min: ${amounts.min():,.2f}")
        print(f"    Max: ${amounts.max():,.2f}")

        print("\nUSER STATISTICS:")
        print(f"  Total users: {len(self.user_agg):,}")
        print(f"  Avg txns per user: {self.user_agg['txn_count'].mean():.1f}")
        print(f"  Median txns per user: {self.user_agg['txn_count'].median():.1f}")
        print(f"  Max txns per user: {self.user_agg['txn_count'].max():,}")

        print("\nMERCHANT STATISTICS:")
        print(f"  Unique merchants: {self.transactions['merchant_id'].nunique():,}")
        print(f"  Top merchant txn count: {self.merchant_agg['txn_count'].iloc[0]:,}")

        print("\nMCC STATISTICS:")
        print(f"  Unique MCC codes: {self.transactions['mcc'].nunique():,}")
        print(f"  Top MCC txn count: {self.mcc_agg['txn_count'].iloc[0]:,}")

    def print_top_items(self, n: int = 10) -> None:
        print(f"\nTop {n} Merchants by Transaction Count:")
        for idx, row in self.merchant_agg.head(n).iterrows():
            print(
                f"  {idx + 1:2d}. Merchant {int(row['merchant_id']):6d} - {int(row['txn_count']):6,} txns, \
                ${row['amount_mean']:8,.2f} avg"
            )

        print(f"\nTop {n} MCC Codes by Transaction Count:")
        for idx, row in self.mcc_agg.head(n).iterrows():
            print(
                f"  {idx + 1:2d}. MCC {int(row['mcc']):6d} - {int(row['txn_count']):6,} txns, \
                {int(row['unique_customers']):6,} customers"
            )

        print(f"\nTop {min(n, len(self.user_agg))} Users by Transaction Count:")
        for idx, row in self.user_agg.head(n).iterrows():
            print(
                f"  {idx + 1:2d}. User {int(row['client_id']):6d} - {int(row['txn_count']):6,} txns, \
                ${row['amount_sum']:12,.2f} total"
            )

    def plot_graph(self, plot_fn: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        """Run ``plot_fn(*args, self.output_dir, **kwargs)``."""
        plot_fn(*args, self.output_dir, **kwargs)


def run(dataset_in_dir: Path, plots_out_dir: Path, force: bool = False) -> None:
    analysis = TransactionAnalysis(dataset_dir=dataset_in_dir, output_dir=plots_out_dir)
    analysis.print_summary_statistics()
    analysis.print_top_items(n=10)

    if not force and any(plots_out_dir.iterdir()):
        print(f"Plots already exist in {plots_out_dir}, skipping. Use `force=True` to re-run.")
        return

    analysis.plot_graph(plot_amount_distribution, analysis.transactions)
    analysis.plot_graph(plot_transactions_over_time, analysis.transactions)
    analysis.plot_graph(plot_time_patterns, analysis.transactions)
    analysis.plot_graph(plot_errors_and_darkweb, analysis.transactions, analysis.cards)
    analysis.plot_graph(plot_credit_score_by_gender, analysis.users)
    analysis.plot_graph(plot_top_merchants, analysis.merchant_agg, top_n=15)
    analysis.plot_graph(plot_top_mcc, analysis.mcc_agg, top_n=15)
    analysis.plot_graph(plot_user_transaction_distribution, analysis.user_agg)
    analysis.plot_graph(plot_correlation_heatmap, analysis.user_agg)
    analysis.plot_graph(plot_demographic_patterns, analysis.user_agg)

    anomalous_user_agg, _ = anomaly_analysis(analysis.user_agg)
    analysis.plot_graph(plot_anomalies, anomalous_user_agg)
    analysis.plot_graph(plot_us_map, analysis.transactions)
    print("Analysis complete. Visualizations saved to:", plots_out_dir)


if __name__ == "__main__":
    run(dataset_in_dir=FRAUD_DATASET_DIR / "cleaned", plots_out_dir=PLOTS_DIR, force=True)
