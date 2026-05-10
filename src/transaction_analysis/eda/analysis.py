from pathlib import Path

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


class TransactionAnalysis:
    """Main EDA analysis class for transaction data."""

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
        """Lazy-load user aggregations."""
        if self._user_agg is None:
            self._user_agg = aggregate_transactions_by_user(self.transactions, self.users, self.cards)
        return self._user_agg

    @property
    def time_agg(self) -> pd.DataFrame:
        """Lazy-load time aggregations."""
        if self._time_agg is None:
            self._time_agg = aggregate_transactions_by_time(self.transactions)
        return self._time_agg

    @property
    def merchant_agg(self) -> pd.DataFrame:
        """Lazy-load merchant aggregations."""
        if self._merchant_agg is None:
            self._merchant_agg = aggregate_by_merchant(self.transactions)
        return self._merchant_agg

    @property
    def mcc_agg(self) -> pd.DataFrame:
        """Lazy-load MCC aggregations."""
        if self._mcc_agg is None:
            self._mcc_agg = aggregate_by_mcc(self.transactions)
        return self._mcc_agg

    @property
    def risk_metrics(self) -> pd.DataFrame:
        """Lazy-load risk metrics."""
        if self._risk_metrics is None:
            self._risk_metrics = calculate_risk_metrics(self.user_agg, self.transactions)
        return self._risk_metrics

    def print_summary_statistics(self) -> None:
        """Print summary statistics about the dataset."""

        print("\nTRANSACTION STATISTICS:")
        print(f"  Total transactions: {len(self.transactions):,}")
        print(f"  Unique users: {self.transactions['client_id'].nunique():,}")
        print(f"  Unique cards: {self.transactions['card_id'].nunique():,}")

        amounts_dollars = pd.to_numeric(self.transactions["amount_cents_usd"], errors="coerce").dropna() / 100
        print("\n  Amount statistics:")
        print(f"    Mean: ${amounts_dollars.mean():,.2f}")
        print(f"    Median: ${amounts_dollars.median():,.2f}")
        print(f"    Std Dev: ${amounts_dollars.std():,.2f}")
        print(f"    Min: ${amounts_dollars.min():,.2f}")
        print(f"    Max: ${amounts_dollars.max():,.2f}")

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
        top_merchants = self.merchant_agg.head(n)
        for idx, row in top_merchants.iterrows():
            print(
                f"  {idx + 1:2d}. Merchant {int(row['merchant_id']):6d} - {int(row['txn_count']):6,} txns, \
                ${row['amount_cents_mean']:8,.2f} avg"
            )

        print(f"\nTop {n} MCC Codes by Transaction Count:")
        top_mcc = self.mcc_agg.head(n)
        for idx, row in top_mcc.iterrows():
            print(
                f"  {idx + 1:2d}. MCC {int(row['mcc']):6d} - {int(row['txn_count']):6,} txns, \
                  {int(row['unique_customers']):6,} customers"
            )

        print(f"\nTop {min(n, len(self.user_agg))} Users by Transaction Count:")
        top_users = self.user_agg.head(n)
        for idx, row in top_users.iterrows():
            print(
                f"  {idx + 1:2d}. User {int(row['user_id']):6d} - {int(row['txn_count']):6,} txns, \
                ${row['amount_cents_sum'] / 100:12,.2f} / 100 total"
            )

        print("\n" + "=" * 70 + "\n")

    def generate_all_visualizations(self) -> None:

        try:
            print("Amount distribution...", end=" ")
            plot_amount_distribution(self.transactions, self.output_dir)
            print("done")
        except Exception as e:
            print(f"{e}")

        try:
            print("Transactions over time...", end=" ")
            plot_transactions_over_time(self.transactions, self.output_dir)
            print("done")
        except Exception as e:
            print(f"{e}")

        try:
            print("Time patterns (hour/dow)...", end=" ")
            plot_time_patterns(self.transactions, self.output_dir)
            print("done")
        except Exception as e:
            print(f"{e}")

        try:
            print("Errors and dark web...", end=" ")
            plot_errors_and_darkweb(self.transactions, self.cards, self.output_dir)
            print("done")
        except Exception as e:
            print(f"{e}")

        try:
            print("Credit score by gender...", end=" ")
            plot_credit_score_by_gender(self.users, self.output_dir)
            print("done")
        except Exception as e:
            print(f"{e}")

        try:
            print("Top merchants...", end=" ")
            plot_top_merchants(self.merchant_agg, self.output_dir, top_n=15)
            print("done")
        except Exception as e:
            print(f"{e}")

        try:
            print("Top MCC codes...", end=" ")
            plot_top_mcc(self.mcc_agg, self.output_dir, top_n=15)
            print("done")
        except Exception as e:
            print(f"{e}")

        try:
            print("User transaction distribution...", end=" ")
            plot_user_transaction_distribution(self.user_agg, self.output_dir)
            print("done")
        except Exception as e:
            print(f"{e}")

        try:
            print("Correlation heatmap...", end=" ")
            plot_correlation_heatmap(self.user_agg, self.output_dir)
            print("done")
        except Exception as e:
            print(f"{e}")

        try:
            print("Demographic patterns...", end=" ")
            plot_demographic_patterns(self.user_agg, self.output_dir)
            print("done")
        except Exception as e:
            print(f"{e}")

        try:
            print("Anomaly detection...", end=" ")
            self._user_agg, _ = anomaly_analysis(self.user_agg, self.output_dir)
            print("done")
        except Exception as e:
            print(f"{e}")

        try:
            print("US merchant map...", end=" ")
            plot_us_map(self.transactions, self.output_dir)
            print("done")
        except Exception as e:
            print(f"{e}")

    def run_complete_analysis(self) -> None:
        """Run complete EDA analysis with all reports and visualizations."""

        self.print_summary_statistics()
        self.print_top_items(n=10)
        self.generate_all_visualizations()

        print("Analysis complete. Visualizations saved to:", self.output_dir)
