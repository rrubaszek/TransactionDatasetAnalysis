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
    plot_us_transaction_map,
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

    def plot_amount_distribution(self, force: bool = False) -> None:
        out_file = self.output_dir / "amount_distribution.png"
        if out_file.exists() and not force:
            print("Amount distribution already plotted, skipping. Use `force=True` to re-run.")
            return
        plot_amount_distribution(self.transactions, self.output_dir)

    def plot_transactions_over_time(self, force: bool = False) -> None:
        out_file = self.output_dir / "transactions_over_time.png"
        if out_file.exists() and not force:
            print("Transactions over time already plotted, skipping. Use `force=True` to re-run.")
            return
        plot_transactions_over_time(self.transactions, self.output_dir)

    def plot_time_patterns(self, force: bool = False) -> None:
        out_file = self.output_dir / "time_patterns.png"
        if out_file.exists() and not force:
            print("Time patterns already plotted, skipping. Use `force=True` to re-run.")
            return
        plot_time_patterns(self.transactions, self.output_dir)

    def plot_errors_and_darkweb(self, force: bool = False) -> None:
        out_file = self.output_dir / "errors_and_darkweb.png"
        if out_file.exists() and not force:
            print("Errors and dark web already plotted, skipping. Use `force=True` to re-run.")
            return
        plot_errors_and_darkweb(self.transactions, self.cards, self.output_dir)

    def plot_credit_score_by_gender(self, force: bool = False) -> None:
        out_file = self.output_dir / "credit_score_by_gender.png"
        if out_file.exists() and not force:
            print("Credit score by gender already plotted, skipping. Use `force=True` to re-run.")
            return
        plot_credit_score_by_gender(self.users, self.output_dir)

    def plot_top_merchants(self, top_n: int = 15, force: bool = False) -> None:
        out_file = self.output_dir / "top_merchants.png"
        if out_file.exists() and not force:
            print("Top merchants already plotted, skipping. Use `force=True` to re-run.")
            return
        plot_top_merchants(self.merchant_agg, self.output_dir, top_n=top_n)

    def plot_top_mcc(self, top_n: int = 15, force: bool = False) -> None:
        out_file = self.output_dir / "top_mcc.png"
        if out_file.exists() and not force:
            print("Top MCC codes already plotted, skipping. Use `force=True` to re-run.")
            return
        plot_top_mcc(self.mcc_agg, self.output_dir, top_n=top_n)

    def plot_user_transaction_distribution(self, force: bool = False) -> None:
        out_file = self.output_dir / "user_transaction_distribution.png"
        if out_file.exists() and not force:
            print("User transaction distribution already plotted, skipping. Use `force=True` to re-run.")
            return
        plot_user_transaction_distribution(self.user_agg, self.output_dir)

    def plot_correlation_heatmap(self, force: bool = False) -> None:
        out_file = self.output_dir / "correlation_heatmap.png"
        if out_file.exists() and not force:
            print("Correlation heatmap already plotted, skipping. Use `force=True` to re-run.")
            return
        plot_correlation_heatmap(self.user_agg, self.output_dir)

    def plot_demographic_patterns(self, force: bool = False) -> None:
        out_file = self.output_dir / "demographic_patterns.png"
        if out_file.exists() and not force:
            print("Demographic patterns already plotted, skipping. Use `force=True` to re-run.")
            return
        plot_demographic_patterns(self.user_agg, self.output_dir)

    def plot_anomalies(self, force: bool = False) -> None:
        out_file = self.output_dir / "anomalies.png"
        if out_file.exists() and not force:
            print("Anomalies already plotted, skipping. Use `force=True` to re-run.")
            return
        anomalous_user_agg, _ = anomaly_analysis(self.user_agg)
        plot_anomalies(anomalous_user_agg, self.output_dir)

    def plot_us_transaction_map(self, force: bool = False) -> None:
        out_file = self.output_dir / "us_transaction_map.png"
        if out_file.exists() and not force:
            print("US transaction map already plotted, skipping. Use `force=True` to re-run.")
            return
        plot_us_transaction_map(self.transactions, self.output_dir, force=force)

    def run(self, force: bool = False) -> None:
        self.print_summary_statistics()
        self.print_top_items(n=10)
        self.plot_amount_distribution(force=force)
        self.plot_transactions_over_time(force=force)
        self.plot_time_patterns(force=force)
        self.plot_errors_and_darkweb(force=force)
        self.plot_credit_score_by_gender(force=force)
        self.plot_top_merchants(force=force)
        self.plot_top_mcc(force=force)
        self.plot_user_transaction_distribution(force=force)
        self.plot_correlation_heatmap(force=force)
        self.plot_demographic_patterns(force=force)
        self.plot_anomalies(force=force)
        self.plot_us_transaction_map(force=force)
        print("Analysis complete. Visualizations saved to:", self.output_dir)


if __name__ == "__main__":
    analysis = TransactionAnalysis(
        dataset_dir=FRAUD_DATASET_DIR / "cleaned",
        output_dir=PLOTS_DIR,
    )

    # Comment/uncomment to run individual analyses
    analysis.print_summary_statistics()
    analysis.print_top_items(n=10)
    analysis.plot_amount_distribution(force=True)
    analysis.plot_transactions_over_time(force=True)
    analysis.plot_time_patterns(force=True)
    analysis.plot_errors_and_darkweb(force=True)
    analysis.plot_credit_score_by_gender(force=True)
    analysis.plot_top_merchants(force=True)
    analysis.plot_top_mcc(force=True)
    analysis.plot_user_transaction_distribution(force=True)
    analysis.plot_correlation_heatmap(force=True)
    analysis.plot_demographic_patterns(force=True)
    analysis.plot_anomalies(force=True)
    analysis.plot_us_transaction_map(force=True)
