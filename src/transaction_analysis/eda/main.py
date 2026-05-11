"""Entry point for EDA analysis."""

import os
from pathlib import Path

from transaction_analysis.eda.analysis import TransactionAnalysis
from transaction_analysis.paths import FRAUD_DATASET_DIR, PLOTS_DIR


def run(
    dataset_dir: Path,
    output_dir: Path,
    force: bool = True,
) -> None:

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    if output_dir.exists() and not force:
        print("EDA plots already generated, skipping. Use `force=True` to re-run.")
        return

    os.makedirs(output_dir, exist_ok=True)

    analysis = TransactionAnalysis(dataset_dir=dataset_dir, output_dir=output_dir)
    analysis.run_complete_analysis()


def main() -> None:
    """Run the EDA analysis with default paths."""
    run(
        dataset_dir=FRAUD_DATASET_DIR / "cleaned",
        output_dir=PLOTS_DIR,
    )


if __name__ == "__main__":
    main()
