"""Entry point for EDA analysis."""

import os
from pathlib import Path

from transaction_analysis.eda.analysis import TransactionAnalysis


def run(
    dataset_dir: Path,
    output_dir: Path,
    force: bool = False,
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
        dataset_dir=Path(__file__).resolve().parents[3] / "dataset" / "cleaned",
        output_dir=Path(__file__).resolve().parents[3] / "plots",
    )


if __name__ == "__main__":
    main()
