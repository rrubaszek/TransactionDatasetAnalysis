import logging

from transaction_analysis.config.logger import setup_logging
from transaction_analysis.config.paths import FRAUD_DATASET_DIR, PLOTS_DIR
from transaction_analysis.data.step import bootstrap, cleanup, preprocess
from transaction_analysis.eda.step import analysis

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Downloading datasets...")
    bootstrap.run(force=False)

    logger.info("Preprocessing data...")
    preprocess.run(
        dataset_in_dir=FRAUD_DATASET_DIR / "raw", dataset_out_dir=FRAUD_DATASET_DIR / "preprocessed", force=False
    )
    logger.info("Preprocessing complete.")

    logger.info("Cleaning data...")
    cleanup.run(
        dataset_in_dir=FRAUD_DATASET_DIR / "preprocessed",
        dataset_out_dir=FRAUD_DATASET_DIR / "cleaned",
        force=False,
    )
    logger.info("Cleaning complete.")

    logger.info("Running analysis...")
    analysis.run(dataset_in_dir=FRAUD_DATASET_DIR / "cleaned", plots_out_dir=PLOTS_DIR, force=True)
    logger.info("Analysis complete.")


if __name__ == "__main__":
    setup_logging()
    main()
