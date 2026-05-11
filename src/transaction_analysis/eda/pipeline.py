import logging
from pathlib import Path

from transaction_analysis.eda.analysis import TransactionAnalysis
from transaction_analysis.eda.utils import configure_plotting

logger = logging.getLogger(__name__)


def run(
    dataset_dir: Path,
    output_dir: Path,
    *,
    force: bool = False,
) -> None:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        logger.info(
            "EDA outputs already present in %s - skipping, use force=True to re-run.",
            output_dir,
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    TransactionAnalysis(dataset_dir, output_dir).run_complete_analysis()

    logger.info("EDA complete, outputs written to %s", output_dir)
