"""Main entry point for transaction analysis pipeline."""

from transaction_analysis.cli import data
from transaction_analysis.config.logger import setup_logging


def main():
    """Run the complete transaction analysis pipeline."""
    setup_logging()
    data.main()


if __name__ == "__main__":
    main()
