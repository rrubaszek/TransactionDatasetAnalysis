"""Main entry point for transaction analysis pipeline."""

from transaction_analysis.cli import data


def main():
    """Run the complete transaction analysis pipeline."""
    data.main()


if __name__ == "__main__":
    main()
