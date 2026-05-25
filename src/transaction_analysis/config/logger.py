"""Centralised logging configuration for the transaction_analysis package."""

import logging

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger. Idempotent: safe to call from any entry point."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=True)
