"""Logging configuration for ZotWatch."""

import logging


def setup_logging(level: int = logging.INFO, verbose: bool = False) -> None:
    """Configure root logger with a sensible default format."""
    if verbose:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name or "zotwatch")


__all__ = ["setup_logging", "get_logger"]
