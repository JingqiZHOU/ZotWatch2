"""Candidate fetching pipeline."""

import logging
from pathlib import Path

from zotwatch.config.settings import Settings
from zotwatch.core.models import CandidateWork
from zotwatch.sources.base import get_enabled_sources

logger = logging.getLogger(__name__)


def fetch_candidates(settings: Settings) -> list[CandidateWork]:
    """Fetch candidates from all enabled sources.

    Args:
        settings: Application settings

    Returns:
        List of candidate works from all sources
    """
    results: list[CandidateWork] = []

    for source in get_enabled_sources(settings):
        try:
            candidates = source.fetch()
            results.extend(candidates)
            logger.info("Fetched %d candidates from %s", len(candidates), source.name)
        except Exception as exc:
            logger.error("Failed to fetch from %s: %s", source.name, exc)

    logger.info("Fetched %d total candidate works", len(results))
    return results


class CandidateFetcher:
    """Wrapper for candidate fetching."""

    def __init__(self, settings: Settings, base_dir: Path):
        self.settings = settings
        self.base_dir = Path(base_dir)

    def fetch_all(self) -> list[CandidateWork]:
        """Fetch all candidates."""
        return fetch_candidates(self.settings)


__all__ = ["fetch_candidates", "CandidateFetcher"]
