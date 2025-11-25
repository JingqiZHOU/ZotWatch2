"""Candidate fetching pipeline."""

import logging
from pathlib import Path
from typing import List

from zotwatch.config.settings import Settings
from zotwatch.core.models import CandidateWork
from zotwatch.sources.base import get_enabled_sources
from zotwatch.sources.crossref import CrossrefSource

logger = logging.getLogger(__name__)


def fetch_candidates(settings: Settings, base_dir: Path) -> List[CandidateWork]:
    """Fetch candidates from all enabled sources.

    Args:
        settings: Application settings
        base_dir: Base directory for profile

    Returns:
        List of candidate works from all sources
    """
    profile_path = base_dir / "data" / "profile.json"
    results: List[CandidateWork] = []

    for source in get_enabled_sources(settings):
        # Set profile path for Crossref top venues
        if isinstance(source, CrossrefSource):
            source.set_profile_path(profile_path)

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

    def fetch_all(self) -> List[CandidateWork]:
        """Fetch all candidates."""
        return fetch_candidates(self.settings, self.base_dir)


__all__ = ["fetch_candidates", "CandidateFetcher"]
