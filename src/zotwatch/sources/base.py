"""Base source definitions and registry."""

import logging
import re
from abc import ABC, abstractmethod

from zotwatch.config.settings import Settings
from zotwatch.core.models import CandidateWork
from zotwatch.utils.datetime import ensure_aware, parse_date
from zotwatch.utils.text import clean_html, clean_title

logger = logging.getLogger(__name__)


class BaseSource(ABC):
    """Abstract base class for candidate sources."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique source identifier."""
        ...

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether this source is enabled in config."""
        ...

    @abstractmethod
    def fetch(self, days_back: int = 7) -> list[CandidateWork]:
        """Fetch candidates from this source."""
        ...

    def validate_config(self) -> bool:
        """Validate source-specific configuration."""
        return True


class SourceRegistry:
    """Registry for dynamically discovering and loading sources."""

    _sources: dict[str, type[BaseSource]] = {}

    @classmethod
    def register(cls, source_class: type[BaseSource]) -> type[BaseSource]:
        """Decorator to register a source."""
        # Get name from class
        _ = object.__new__(source_class)
        name = source_class.__name__.lower().replace("source", "")
        cls._sources[name] = source_class
        return source_class

    @classmethod
    def get_source(cls, name: str) -> type[BaseSource] | None:
        """Get source class by name."""
        return cls._sources.get(name.lower())

    @classmethod
    def get_enabled_sources(cls, settings: Settings) -> list[BaseSource]:
        """Return instantiated sources that are enabled in config."""
        enabled = []
        for name, source_class in cls._sources.items():
            source = source_class(settings)
            if source.enabled:
                enabled.append(source)
        return enabled

    @classmethod
    def all_sources(cls) -> dict[str, type[BaseSource]]:
        """Get all registered sources."""
        return cls._sources.copy()


def get_enabled_sources(settings: Settings) -> list[BaseSource]:
    """Convenience function to get enabled sources."""
    return SourceRegistry.get_enabled_sources(settings)


# Patterns for non-article entries (journal metadata pages)
_NON_ARTICLE_PATTERNS = [
    # Exact match patterns (case-insensitive)
    r"^table of contents$",
    r"^masthead$",
    r"^editorial board$",
    r"^errat(a|um)$",
    r"^correction(s)?$",
    r"^retraction$",
    r"^connect\. support\. inspire\.$",  # IEEE slogan
    # Contains patterns
    r"information for authors",
    r"publication information",
    r"author guidelines",
    r"instructions for authors",
    r"guide for authors",
    # Journal name only patterns (matches "IEEE Transactions on X" alone)
    r"^ieee transactions on [a-z\s]+$",
    r"^ieee journal of [a-z\s]+$",
    r"^proceedings of the ieee$",
    r"^ieee [a-z\s]+ magazine$",
]

# Compile patterns for efficiency
_NON_ARTICLE_REGEX = [re.compile(p, re.IGNORECASE) for p in _NON_ARTICLE_PATTERNS]


def is_non_article_title(title: str, venue: str | None = None) -> bool:
    """Check if title indicates a non-article entry (journal metadata page).

    IEEE and other publishers register DOIs for non-article content like:
    - Table of Contents
    - Publication Information
    - Information for Authors
    - Journal masthead
    - Connect. Support. Inspire. (IEEE slogan)

    Args:
        title: Paper title to check.
        venue: Optional venue/journal name for comparison.

    Returns:
        True if title indicates non-article content.
    """
    if not title:
        return True

    title_clean = title.strip()

    # Check against known patterns
    for pattern in _NON_ARTICLE_REGEX:
        if pattern.search(title_clean):
            logger.debug("Filtered non-article title: %s (matched pattern)", title_clean)
            return True

    # Check if title is just the venue name
    if venue and title_clean.lower() == venue.lower():
        logger.debug("Filtered non-article title: %s (equals venue name)", title_clean)
        return True

    return False


__all__ = [
    "BaseSource",
    "SourceRegistry",
    "get_enabled_sources",
    "clean_title",
    "ensure_aware",
    "parse_date",
    "clean_html",
    "is_non_article_title",
]
