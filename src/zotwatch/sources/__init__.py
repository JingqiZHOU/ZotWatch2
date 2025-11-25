"""Data source implementations."""

from .base import SourceRegistry, get_enabled_sources
from .arxiv import ArxivSource
from .crossref import CrossrefSource
from .zotero import ZoteroClient, ZoteroIngestor

__all__ = [
    "SourceRegistry",
    "get_enabled_sources",
    "ArxivSource",
    "CrossrefSource",
    "ZoteroClient",
    "ZoteroIngestor",
]
