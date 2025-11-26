"""Paper metadata enrichment infrastructure."""

from .browser_pool import BrowserPool, FetchResult, SyncBrowserPool
from .cache import MetadataCache
from .llm_extractor import LLMAbstractExtractor
from .publisher_extractors import PublisherExtractor, extract_abstract, detect_publisher
from .publisher_scraper import AbstractScraper, PlaywrightManager, UniversalScraper
from .semantic_scholar import SemanticScholarClient
from .stealth_browser import StealthBrowser

__all__ = [
    # Browser pool
    "BrowserPool",
    "FetchResult",
    "SyncBrowserPool",
    # Abstract extraction
    "AbstractScraper",
    "LLMAbstractExtractor",
    "PublisherExtractor",
    "detect_publisher",
    "extract_abstract",
    # Other
    "MetadataCache",
    "SemanticScholarClient",
    "StealthBrowser",
    # Backward compatibility
    "PlaywrightManager",
    "UniversalScraper",
]
