"""Universal abstract scraper with Firefox browser and LLM extraction.

Replaces publisher-specific scrapers (IEEE, Springer, ScienceDirect) with a
universal approach that works with any DOI-resolvable publisher.

Flow: DOI -> doi.org redirect -> final page -> HTML -> LLM extract -> abstract
"""

import logging
import time
from typing import Dict, List, Optional

from zotwatch.llm.base import BaseLLMProvider

from .stealth_browser import StealthBrowser
from .llm_extractor import LLMAbstractExtractor

logger = logging.getLogger(__name__)


class UniversalScraper:
    """Universal DOI-based abstract scraper.

    Features:
    - Firefox browser for bypassing bot detection
    - LLM-based extraction (no fragile CSS selectors)
    - Works with any publisher that has DOI resolution
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        rate_limit_delay: float = 2.0,
        timeout: int = 30000,
        max_html_chars: int = 15000,
        llm_max_tokens: int = 1024,
        llm_temperature: float = 0.1,
    ):
        """Initialize the universal scraper.

        Args:
            llm: LLM provider for abstract extraction.
            rate_limit_delay: Minimum seconds between requests.
            timeout: Page load timeout in milliseconds.
            max_html_chars: Maximum HTML chars to send to LLM.
            llm_max_tokens: Maximum tokens for LLM response.
            llm_temperature: LLM temperature for extraction.
        """
        self.extractor = LLMAbstractExtractor(
            llm=llm,
            max_html_chars=max_html_chars,
            max_tokens=llm_max_tokens,
            temperature=llm_temperature,
        )
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self._last_request_time = 0.0

    def _wait_for_rate_limit(self):
        """Respect rate limit between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug("Rate limiting: sleeping %.1fs", sleep_time)
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def fetch_abstract(
        self,
        doi: str,
        title: Optional[str] = None,
    ) -> Optional[str]:
        """Fetch abstract for any DOI.

        Args:
            doi: Digital Object Identifier.
            title: Optional paper title for better LLM context.

        Returns:
            Extracted abstract or None.
        """
        self._wait_for_rate_limit()

        # Resolve DOI to actual page
        doi_url = f"https://doi.org/{doi}"
        html, final_url = StealthBrowser.fetch_page(doi_url, self.timeout)

        if not html:
            logger.debug("Failed to fetch page for DOI %s", doi)
            return None

        logger.debug("DOI %s resolved to %s", doi, final_url)

        # Extract abstract using LLM
        abstract = self.extractor.extract(html, title)

        if abstract:
            logger.info("Extracted abstract for %s (%d chars)", doi, len(abstract))

        return abstract

    def fetch_batch(
        self,
        items: List[Dict[str, str]],
    ) -> Dict[str, str]:
        """Fetch abstracts for multiple DOIs sequentially.

        Args:
            items: List of dicts with 'doi' and optional 'title'.

        Returns:
            Dict mapping DOI to abstract.
        """
        results = {}
        total = len(items)
        for idx, item in enumerate(items, 1):
            doi = item.get("doi")
            if not doi:
                continue
            title = item.get("title")
            logger.info("Fetching [%d/%d]: %s", idx, total, doi)
            abstract = self.fetch_abstract(doi, title)
            if abstract:
                results[doi] = abstract
                logger.info("Fetching [%d/%d]: success (%d chars)", idx, total, len(abstract))
            else:
                logger.info("Fetching [%d/%d]: no abstract found", idx, total)
        return results

    def close(self):
        """Clean up browser resources."""
        StealthBrowser.close()


# Legacy compatibility aliases
PlaywrightManager = StealthBrowser


__all__ = [
    "UniversalScraper",
    "PlaywrightManager",
]
