"""Abstract scraper with Camoufox browser and rule-based + LLM extraction.

Features:
- Parallel batch fetching with configurable concurrency
- Publisher-specific rule-based extraction (ACM, IEEE, Springer, Elsevier, etc.)
- LLM fallback for unknown publishers or failed rules
- Cloudflare bypass via camoufox-captcha

Flow: DOI -> doi.org redirect -> HTML -> Rules extract -> (LLM fallback) -> abstract
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

from zotwatch.llm.base import BaseLLMProvider

from .browser_pool import BrowserPool, FetchResult
from .llm_extractor import LLMAbstractExtractor
from .publisher_extractors import PublisherExtractor, extract_abstract
from .stealth_browser import StealthBrowser

logger = logging.getLogger(__name__)


class AbstractScraper:
    """DOI-based abstract scraper with parallel fetching.

    Features:
    - Parallel batch processing with configurable concurrency
    - Rule-based extraction for major publishers
    - LLM fallback when rules fail
    - Cloudflare bypass via Camoufox
    """

    def __init__(
        self,
        llm: Optional[BaseLLMProvider] = None,
        max_concurrent: int = 3,
        rate_limit_delay: float = 1.0,
        timeout: int = 60000,
        max_retries: int = 2,
        max_html_chars: int = 15000,
        llm_max_tokens: int = 1024,
        llm_temperature: float = 0.1,
        use_llm_fallback: bool = True,
    ):
        """Initialize the abstract scraper.

        Args:
            llm: LLM provider for fallback extraction. Optional if use_llm_fallback=False.
            max_concurrent: Maximum parallel browser instances.
            rate_limit_delay: Minimum seconds between requests (for sequential mode).
            timeout: Page load timeout in milliseconds.
            max_retries: Maximum retry attempts for Cloudflare challenges.
            max_html_chars: Maximum HTML chars to send to LLM.
            llm_max_tokens: Maximum tokens for LLM response.
            llm_temperature: LLM temperature for extraction.
            use_llm_fallback: Whether to use LLM when rules fail.
        """
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_llm_fallback = use_llm_fallback

        # Publisher-specific extractor
        self.publisher_extractor = PublisherExtractor(use_llm_fallback=use_llm_fallback)

        # LLM extractor (optional fallback)
        self.llm_extractor: Optional[LLMAbstractExtractor] = None
        if llm and use_llm_fallback:
            self.llm_extractor = LLMAbstractExtractor(
                llm=llm,
                max_html_chars=max_html_chars,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
            )

        self._last_request_time = 0.0
        self._pool: Optional[BrowserPool] = None

    def _wait_for_rate_limit(self):
        """Respect rate limit between requests (sequential mode)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug("Rate limiting: sleeping %.1fs", sleep_time)
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _extract_abstract(
        self,
        html: str,
        url: str,
        title: Optional[str] = None,
    ) -> Optional[str]:
        """Extract abstract using rules first, then LLM fallback.

        Args:
            html: Page HTML content.
            url: Final page URL (for publisher detection).
            title: Optional paper title for LLM context.

        Returns:
            Extracted abstract or None.
        """
        # Try rule-based extraction first
        abstract = extract_abstract(html, url)
        if abstract:
            return abstract

        # LLM fallback
        if self.llm_extractor and self.use_llm_fallback:
            logger.debug("Rule extraction failed, trying LLM fallback")
            abstract = self.llm_extractor.extract(html, title)
            if abstract:
                return abstract

        return None

    def fetch_abstract(
        self,
        doi: str,
        title: Optional[str] = None,
    ) -> Optional[str]:
        """Fetch abstract for a single DOI.

        Args:
            doi: Digital Object Identifier.
            title: Optional paper title for better extraction.

        Returns:
            Extracted abstract or None.
        """
        self._wait_for_rate_limit()

        doi_url = f"https://doi.org/{doi}"
        html, final_url = StealthBrowser.fetch_page(
            doi_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        if not html:
            logger.debug("Failed to fetch page for DOI %s", doi)
            return None

        logger.debug("DOI %s resolved to %s", doi, final_url)

        abstract = self._extract_abstract(html, final_url or doi_url, title)

        if abstract:
            logger.info("Extracted abstract for %s (%d chars)", doi, len(abstract))

        return abstract

    async def _fetch_batch_async(
        self,
        items: List[Dict[str, str]],
    ) -> Dict[str, str]:
        """Fetch abstracts for multiple DOIs in parallel (async).

        Args:
            items: List of dicts with 'doi' and optional 'title'.

        Returns:
            Dict mapping DOI to abstract.
        """
        if not items:
            return {}

        # Build DOI URL list
        doi_urls = []
        doi_map = {}  # url -> (doi, title)
        for item in items:
            doi = item.get("doi")
            if not doi:
                continue
            url = f"https://doi.org/{doi}"
            doi_urls.append(url)
            doi_map[url] = (doi, item.get("title"))

        if not doi_urls:
            return {}

        # Fetch pages in parallel
        async with BrowserPool(
            max_concurrent=self.max_concurrent,
            timeout=self.timeout,
            max_retries=self.max_retries,
        ) as pool:
            results = await pool.fetch_pages(doi_urls)

        # Extract abstracts from fetched pages
        abstracts = {}
        for result in results:
            if not result.success or not result.html:
                continue

            original_url = result.url
            if original_url not in doi_map:
                continue

            doi, title = doi_map[original_url]
            final_url = result.final_url or original_url

            abstract = self._extract_abstract(result.html, final_url, title)
            if abstract:
                abstracts[doi] = abstract
                logger.info("Extracted abstract for %s (%d chars)", doi, len(abstract))

        return abstracts

    def fetch_batch(
        self,
        items: List[Dict[str, str]],
        parallel: bool = True,
    ) -> Dict[str, str]:
        """Fetch abstracts for multiple DOIs.

        Args:
            items: List of dicts with 'doi' and optional 'title'.
            parallel: Use parallel fetching (default True).

        Returns:
            Dict mapping DOI to abstract.
        """
        if not items:
            return {}

        if parallel and len(items) > 1:
            # Use async parallel fetching
            logger.info("Batch fetching %d DOIs (parallel, max_concurrent=%d)", len(items), self.max_concurrent)
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self._fetch_batch_async(items))
                finally:
                    loop.close()
            except Exception as e:
                logger.warning("Parallel fetch failed, falling back to sequential: %s", e)
                # Fall through to sequential

        # Sequential fetching
        logger.info("Batch fetching %d DOIs (sequential)", len(items))
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


# Backward compatibility aliases
UniversalScraper = AbstractScraper
PlaywrightManager = StealthBrowser


__all__ = [
    "AbstractScraper",
    "UniversalScraper",
    "PlaywrightManager",
]
