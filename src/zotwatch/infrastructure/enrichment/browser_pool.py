"""Browser pool for parallel page fetching with Camoufox.

Features:
- Parallel browser instances with configurable concurrency
- Semaphore-controlled request limiting
- Automatic Cloudflare bypass per request
- Random jitter between requests to avoid detection
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Cloudflare challenge detection patterns
CLOUDFLARE_TITLE_INDICATORS = [
    "Just a moment...",
    "Checking your browser",
]

CLOUDFLARE_BODY_INDICATORS = [
    "Verify you are human",
    "Please wait while we verify your browser",
    "Enable JavaScript and cookies to continue",
]


@dataclass
class FetchResult:
    """Result of a page fetch operation."""

    url: str
    html: Optional[str]
    final_url: Optional[str]
    success: bool
    error: Optional[str] = None


def _is_cloudflare_challenge(html: str) -> bool:
    """Detect if page contains Cloudflare challenge."""
    if not html:
        return False
    html_lower = html.lower()

    for indicator in CLOUDFLARE_TITLE_INDICATORS:
        if indicator.lower() in html_lower:
            return True

    for indicator in CLOUDFLARE_BODY_INDICATORS:
        if indicator.lower() in html_lower:
            return True

    return False


class BrowserPool:
    """Pool of browser instances for parallel page fetching.

    Uses Camoufox (Firefox-based anti-detect browser) with:
    - Configurable concurrency (default: 3 parallel requests)
    - Automatic Cloudflare bypass
    - Random jitter between requests
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        timeout: int = 60000,
        max_retries: int = 3,
        jitter_range: Tuple[float, float] = (0.5, 2.0),
    ):
        """Initialize browser pool.

        Args:
            max_concurrent: Maximum parallel browser instances.
            timeout: Page load timeout in milliseconds.
            max_retries: Maximum retry attempts per URL.
            jitter_range: Random delay range (min, max) seconds between requests.
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = max_retries
        self.jitter_range = jitter_range
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._browser = None
        self._initialized = False

    async def _ensure_browser(self):
        """Ensure browser is initialized."""
        if self._browser is None:
            from camoufox import AsyncCamoufox

            logger.info("Initializing Camoufox browser pool (max_concurrent=%d)", self.max_concurrent)
            self._browser = await AsyncCamoufox(
                headless=True,
                geoip=True,
                config={"forceScopeAccess": True},
                disable_coop=True,
                i_know_what_im_doing=True,
                humanize=True,
            ).__aenter__()
            self._initialized = True

    async def _solve_cloudflare(self, page) -> bool:
        """Attempt to solve Cloudflare challenge.

        Args:
            page: Camoufox page object.

        Returns:
            True if solved, False otherwise.
        """
        try:
            from camoufox_captcha import solve_captcha

            logger.debug("Attempting Cloudflare bypass...")

            success = await solve_captcha(
                page,
                captcha_type="cloudflare",
                challenge_type="interstitial",
                solve_attempts=3,
                solve_click_delay=3.0,
            )

            if success:
                await asyncio.sleep(5)
                try:
                    await page.wait_for_load_state("networkidle", timeout=20000)
                except Exception:
                    pass
                await asyncio.sleep(3)

                html = await page.content()
                if not _is_cloudflare_challenge(html):
                    logger.info("Cloudflare bypass successful")
                    return True

            return False

        except Exception as e:
            logger.debug("Cloudflare solve failed: %s", e)
            return False

    async def _fetch_single(self, url: str) -> FetchResult:
        """Fetch a single page with Cloudflare handling.

        Args:
            url: URL to fetch.

        Returns:
            FetchResult with HTML content or error.
        """
        await self._ensure_browser()

        for attempt in range(self.max_retries):
            page = None
            try:
                page = await self._browser.new_page()
                logger.debug("Fetching %s (attempt %d/%d)", url, attempt + 1, self.max_retries)

                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
                except Exception as e:
                    logger.debug("Navigation exception (may be normal): %s", str(e)[:100])

                try:
                    await page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    pass

                html = await page.content()
                final_url = page.url

                # Handle Cloudflare challenge
                if _is_cloudflare_challenge(html):
                    if await self._solve_cloudflare(page):
                        await asyncio.sleep(2)
                        html = await page.content()
                        final_url = page.url
                        if not _is_cloudflare_challenge(html):
                            return FetchResult(
                                url=url,
                                html=html,
                                final_url=final_url,
                                success=True,
                            )

                    if attempt < self.max_retries - 1:
                        logger.debug("Cloudflare bypass failed, retrying...")
                        await asyncio.sleep(5)
                        continue

                    return FetchResult(
                        url=url,
                        html=html,
                        final_url=final_url,
                        success=False,
                        error="Cloudflare bypass failed",
                    )

                return FetchResult(
                    url=url,
                    html=html,
                    final_url=final_url,
                    success=True,
                )

            except Exception as e:
                logger.warning("Fetch failed for %s: %s", url, e)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(3)
                else:
                    return FetchResult(
                        url=url,
                        html=None,
                        final_url=None,
                        success=False,
                        error=str(e),
                    )

            finally:
                if page:
                    try:
                        await page.close()
                    except Exception:
                        pass

        return FetchResult(
            url=url,
            html=None,
            final_url=None,
            success=False,
            error="Max retries exceeded",
        )

    async def _fetch_with_semaphore(self, url: str) -> FetchResult:
        """Fetch with semaphore for concurrency control.

        Args:
            url: URL to fetch.

        Returns:
            FetchResult.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)

        async with self._semaphore:
            # Add random jitter between requests
            jitter = random.uniform(*self.jitter_range)
            await asyncio.sleep(jitter)

            return await self._fetch_single(url)

    async def fetch_pages(self, urls: List[str]) -> List[FetchResult]:
        """Fetch multiple pages in parallel.

        Args:
            urls: List of URLs to fetch.

        Returns:
            List of FetchResults in same order as input URLs.
        """
        if not urls:
            return []

        logger.info("Fetching %d pages (max_concurrent=%d)", len(urls), self.max_concurrent)
        start_time = time.time()

        tasks = [self._fetch_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to FetchResults
        final_results = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                final_results.append(
                    FetchResult(
                        url=url,
                        html=None,
                        final_url=None,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        elapsed = time.time() - start_time
        success_count = sum(1 for r in final_results if r.success)
        logger.info(
            "Fetched %d/%d pages in %.1fs (%.1f pages/sec)",
            success_count,
            len(urls),
            elapsed,
            len(urls) / elapsed if elapsed > 0 else 0,
        )

        return final_results

    async def fetch_page(self, url: str) -> FetchResult:
        """Fetch a single page.

        Convenience method for single URL fetch.

        Args:
            url: URL to fetch.

        Returns:
            FetchResult.
        """
        results = await self.fetch_pages([url])
        return results[0]

    async def close(self):
        """Clean up browser resources."""
        if self._browser:
            try:
                await self._browser.__aexit__(None, None, None)
            except Exception as e:
                logger.debug("Error closing browser: %s", e)
            self._browser = None
            self._initialized = False
            logger.info("Browser pool closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class SyncBrowserPool:
    """Synchronous wrapper for BrowserPool.

    Provides a sync interface for code that can't use async/await.
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        timeout: int = 60000,
        max_retries: int = 3,
    ):
        """Initialize sync browser pool.

        Args:
            max_concurrent: Maximum parallel browser instances.
            timeout: Page load timeout in milliseconds.
            max_retries: Maximum retry attempts per URL.
        """
        self._pool = BrowserPool(
            max_concurrent=max_concurrent,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def fetch_pages(self, urls: List[str]) -> List[FetchResult]:
        """Fetch multiple pages in parallel (sync).

        Args:
            urls: List of URLs to fetch.

        Returns:
            List of FetchResults.
        """
        loop = self._get_loop()
        return loop.run_until_complete(self._pool.fetch_pages(urls))

    def fetch_page(self, url: str) -> FetchResult:
        """Fetch a single page (sync).

        Args:
            url: URL to fetch.

        Returns:
            FetchResult.
        """
        results = self.fetch_pages([url])
        return results[0]

    def close(self):
        """Clean up resources."""
        if self._loop and not self._loop.is_closed():
            self._loop.run_until_complete(self._pool.close())

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


__all__ = [
    "BrowserPool",
    "SyncBrowserPool",
    "FetchResult",
]
