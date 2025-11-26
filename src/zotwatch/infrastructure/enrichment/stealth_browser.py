"""Firefox browser for bypassing bot detection.

Firefox naturally bypasses many bot detection mechanisms used by publishers
like ScienceDirect, without needing additional stealth plugins.
"""

import logging
import threading
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class StealthBrowser:
    """Manages Playwright Firefox browser.

    Firefox is preferred over Chromium because it naturally bypasses
    many bot detection mechanisms without requiring stealth plugins.
    Verified to work with ScienceDirect and other major publishers.

    Thread-safe: supports concurrent page fetching via multiple contexts.
    """

    _playwright = None
    _browser = None
    _initialized = False
    _init_lock = threading.Lock()  # Protects browser initialization

    @classmethod
    def get_browser(cls):
        """Get or create Firefox browser instance (thread-safe)."""
        with cls._init_lock:
            if cls._initialized:
                return cls._browser

            cls._initialized = True
            try:
                from playwright.sync_api import sync_playwright

                cls._playwright = sync_playwright().start()
                cls._browser = cls._playwright.firefox.launch(headless=True)
                logger.info("Firefox browser initialized")
                return cls._browser
            except Exception as e:
                logger.warning("Failed to initialize Firefox browser: %s", e)
                cls._browser = None
                return None

    @classmethod
    def fetch_page(cls, url: str, timeout: int = 30000) -> Tuple[Optional[str], Optional[str]]:
        """Fetch page content using Firefox.

        Args:
            url: URL to fetch.
            timeout: Timeout in milliseconds.

        Returns:
            Tuple of (html_content, final_url) or (None, None) on failure.
        """
        browser = cls.get_browser()
        if browser is None:
            return None, None

        try:
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
            )
            page = context.new_page()

            try:
                page.goto(url, wait_until="networkidle", timeout=timeout)
                html = page.content()
                final_url = page.url
                return html, final_url
            finally:
                context.close()

        except Exception as e:
            logger.warning("Firefox fetch failed for %s: %s", url, e)
            return None, None

    @classmethod
    def close(cls):
        """Clean up browser resources."""
        if cls._browser:
            try:
                cls._browser.close()
            except Exception:
                pass
            cls._browser = None
        if cls._playwright:
            try:
                cls._playwright.stop()
            except Exception:
                pass
            cls._playwright = None
        cls._initialized = False


__all__ = ["StealthBrowser"]
