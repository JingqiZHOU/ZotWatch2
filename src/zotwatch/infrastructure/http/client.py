"""HTTP client with retry logic."""

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


class HTTPClient:
    """HTTP client with session management and retry logic."""

    def __init__(
        self,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ):
        self.session = requests.Session()
        if headers:
            self.session.headers.update(headers)
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> requests.Response:
        """Make GET request with retry logic."""
        return self._request("GET", url, params=params, headers=headers, **kwargs)

    def post(
        self,
        url: str,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> requests.Response:
        """Make POST request with retry logic."""
        return self._request("POST", url, json=json, headers=headers, **kwargs)

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make request with retry logic."""
        kwargs.setdefault("timeout", self.timeout)
        last_exception = None
        delay = 1.0

        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, url, **kwargs)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", delay))
                    logger.warning(
                        "Rate limited on %s, waiting %d seconds (attempt %d/%d)",
                        url,
                        retry_after,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(retry_after)
                    delay *= self.backoff_factor
                    continue
                return response
            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(
                    "Request failed: %s (attempt %d/%d)",
                    str(e),
                    attempt + 1,
                    self.max_retries,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= self.backoff_factor

        if last_exception:
            raise last_exception
        raise RuntimeError("Request failed after all retries")


__all__ = ["HTTPClient"]
