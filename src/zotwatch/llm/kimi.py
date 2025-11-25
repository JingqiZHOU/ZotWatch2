"""Kimi (Moonshot AI) LLM provider implementation."""

import logging
from typing import List, Optional

import requests

from zotwatch.config.settings import LLMConfig

from .base import BaseLLMProvider, LLMResponse
from .retry import with_retry

logger = logging.getLogger(__name__)


class KimiClient(BaseLLMProvider):
    """Kimi (Moonshot AI) API client.

    Supports both thinking models (kimi-k2-thinking-*) and standard models.
    Thinking models automatically use temperature=1.0 and max_tokens>=16000.
    """

    BASE_URL = "https://api.moonshot.cn/v1"
    # Models that use the thinking/reasoning feature
    THINKING_MODEL_PREFIXES = ("kimi-k2-thinking",)
    MIN_THINKING_TOKENS = 16000

    def __init__(
        self,
        api_key: str,
        default_model: str = "kimi-k2-thinking-turbo",
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ):
        """Initialize Kimi client.

        Args:
            api_key: Moonshot API key.
            default_model: Default model to use.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
            backoff_factor: Exponential backoff factor.
        """
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._session = requests.Session()

    @classmethod
    def from_config(cls, config: LLMConfig) -> "KimiClient":
        """Create client from LLM configuration."""
        return cls(
            api_key=config.api_key,
            default_model=config.model,
            timeout=120.0,  # Thinking models need longer timeout
            max_retries=config.retry.max_attempts,
            backoff_factor=config.retry.backoff_factor,
        )

    @property
    def name(self) -> str:
        return "kimi"

    def _is_thinking_model(self, model: str) -> bool:
        """Check if the model is a thinking model."""
        return any(model.startswith(prefix) for prefix in self.THINKING_MODEL_PREFIXES)

    def complete(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Send completion request to Kimi API.

        For thinking models, temperature is forced to 1.0 and max_tokens is
        ensured to be at least 16000 for proper reasoning output.
        """
        return self._complete_with_retry(
            prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @with_retry(max_attempts=3, backoff_factor=2.0, initial_delay=1.0)
    def _complete_with_retry(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Internal completion with retry logic."""
        use_model = model or self.default_model

        # Adjust parameters for thinking models
        if self._is_thinking_model(use_model):
            temperature = 1.0  # Thinking models require temperature=1.0
            if max_tokens < self.MIN_THINKING_TOKENS:
                max_tokens = self.MIN_THINKING_TOKENS
                logger.debug(
                    "Increased max_tokens to %d for thinking model %s",
                    max_tokens,
                    use_model,
                )

        response = self._session.post(
            f"{self.BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": use_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        message = data["choices"][0]["message"]

        # Extract content, ignoring reasoning_content for thinking models
        content = message.get("content", "")
        tokens_used = data.get("usage", {}).get("total_tokens", 0)

        return LLMResponse(
            content=content,
            model=data.get("model", use_model),
            tokens_used=tokens_used,
        )

    def available_models(self) -> List[str]:
        """List available Kimi models."""
        # Kimi doesn't have a models endpoint, return known models
        return [
            "kimi-k2-thinking-turbo",
            "kimi-k2-thinking",
            "kimi-k2-turbo-preview",
        ]


__all__ = ["KimiClient"]
