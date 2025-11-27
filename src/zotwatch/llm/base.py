"""LLM provider base classes."""

from abc import ABC, abstractmethod

from zotwatch.core.protocols import LLMResponse


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate completion for the given prompt."""
        ...

    def available_models(self) -> list[str]:
        """List available models."""
        return []


__all__ = ["LLMResponse", "BaseLLMProvider"]
