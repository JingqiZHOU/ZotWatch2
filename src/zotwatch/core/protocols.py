"""Protocol definitions for ZotWatch components.

Note: CandidateSource, LLMProvider, and EmbeddingProvider protocols have been
removed in favor of their ABC counterparts:
- CandidateSource -> sources.base.BaseSource
- LLMProvider -> llm.base.BaseLLMProvider
- EmbeddingProvider -> infrastructure.embedding.base.BaseEmbeddingProvider
"""

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

from .models import PaperSummary, ZoteroItem


@dataclass
class LLMResponse:
    """Response from LLM provider."""

    content: str
    model: str
    tokens_used: int
    cached: bool = False


@runtime_checkable
class ItemStorage(Protocol):
    """Protocol for item storage backends."""

    def initialize(self) -> None:
        """Initialize storage schema."""
        ...

    def upsert_item(self, item: ZoteroItem, content_hash: str | None = None) -> None:
        """Insert or update an item."""
        ...

    def remove_items(self, keys: Iterable[str]) -> None:
        """Remove items by keys."""
        ...

    def iter_items(self) -> Iterable[ZoteroItem]:
        """Iterate over all items."""
        ...

    def get_metadata(self, key: str) -> str | None:
        """Get metadata value by key."""
        ...

    def set_metadata(self, key: str, value: str) -> None:
        """Set metadata value."""
        ...


@runtime_checkable
class SummaryStorage(Protocol):
    """Protocol for LLM summary storage."""

    def get_summary(self, paper_id: str) -> PaperSummary | None:
        """Get cached summary by paper ID."""
        ...

    def save_summary(self, paper_id: str, summary: PaperSummary) -> None:
        """Save summary to cache."""
        ...

    def has_summary(self, paper_id: str) -> bool:
        """Check if summary exists."""
        ...


__all__ = [
    "LLMResponse",
    "ItemStorage",
    "SummaryStorage",
]
