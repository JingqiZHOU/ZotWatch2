"""Core domain models and interfaces."""

from .models import (
    ZoteroItem,
    CandidateWork,
    RankedWork,
    ProfileArtifacts,
    BulletSummary,
    DetailedAnalysis,
    PaperSummary,
)
from .protocols import (
    LLMResponse,
    ItemStorage,
    SummaryStorage,
)
from .exceptions import (
    ZotWatchError,
    ConfigurationError,
    SourceFetchError,
    EmbeddingError,
    LLMError,
)

__all__ = [
    # Models
    "ZoteroItem",
    "CandidateWork",
    "RankedWork",
    "ProfileArtifacts",
    "BulletSummary",
    "DetailedAnalysis",
    "PaperSummary",
    # Protocols
    "LLMResponse",
    "ItemStorage",
    "SummaryStorage",
    # Exceptions
    "ZotWatchError",
    "ConfigurationError",
    "SourceFetchError",
    "EmbeddingError",
    "LLMError",
]
