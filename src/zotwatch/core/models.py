"""Core domain models for ZotWatch."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ZoteroItem(BaseModel):
    """Represents an item from user's Zotero library."""

    key: str
    version: int
    title: str
    abstract: Optional[str] = None
    creators: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    collections: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    raw: Dict[str, object] = Field(default_factory=dict)
    content_hash: Optional[str] = None  # Hash of content used for embedding

    def content_for_embedding(self) -> str:
        """Generate text content for embedding."""
        parts = [self.title]
        if self.abstract:
            parts.append(self.abstract)
        if self.creators:
            parts.append("; ".join(self.creators))
        if self.tags:
            parts.append("; ".join(self.tags))
        return "\n".join(filter(None, parts))

    @classmethod
    def from_zotero_api(cls, item: Dict[str, object]) -> "ZoteroItem":
        """Parse item from Zotero API response."""
        data = item.get("data", {})
        creators = [
            " ".join(filter(None, [c.get("firstName"), c.get("lastName")])).strip() for c in data.get("creators", [])
        ]
        return cls(
            key=data.get("key") or item.get("key"),
            version=data.get("version") or item.get("version", 0),
            title=data.get("title") or "",
            abstract=data.get("abstractNote"),
            creators=[c for c in creators if c],
            tags=[t.get("tag") for t in data.get("tags", []) if isinstance(t, dict)],
            collections=data.get("collections", []),
            year=_safe_int(data.get("date")),
            doi=data.get("DOI"),
            url=data.get("url"),
            raw=item,
        )


def _safe_int(value: Optional[str]) -> Optional[int]:
    """Safely parse year from date string."""
    if not value:
        return None
    for part in value.split("-"):
        if part.isdigit():
            return int(part)
    return None


class CandidateWork(BaseModel):
    """Represents a candidate paper from external sources."""

    source: str
    identifier: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    doi: Optional[str] = None
    url: Optional[str] = None
    published: Optional[datetime] = None
    venue: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    extra: Dict[str, object] = Field(default_factory=dict)

    def content_for_embedding(self) -> str:
        """Generate text content for embedding."""
        parts = [self.title]
        if self.abstract:
            parts.append(self.abstract)
        if self.authors:
            parts.append("; ".join(self.authors))
        return "\n".join(filter(None, parts))


class RankedWork(CandidateWork):
    """Extends CandidateWork with scoring information."""

    score: float  # Final score (equals similarity)
    similarity: float  # Embedding similarity
    label: str  # must_read/consider/ignore
    summary: Optional["PaperSummary"] = None


class FeaturedWork(RankedWork):
    """Featured paper with rerank score."""

    rerank_score: float


class RefinedInterests(BaseModel):
    """LLM-refined research interests."""

    refined_query: str
    include_keywords: List[str] = Field(default_factory=list)
    exclude_keywords: List[str] = Field(default_factory=list)


@dataclass
class ProfileArtifacts:
    """Paths to profile artifact files."""

    sqlite_path: str
    faiss_path: str
    profile_json_path: str


# LLM Summary Models


class BulletSummary(BaseModel):
    """Short bullet-point summary."""

    research_question: str
    methodology: str
    key_findings: str
    innovation: str
    relevance_note: Optional[str] = None


class DetailedAnalysis(BaseModel):
    """Expanded detailed analysis."""

    background: str
    methodology_details: str
    results: str
    limitations: str
    future_directions: Optional[str] = None
    relevance_to_interests: str


class PaperSummary(BaseModel):
    """Complete paper summary with both formats."""

    paper_id: str
    bullets: BulletSummary
    detailed: DetailedAnalysis
    model_used: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    tokens_used: int = 0


class OverallSummary(BaseModel):
    """Overall summary for a section of papers."""

    section_type: str  # "featured" or "similarity"
    summary_text: str  # 4-6 sentences in Chinese
    paper_count: int
    key_themes: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str
    tokens_used: int = 0


# Update forward reference
RankedWork.model_rebuild()


__all__ = [
    "ZoteroItem",
    "CandidateWork",
    "RankedWork",
    "FeaturedWork",
    "RefinedInterests",
    "ProfileArtifacts",
    "BulletSummary",
    "DetailedAnalysis",
    "PaperSummary",
    "OverallSummary",
]
