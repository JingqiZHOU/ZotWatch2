"""Paper summarization service."""

import hashlib
import json
import logging
from dataclasses import dataclass

from zotwatch.core.models import (
    BulletSummary,
    DetailedAnalysis,
    PaperSummary,
    RankedWork,
)
from zotwatch.infrastructure.storage import ProfileStorage
from zotwatch.utils.datetime import utc_now

from .base import BaseLLMProvider
from .prompts import BULLET_SUMMARY_PROMPT, DETAILED_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class SummarizationResult:
    """Result of batch summarization.

    Attributes:
        summaries: Successfully generated summaries.
        failed_ids: Paper identifiers that failed to summarize.
    """

    summaries: list[PaperSummary]
    failed_ids: list[str]

    @property
    def success_count(self) -> int:
        """Number of successfully summarized papers."""
        return len(self.summaries)

    @property
    def failure_count(self) -> int:
        """Number of papers that failed to summarize."""
        return len(self.failed_ids)


class PaperSummarizer:
    """Service for generating and caching paper summaries."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        storage: ProfileStorage | None = None,
        model: str | None = None,
    ):
        self.llm = llm
        self.storage = storage
        self.model = model
        if self.storage:
            self._ensure_cache_signature()

    def summarize(self, work: RankedWork, *, force: bool = False) -> PaperSummary:
        """Generate or retrieve cached summary for a paper."""
        paper_id = work.identifier

        # Check cache first
        if not force and self.storage:
            cached = self.storage.get_summary(paper_id)
            if cached:
                logger.debug("Using cached summary for %s", paper_id)
                return cached

        # Generate bullet summary
        bullets_prompt = BULLET_SUMMARY_PROMPT.format(
            title=work.title,
            abstract=work.abstract or "No abstract available",
            authors=", ".join(work.authors[:5]) if work.authors else "Unknown",
            venue=work.venue or "Unknown",
        )
        bullets_response = self.llm.complete(bullets_prompt, model=self.model)
        bullets = self._parse_bullets(bullets_response.content)

        # Generate detailed analysis
        detailed_prompt = DETAILED_ANALYSIS_PROMPT.format(
            title=work.title,
            abstract=work.abstract or "No abstract available",
            authors=", ".join(work.authors[:5]) if work.authors else "Unknown",
            venue=work.venue or "Unknown",
        )
        detailed_response = self.llm.complete(detailed_prompt, model=self.model)
        detailed = self._parse_detailed(detailed_response.content)

        # Create summary
        summary = PaperSummary(
            paper_id=paper_id,
            bullets=bullets,
            detailed=detailed,
            model_used=bullets_response.model,
            generated_at=utc_now(),
            tokens_used=bullets_response.tokens_used + detailed_response.tokens_used,
        )

        # Cache summary
        if self.storage:
            self.storage.save_summary(paper_id, summary)
            logger.info("Generated and cached summary for %s using %s", paper_id, summary.model_used)
        else:
            logger.info("Generated summary for %s using %s", paper_id, summary.model_used)

        return summary

    def _summary_cache_signature(self) -> str:
        payload = {
            "provider": self.llm.name,
            "model": self.model or "",
            "bullet_prompt": BULLET_SUMMARY_PROMPT,
            "detailed_prompt": DETAILED_ANALYSIS_PROMPT,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _ensure_cache_signature(self) -> None:
        signature = self._summary_cache_signature()
        self.storage.ensure_summary_cache_signature(signature)

    def _parse_bullets(self, content: str) -> BulletSummary:
        """Parse bullet summary from LLM response."""
        try:
            # Try to extract JSON from response
            content = content.strip()
            if content.startswith("```"):
                # Remove markdown code blocks
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
            return BulletSummary(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to parse bullet summary: %s", e)
            return BulletSummary(
                research_question="Unable to extract research question",
                methodology="Unable to extract methodology",
                key_findings="Unable to extract findings",
                innovation="Unable to extract innovation",
                relevance_note=content[:200] if content else None,
            )

    def _parse_detailed(self, content: str) -> DetailedAnalysis:
        """Parse detailed analysis from LLM response."""
        try:
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
            return DetailedAnalysis(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to parse detailed analysis: %s", e)
            return DetailedAnalysis(
                background="Unable to extract background",
                methodology_details="Unable to extract methodology details",
                results="Unable to extract results",
                limitations="Unable to extract limitations",
                future_directions=None,
                relevance_to_interests=content[:200] if content else "Unable to determine relevance",
            )

    def summarize_batch(
        self,
        works: list[RankedWork],
        *,
        force: bool = False,
        limit: int | None = None,
    ) -> SummarizationResult:
        """Generate summaries for multiple papers.

        Args:
            works: List of ranked works to summarize.
            force: If True, regenerate even if cached.
            limit: Maximum number of papers to summarize.

        Returns:
            SummarizationResult containing successful summaries and failed paper IDs.
        """
        if limit:
            works = works[:limit]

        summaries: list[PaperSummary] = []
        failed_ids: list[str] = []

        for i, work in enumerate(works):
            try:
                logger.info("Summarizing paper %d/%d: %s", i + 1, len(works), work.title[:50])
                summary = self.summarize(work, force=force)
                summaries.append(summary)
            except Exception as e:
                logger.error("Failed to summarize %s: %s", work.identifier, e)
                failed_ids.append(work.identifier)

        if failed_ids:
            logger.warning(
                "Summarization completed with %d failures out of %d papers",
                len(failed_ids),
                len(works),
            )

        return SummarizationResult(summaries=summaries, failed_ids=failed_ids)


__all__ = ["PaperSummarizer", "SummarizationResult"]
