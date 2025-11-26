"""Abstract enrichment pipeline for candidates with missing abstracts."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from zotwatch.config.settings import Settings
from zotwatch.core.models import CandidateWork
from zotwatch.infrastructure.enrichment.cache import MetadataCache
from zotwatch.infrastructure.enrichment.publisher_scraper import UniversalScraper
from zotwatch.infrastructure.enrichment.semantic_scholar import SemanticScholarClient
from zotwatch.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentStats:
    """Statistics from the enrichment process."""

    total_candidates: int
    with_abstract: int
    missing_abstracts: int
    skipped_no_doi: int
    cache_hits: int
    api_fetched: int
    scraper_fetched: int = 0  # Abstracts fetched via universal scraper
    enriched: int = 0
    failed: int = 0

    @property
    def original_rate(self) -> float:
        """Original abstract completeness rate before enrichment."""
        if self.total_candidates == 0:
            return 0.0
        return self.with_abstract / self.total_candidates * 100

    @property
    def final_rate(self) -> float:
        """Final abstract completeness rate after enrichment."""
        if self.total_candidates == 0:
            return 0.0
        return (self.with_abstract + self.enriched) / self.total_candidates * 100


class AbstractEnricher:
    """Enriches candidates with missing abstracts from multiple sources.

    Uses a three-tier strategy:
    1. Check local cache first (SQLite-backed)
    2. Query Semantic Scholar API for cache misses
    3. Use universal scraper (Firefox + LLM) for remaining papers
    """

    def __init__(
        self,
        settings: Settings,
        base_dir: Path,
        llm: Optional[BaseLLMProvider] = None,
        client: Optional[SemanticScholarClient] = None,
        cache: Optional[MetadataCache] = None,
    ):
        """Initialize the enricher.

        Args:
            settings: Application settings.
            base_dir: Base directory for data files.
            llm: LLM provider for universal scraper extraction.
            client: Optional pre-configured client (for testing).
            cache: Optional pre-configured cache (for testing).
        """
        self.config = settings.sources.semantic_scholar
        self.base_dir = Path(base_dir)
        self.llm = llm

        # Initialize cache
        if cache is not None:
            self.cache = cache
        else:
            cache_path = self.base_dir / "data" / "metadata_cache.sqlite"
            self.cache = MetadataCache(cache_path)

        # Initialize client
        if client is not None:
            self.client = client
        else:
            api_key = self.config.api_key if self.config.api_key else None
            self.client = SemanticScholarClient(
                api_key=api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                backoff_factor=self.config.backoff_factor,
                rate_limit_delay=self.config.rate_limit_delay,
            )

    def enrich(self, candidates: List[CandidateWork]) -> Tuple[List[CandidateWork], EnrichmentStats]:
        """Enrich candidates with missing abstracts.

        Args:
            candidates: List of candidate works.

        Returns:
            Tuple of (enriched candidates, statistics).
        """
        if not self.config.enabled:
            logger.debug("Semantic Scholar enrichment is disabled")
            with_abstract = sum(1 for c in candidates if c.abstract)
            return candidates, EnrichmentStats(
                total_candidates=len(candidates),
                with_abstract=with_abstract,
                missing_abstracts=len(candidates) - with_abstract,
                skipped_no_doi=0,
                cache_hits=0,
                api_fetched=0,
                enriched=0,
                failed=0,
            )

        # Categorize candidates
        with_abstract = []
        needs_enrichment = []
        no_doi = []

        for c in candidates:
            if c.abstract:
                with_abstract.append(c)
            elif c.doi:
                needs_enrichment.append(c)
            else:
                no_doi.append(c)

        logger.info(
            "Abstract status: %d/%d (%.1f%%) have abstracts, %d need enrichment, %d have no DOI",
            len(with_abstract),
            len(candidates),
            len(with_abstract) / len(candidates) * 100 if candidates else 0,
            len(needs_enrichment),
            len(no_doi),
        )

        if not needs_enrichment:
            return candidates, EnrichmentStats(
                total_candidates=len(candidates),
                with_abstract=len(with_abstract),
                missing_abstracts=0,
                skipped_no_doi=len(no_doi),
                cache_hits=0,
                api_fetched=0,
                enriched=0,
                failed=0,
            )

        # Step 1: Check cache
        dois_to_check = [c.doi for c in needs_enrichment]
        cached_abstracts = self.cache.get_batch(dois_to_check)
        cache_hits = len(cached_abstracts)

        logger.debug("Cache hits: %d/%d", cache_hits, len(dois_to_check))

        # Step 2: Query Semantic Scholar API for cache misses
        # Results are cached immediately inside _fetch_abstracts
        uncached_dois = [doi for doi in dois_to_check if doi not in cached_abstracts]
        api_abstracts: Dict[str, str] = {}

        if uncached_dois:
            logger.info("Querying Semantic Scholar for %d papers...", len(uncached_dois))
            api_abstracts = self._fetch_abstracts(uncached_dois)

        # Step 3: Universal scraper fallback for remaining DOIs
        # Results are cached immediately inside _fetch_with_scraper
        still_missing_dois = [doi for doi in uncached_dois if doi not in api_abstracts]
        scraper_abstracts: Dict[str, str] = {}

        if still_missing_dois and self.config.scraper.enabled and self.llm:
            logger.info("Universal scraper: fetching %d papers...", len(still_missing_dois))
            scraper_abstracts = self._fetch_with_scraper(still_missing_dois, needs_enrichment)
        elif still_missing_dois and self.config.scraper.enabled and not self.llm:
            logger.debug("Universal scraper skipped: LLM provider not available")

        # Merge results from all sources
        all_abstracts = {**cached_abstracts, **api_abstracts, **scraper_abstracts}

        # Step 4: Apply abstracts to candidates
        enriched_count = 0
        for candidate in needs_enrichment:
            if candidate.doi in all_abstracts:
                candidate.abstract = all_abstracts[candidate.doi]
                enriched_count += 1

        failed = len(needs_enrichment) - enriched_count

        stats = EnrichmentStats(
            total_candidates=len(candidates),
            with_abstract=len(with_abstract),
            missing_abstracts=len(needs_enrichment),
            skipped_no_doi=len(no_doi),
            cache_hits=cache_hits,
            api_fetched=len(api_abstracts),
            scraper_fetched=len(scraper_abstracts),
            enriched=enriched_count,
            failed=failed,
        )

        logger.info(
            "Enrichment complete: %d/%d abstracts added (cache: %d, S2 API: %d, scraper: %d, not found: %d)",
            stats.enriched,
            stats.missing_abstracts,
            stats.cache_hits,
            stats.api_fetched,
            stats.scraper_fetched,
            stats.failed,
        )

        # Provide helpful context about unindexed papers
        if stats.failed > 0 and stats.failed > stats.enriched:
            logger.info(
                "Note: %d papers not found in any source - this may be due to indexing delays or restricted access",
                stats.failed,
            )

        logger.info(
            "Abstract rate: %.1f%% -> %.1f%%",
            stats.original_rate,
            stats.final_rate,
        )

        return candidates, stats

    def _fetch_abstracts(self, dois: List[str]) -> Dict[str, str]:
        """Fetch abstracts from Semantic Scholar in batches.

        Caches each batch immediately after fetching to prevent data loss.

        Args:
            dois: List of DOIs to fetch.

        Returns:
            Dict mapping DOI to abstract.
        """
        results: Dict[str, str] = {}
        batch_size = self.config.batch_size

        for i in range(0, len(dois), batch_size):
            batch = dois[i : i + batch_size]
            try:
                batch_results = self.client.get_abstracts_batch(batch)
                results.update(batch_results)

                # Cache immediately after each batch to prevent data loss
                if batch_results:
                    self.cache.put_batch(
                        [(doi, abstract) for doi, abstract in batch_results.items()],
                        source="semantic_scholar",
                        ttl_days=self.config.cache_ttl_days,
                    )
                    logger.debug(
                        "Batch %d-%d: fetched and cached %d/%d abstracts",
                        i,
                        i + len(batch),
                        len(batch_results),
                        len(batch),
                    )
            except Exception as e:
                logger.warning("Failed to fetch batch %d-%d: %s", i, i + len(batch), e)

        return results

    def _fetch_with_scraper(
        self,
        dois: List[str],
        candidates: List[CandidateWork],
    ) -> Dict[str, str]:
        """Fetch abstracts using universal scraper (Firefox + LLM).

        Caches each result immediately after fetching to prevent data loss.

        Args:
            dois: List of DOIs to fetch.
            candidates: List of candidates (for title context).

        Returns:
            Dict mapping DOI to abstract.
        """
        if not self.llm:
            return {}

        scraper_config = self.config.scraper

        # Create DOI -> title mapping for LLM context
        doi_to_title = {c.doi: c.title for c in candidates if c.doi}

        scraper = UniversalScraper(
            llm=self.llm,
            rate_limit_delay=scraper_config.rate_limit_delay,
            timeout=scraper_config.timeout,
            max_html_chars=scraper_config.max_html_chars,
            llm_max_tokens=scraper_config.llm_max_tokens,
            llm_temperature=scraper_config.llm_temperature,
        )

        results: Dict[str, str] = {}
        failed_dois: List[str] = []
        total = len(dois)

        try:
            # First pass: fetch all DOIs
            for idx, doi in enumerate(dois, 1):
                title = doi_to_title.get(doi)
                logger.info("Scraper [%d/%d]: fetching %s", idx, total, doi)
                abstract = scraper.fetch_abstract(doi, title)

                if abstract:
                    results[doi] = abstract
                    # Cache immediately after each successful fetch
                    self.cache.put(
                        doi=doi,
                        abstract=abstract,
                        source="llm_scraper",
                        title=title,
                        ttl_days=self.config.cache_ttl_days,
                    )
                    logger.info("Scraper [%d/%d]: success (%d chars)", idx, total, len(abstract))
                else:
                    logger.info("Scraper [%d/%d]: no abstract found", idx, total)
                    failed_dois.append(doi)

            # Second pass: retry failed DOIs once
            if failed_dois:
                logger.info("Retrying %d failed DOIs...", len(failed_dois))
                retry_total = len(failed_dois)
                for idx, doi in enumerate(failed_dois, 1):
                    title = doi_to_title.get(doi)
                    logger.info("Retry [%d/%d]: fetching %s", idx, retry_total, doi)
                    abstract = scraper.fetch_abstract(doi, title)

                    if abstract:
                        results[doi] = abstract
                        self.cache.put(
                            doi=doi,
                            abstract=abstract,
                            source="llm_scraper",
                            title=title,
                            ttl_days=self.config.cache_ttl_days,
                        )
                        logger.info("Retry [%d/%d]: success (%d chars)", idx, retry_total, len(abstract))
                    else:
                        logger.info("Retry [%d/%d]: still failed", idx, retry_total)

            if results:
                logger.info("Universal scraper: fetched %d/%d abstracts", len(results), total)
            return results
        finally:
            scraper.close()


def enrich_candidates(
    candidates: List[CandidateWork],
    settings: Settings,
    base_dir: Path,
    llm: Optional[BaseLLMProvider] = None,
) -> Tuple[List[CandidateWork], EnrichmentStats]:
    """Convenience function to enrich candidates with missing abstracts.

    Args:
        candidates: List of candidate works.
        settings: Application settings.
        base_dir: Base directory for data files.
        llm: Optional LLM provider for universal scraper.

    Returns:
        Tuple of (enriched candidates, statistics).
    """
    enricher = AbstractEnricher(settings, base_dir, llm=llm)
    return enricher.enrich(candidates)


__all__ = ["AbstractEnricher", "EnrichmentStats", "enrich_candidates"]
