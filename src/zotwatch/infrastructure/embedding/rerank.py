"""Voyage AI Reranker service."""

import logging
from dataclasses import dataclass

import voyageai

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Single rerank result."""

    index: int
    relevance_score: float
    document: str


class VoyageReranker:
    """Voyage AI Reranker service for semantic re-ranking of documents."""

    def __init__(self, api_key: str, model: str = "rerank-2"):
        """Initialize Voyage Reranker.

        Args:
            api_key: Voyage AI API key
            model: Rerank model name (default: rerank-2)
        """
        self.client = voyageai.Client(api_key=api_key)
        self.model = model

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: Search query (refined interests)
            documents: List of document texts (title + abstract)
            top_k: Number of top results to return

        Returns:
            List of (original_index, relevance_score) tuples, sorted by score descending
        """
        if not documents:
            return []

        # Ensure top_k doesn't exceed document count
        top_k = min(top_k, len(documents))

        logger.info(
            "Reranking %d documents with query (top_k=%d)",
            len(documents),
            top_k,
        )

        try:
            result = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_k=top_k,
            )

            rerank_results = [(r.index, r.relevance_score) for r in result.results]

            logger.info(
                "Reranking complete: %d results, top score=%.4f",
                len(rerank_results),
                rerank_results[0][1] if rerank_results else 0.0,
            )

            return rerank_results

        except Exception as e:
            logger.error("Reranking failed: %s", e)
            raise

    def rerank_with_details(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[RerankResult]:
        """Rerank documents and return detailed results.

        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top results to return

        Returns:
            List of RerankResult objects with full details
        """
        if not documents:
            return []

        top_k = min(top_k, len(documents))

        result = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_k=top_k,
        )

        return [
            RerankResult(
                index=r.index,
                relevance_score=r.relevance_score,
                document=documents[r.index],
            )
            for r in result.results
        ]


__all__ = ["VoyageReranker", "RerankResult"]
