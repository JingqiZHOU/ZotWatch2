"""Metadata cache storage layer for paper enrichment."""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MetadataCache:
    """Cache for paper metadata from external APIs.

    Stores paper abstracts and other metadata with TTL support.
    Uses SQLite backend similar to EmbeddingCache pattern.

    Thread-safe: uses write lock for concurrent access.
    """

    def __init__(self, db_path: Path | str) -> None:
        """Initialize the cache.

        Args:
            db_path: Path to SQLite database file.
        """
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._write_lock = threading.Lock()  # Protects concurrent writes
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            # Ensure parent directory exists
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        """Create metadata table if not exists."""
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS paper_metadata (
                doi TEXT PRIMARY KEY,
                abstract TEXT,
                title TEXT,
                authors_json TEXT,
                citation_count INTEGER,
                source TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_meta_expires
                ON paper_metadata(expires_at) WHERE expires_at IS NOT NULL;

            CREATE INDEX IF NOT EXISTS idx_meta_source
                ON paper_metadata(source);
        """)
        conn.commit()

    def get_abstract(self, doi: str) -> Optional[str]:
        """Get cached abstract for DOI.

        Args:
            doi: Digital Object Identifier.

        Returns:
            Abstract text if found and not expired, None otherwise.
        """
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT abstract FROM paper_metadata
            WHERE doi = ?
              AND (expires_at IS NULL OR expires_at > datetime('now'))
            """,
            (doi.lower(),),
        )
        row = cur.fetchone()
        return row["abstract"] if row else None

    def get_batch(self, dois: List[str]) -> Dict[str, str]:
        """Batch fetch cached abstracts.

        Args:
            dois: List of DOIs to fetch.

        Returns:
            Dict mapping DOI to abstract for found items.
        """
        if not dois:
            return {}

        # Normalize DOIs to lowercase
        normalized = [d.lower() for d in dois]
        doi_map = {d.lower(): d for d in dois}  # Map back to original case

        conn = self._connect()
        placeholders = ",".join("?" for _ in normalized)
        cur = conn.execute(
            f"""
            SELECT doi, abstract FROM paper_metadata
            WHERE doi IN ({placeholders})
              AND abstract IS NOT NULL
              AND (expires_at IS NULL OR expires_at > datetime('now'))
            """,
            normalized,
        )
        # Return with original DOI case
        return {doi_map.get(row["doi"], row["doi"]): row["abstract"] for row in cur}

    def put(
        self,
        doi: str,
        abstract: Optional[str],
        source: str,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        citation_count: Optional[int] = None,
        ttl_days: int = 30,
    ) -> None:
        """Store paper metadata with TTL (thread-safe).

        Args:
            doi: Digital Object Identifier.
            abstract: Paper abstract text.
            source: Source identifier (e.g., "semantic_scholar").
            title: Paper title.
            authors: List of author names.
            citation_count: Citation count.
            ttl_days: Time-to-live in days.
        """
        expires_at = (datetime.now() + timedelta(days=ttl_days)).isoformat()
        authors_json = json.dumps(authors) if authors else None

        with self._write_lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_metadata
                    (doi, abstract, title, authors_json, citation_count, source, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (doi.lower(), abstract, title, authors_json, citation_count, source, expires_at),
            )
            conn.commit()

    def put_batch(
        self,
        items: List[Tuple[str, Optional[str]]],
        source: str,
        ttl_days: int = 30,
    ) -> None:
        """Batch store abstracts (thread-safe).

        Args:
            items: List of (doi, abstract) tuples.
            source: Source identifier.
            ttl_days: Time-to-live in days.
        """
        if not items:
            return

        expires_at = (datetime.now() + timedelta(days=ttl_days)).isoformat()

        with self._write_lock:
            conn = self._connect()
            conn.executemany(
                """
                INSERT OR REPLACE INTO paper_metadata
                    (doi, abstract, source, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                [(doi.lower(), abstract, source, expires_at) for doi, abstract in items],
            )
            conn.commit()

    def cleanup_expired(self) -> int:
        """Remove expired metadata entries.

        Returns:
            Number of deleted rows.
        """
        conn = self._connect()
        cur = conn.execute(
            """
            DELETE FROM paper_metadata
            WHERE expires_at IS NOT NULL AND expires_at <= datetime('now')
            """
        )
        count = cur.rowcount
        conn.commit()
        if count > 0:
            logger.info("Cleaned up %d expired metadata cache entries", count)
        return count

    def count(self, source: Optional[str] = None) -> int:
        """Count cached metadata entries.

        Args:
            source: Optional filter by source.

        Returns:
            Number of cached entries.
        """
        conn = self._connect()
        if source:
            cur = conn.execute(
                "SELECT COUNT(*) FROM paper_metadata WHERE source = ?",
                (source,),
            )
        else:
            cur = conn.execute("SELECT COUNT(*) FROM paper_metadata")
        return cur.fetchone()[0]

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


__all__ = ["MetadataCache"]
