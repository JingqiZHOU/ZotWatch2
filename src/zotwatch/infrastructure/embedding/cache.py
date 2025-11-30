"""Unified embedding cache storage layer."""

import logging
from datetime import timedelta

from zotwatch.infrastructure.cache_base import BaseSQLiteCache
from zotwatch.utils.datetime import format_sqlite_datetime, utc_now

logger = logging.getLogger(__name__)


class EmbeddingCache(BaseSQLiteCache):
    """Unified embedding cache with SQLite backend.

    Stores embeddings with composite key (content_hash, model) to support
    automatic invalidation when switching embedding models.
    """

    def _ensure_schema(self) -> None:
        """Create embeddings table if not exists."""
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                source_type TEXT NOT NULL,
                source_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                PRIMARY KEY (content_hash, model)
            );

            CREATE INDEX IF NOT EXISTS idx_emb_expires
                ON embeddings(expires_at) WHERE expires_at IS NOT NULL;

            CREATE INDEX IF NOT EXISTS idx_emb_source
                ON embeddings(source_type);

            CREATE INDEX IF NOT EXISTS idx_emb_model
                ON embeddings(model);
        """)
        conn.commit()

    def _get_expires_column(self) -> str:
        """Return the column name for expiration timestamps."""
        return "expires_at"

    def _get_table_name(self) -> str:
        """Return the main table name."""
        return "embeddings"

    def get(self, content_hash: str, model: str) -> bytes | None:
        """Get a single cached embedding.

        Args:
            content_hash: SHA256 hash of content.
            model: Model identifier.

        Returns:
            Embedding bytes if found and not expired, None otherwise.
        """
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT embedding FROM embeddings
            WHERE content_hash = ? AND model = ?
              AND (expires_at IS NULL OR expires_at > datetime('now'))
            """,
            (content_hash, model),
        )
        row = cur.fetchone()
        return row["embedding"] if row else None

    def get_batch(
        self,
        content_hashes: list[str],
        model: str,
    ) -> dict[str, bytes]:
        """Batch fetch cached embeddings.

        Args:
            content_hashes: List of content hashes to fetch.
            model: Model identifier.

        Returns:
            Dict mapping content_hash to embedding bytes for found items.
        """
        if not content_hashes:
            return {}

        conn = self._connect()
        placeholders = ",".join("?" for _ in content_hashes)
        cur = conn.execute(
            f"""
            SELECT content_hash, embedding FROM embeddings
            WHERE content_hash IN ({placeholders})
              AND model = ?
              AND (expires_at IS NULL OR expires_at > datetime('now'))
            """,
            (*content_hashes, model),
        )
        return {row["content_hash"]: row["embedding"] for row in cur}

    def put(
        self,
        content_hash: str,
        embedding: bytes,
        model: str,
        source_type: str,
        source_id: str | None = None,
        ttl_days: int | None = None,
    ) -> None:
        """Store a single embedding (thread-safe).

        Args:
            content_hash: SHA256 hash of content.
            embedding: Embedding vector as bytes.
            model: Model identifier.
            source_type: Type of source ("profile" or "candidate").
            source_id: Optional source identifier.
            ttl_days: Time-to-live in days. None for permanent.
        """
        expires_at = None
        if ttl_days is not None:
            expires_at = format_sqlite_datetime(utc_now() + timedelta(days=ttl_days))

        with self._write_lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings
                    (content_hash, model, embedding, source_type, source_id, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (content_hash, model, embedding, source_type, source_id, expires_at),
            )
            conn.commit()

    def put_batch(
        self,
        items: list[tuple[str, bytes]],
        model: str,
        source_type: str,
        source_ids: list[str] | None = None,
        ttl_days: int | None = None,
    ) -> None:
        """Batch store embeddings (thread-safe).

        Args:
            items: List of (content_hash, embedding_bytes) tuples.
            model: Model identifier.
            source_type: Type of source ("profile" or "candidate").
            source_ids: Optional list of source identifiers (same order as items).
            ttl_days: Time-to-live in days. None for permanent.
        """
        if not items:
            return

        expires_at = None
        if ttl_days is not None:
            expires_at = format_sqlite_datetime(utc_now() + timedelta(days=ttl_days))

        if source_ids is None:
            source_ids = [None] * len(items)

        with self._write_lock:
            conn = self._connect()
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                    (content_hash, model, embedding, source_type, source_id, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [(h, model, emb, source_type, sid, expires_at) for (h, emb), sid in zip(items, source_ids)],
            )
            conn.commit()

    def invalidate_model(self, model: str) -> int:
        """Delete all embeddings for a specific model.

        Args:
            model: Model identifier to invalidate.

        Returns:
            Number of deleted rows.
        """
        conn = self._connect()
        cur = conn.execute(
            "DELETE FROM embeddings WHERE model = ?",
            (model,),
        )
        count = cur.rowcount
        conn.commit()
        if count > 0:
            logger.info("Invalidated %d embeddings for model '%s'", count, model)
        return count

    def invalidate_source(self, source_type: str) -> int:
        """Delete all embeddings for a specific source type.

        Args:
            source_type: Source type to invalidate ("profile" or "candidate").

        Returns:
            Number of deleted rows.
        """
        conn = self._connect()
        cur = conn.execute(
            "DELETE FROM embeddings WHERE source_type = ?",
            (source_type,),
        )
        count = cur.rowcount
        conn.commit()
        if count > 0:
            logger.info("Invalidated %d embeddings for source '%s'", count, source_type)
        return count

    def count(self, source_type: str | None = None, model: str | None = None) -> int:
        """Count cached embeddings.

        Args:
            source_type: Optional filter by source type.
            model: Optional filter by model.

        Returns:
            Number of cached embeddings matching the criteria.
        """
        conn = self._connect()
        query = "SELECT COUNT(*) FROM embeddings WHERE 1=1"
        params: list = []

        if source_type is not None:
            query += " AND source_type = ?"
            params.append(source_type)

        if model is not None:
            query += " AND model = ?"
            params.append(model)

        cur = conn.execute(query, params)
        return cur.fetchone()[0]


__all__ = ["EmbeddingCache"]
