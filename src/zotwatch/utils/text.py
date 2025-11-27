"""Text processing utilities for ZotWatch."""

import html
import json
import re
from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def iter_batches(items: Sequence[T], batch_size: int) -> Iterable[Sequence[T]]:
    """Yield batches of items."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def json_dumps(data: Any, *, indent: int | None = None) -> str:
    """Serialize data to JSON string."""
    return json.dumps(data, ensure_ascii=False, indent=indent, sort_keys=True)


def chunk_dict(d: dict[str, Any], *, max_len: int = 80) -> dict[str, Any]:
    """Truncate long string values in dictionary."""
    result = {}
    for key, value in d.items():
        if isinstance(value, str) and len(value) > max_len:
            result[key] = value[:max_len] + "..."
        else:
            result[key] = value
    return result


def clean_title(value: str | None) -> str:
    """Clean and normalize title string."""
    if not value:
        return ""
    return value.strip()


def clean_html(value: str | None) -> str | None:
    """Clean HTML tags from string."""
    if not value:
        return None
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


__all__ = [
    "iter_batches",
    "json_dumps",
    "chunk_dict",
    "clean_title",
    "clean_html",
]
