"""RSS feed generation with Dublin Core and PRISM namespaces for Zotero compatibility."""

import html
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from zotwatch.core.models import RankedWork

logger = logging.getLogger(__name__)

# Namespace URIs
NS_DC = "http://purl.org/dc/elements/1.1/"
NS_PRISM = "http://prismstandard.org/namespaces/basic/3.0/"
NS_CONTENT = "http://purl.org/rss/1.0/modules/content/"

# Register namespaces for proper prefix handling
ET.register_namespace("dc", NS_DC)
ET.register_namespace("prism", NS_PRISM)
ET.register_namespace("content", NS_CONTENT)


def write_rss(
    works: Iterable[RankedWork],
    output_path: Path | str,
    *,
    title: str = "ZotWatch Feed",
    link: str = "https://example.com",
    description: str = "AI-assisted literature watch",
) -> Path:
    """Write RSS feed from ranked works.

    Args:
        works: Ranked works to include in feed
        output_path: Path to write RSS file
        title: Feed title
        link: Feed link
        description: Feed description

    Returns:
        Path to written RSS file
    """
    works_list = list(works)
    # Create RSS element - namespace declarations are added automatically
    # by ET.register_namespace() when elements with those namespaces are used
    rss = ET.Element("rss", {"version": "2.0"})
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = title
    ET.SubElement(channel, "link").text = link
    ET.SubElement(channel, "description").text = description
    ET.SubElement(channel, "lastBuildDate").text = _format_rfc822(datetime.now(timezone.utc))

    for work in works_list:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = work.title
        if work.url:
            ET.SubElement(item, "link").text = work.url
        ET.SubElement(item, "guid").text = work.identifier
        ET.SubElement(item, "pubDate").text = _format_rfc822(work.published)

        # Dublin Core: Authors (one element per author)
        for author in work.authors:
            ET.SubElement(item, f"{{{NS_DC}}}creator").text = author

        # Dublin Core: Item type (for Zotero 6.0+ preprint support)
        if work.source == "arxiv":
            ET.SubElement(item, f"{{{NS_DC}}}type").text = "preprint"

        # PRISM: Publication metadata
        if work.venue:
            ET.SubElement(item, f"{{{NS_PRISM}}}publicationName").text = work.venue
        if work.doi:
            ET.SubElement(item, f"{{{NS_PRISM}}}doi").text = work.doi

        # Plain text description (for basic RSS readers)
        description_lines = []
        if work.abstract:
            description_lines.append(work.abstract)
        published_text = work.published.isoformat() if work.published else "Unknown"
        description_lines.append(f"Published: {published_text}")
        description_lines.append(f"Venue: {work.venue or 'Unknown'}")
        description_lines.append(f"Score: {work.score:.3f} ({work.label})")
        ET.SubElement(item, "description").text = "\n".join(description_lines)

        # HTML-formatted content (for enhanced display)
        html_content = _build_html_content(work)
        content_elem = ET.SubElement(item, f"{{{NS_CONTENT}}}encoded")
        content_elem.text = html_content

    tree = ET.ElementTree(rss)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    logger.info("Wrote RSS feed with %d items to %s", len(works_list), path)
    return path


def _format_rfc822(dt: datetime | None) -> str:
    """Format datetime as RFC 822."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")


def _build_html_content(work: RankedWork) -> str:
    """Build HTML-formatted content for content:encoded element.

    Args:
        work: The ranked work to format

    Returns:
        HTML string wrapped in CDATA-style format
    """
    parts = []

    # Authors
    if work.authors:
        authors_str = html.escape("; ".join(work.authors))
        parts.append(f"<p><strong>Authors:</strong> {authors_str}</p>")

    # Journal/Venue
    if work.venue:
        venue_str = html.escape(work.venue)
        parts.append(f"<p><strong>Journal:</strong> {venue_str}</p>")

    # DOI
    if work.doi:
        doi_str = html.escape(work.doi)
        parts.append(f'<p><strong>DOI:</strong> <a href="https://doi.org/{doi_str}">{doi_str}</a></p>')

    # Score and label
    parts.append(f"<p><strong>Relevance:</strong> {work.score:.3f} ({work.label})</p>")

    # Abstract
    if work.abstract:
        abstract_str = html.escape(work.abstract)
        parts.append(f"<p>{abstract_str}</p>")

    return "\n".join(parts)


__all__ = ["write_rss"]
