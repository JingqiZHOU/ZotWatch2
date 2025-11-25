"""Crossref source implementation."""

import csv
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests

from zotwatch.config.settings import Settings
from zotwatch.core.models import CandidateWork

from .base import BaseSource, SourceRegistry, clean_html, clean_title, parse_date

# Crossref member IDs for major publishers (aligned with OpenAlex publisher list)
# Verified via https://api.crossref.org/members?query=<name> and DOI prefix lookups
MEMBER_IDS: Dict[str, int] = {
    "Institute of Electrical and Electronics Engineers": 263,  # IEEE
    "Springer Nature": 297,  # Springer Science and Business Media LLC
    "Elsevier BV": 78,
    "SPIE": 189,  # SPIE-Intl Soc Optical Eng (prefix 10.1117)
    "Wiley": 311,
    "Taylor & Francis": 301,  # Informa UK Limited
    "Multidisciplinary Digital Publishing Institute": 1968,  # MDPI AG
    "Frontiers Media": 1965,  # Frontiers Media SA
}

logger = logging.getLogger(__name__)


@SourceRegistry.register
class CrossrefSource(BaseSource):
    """Crossref journal articles source."""

    def __init__(self, settings: Settings, profile_path: Optional[Path] = None):
        super().__init__(settings)
        self.config = settings.sources.crossref
        self.session = requests.Session()
        self.profile_path = profile_path
        self._top_venues: Optional[List[str]] = None

    @property
    def name(self) -> str:
        return "crossref"

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def top_venues(self) -> List[str]:
        """Load top venues from profile."""
        if self._top_venues is not None:
            return self._top_venues

        if not self.profile_path or not self.profile_path.exists():
            self._top_venues = []
            return self._top_venues

        try:
            data = json.loads(self.profile_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load profile when reading top venues: %s", exc)
            self._top_venues = []
            return self._top_venues

        venues: List[str] = []
        for entry in data.get("top_venues", []):
            name = entry.get("venue") if isinstance(entry, dict) else None
            if name:
                venues.append(name)

        unique = list(dict.fromkeys(venues)) if venues else []
        self._top_venues = unique[:20]
        return self._top_venues

    def set_profile_path(self, path: Path) -> None:
        """Set profile path for top venues loading."""
        self.profile_path = path
        self._top_venues = None

    def fetch(self, days_back: int | None = None) -> List[CandidateWork]:
        """Fetch Crossref works with publisher filtering and abstract requirement."""
        if days_back is None:
            days_back = self.config.days_back

        max_results = self.config.max_results

        # Use ISSN whitelist if enabled
        if self.config.use_issn_whitelist:
            issns = self._load_issn_whitelist()
            if issns:
                logger.info("Using ISSN whitelist with %d journals", len(issns))
                results = self._fetch_by_issn(days_back, issns, max_results)
                results.extend(self._fetch_top_venues(days_back))
                return results

        # Fall back to publisher filter
        member_ids = self._get_member_ids()
        if member_ids:
            logger.info("Filtering by publishers: %s", ", ".join(self.config.publishers))
            results = self._fetch_with_filters(days_back, member_ids, max_results)
        else:
            results = self._fetch_general(days_back)

        results.extend(self._fetch_top_venues(days_back))
        return results

    def _load_issn_whitelist(self) -> List[str]:
        """Load ISSN whitelist from CSV file."""
        # Look for whitelist in data directory
        path = Path(__file__).parents[3] / "data" / "journal_whitelist.csv"
        if not path.exists():
            logger.warning("Journal whitelist not found: %s", path)
            return []

        issns: List[str] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    issn = (row.get("issn") or "").strip()
                    if issn:
                        issns.append(issn)
        except Exception as exc:
            logger.warning("Failed to load journal whitelist: %s", exc)
            return []

        logger.info("Loaded %d ISSNs from whitelist", len(issns))
        return issns

    def _fetch_by_issn(
        self,
        days_back: int,
        issns: List[str],
        max_results: int,
    ) -> List[CandidateWork]:
        """Fetch works from specific journals by ISSN."""
        since = datetime.now(timezone.utc) - timedelta(days=days_back)

        # Build filter string with ISSNs (OR logic)
        issn_filter = ",".join(f"issn:{issn}" for issn in issns)
        filter_str = f"from-created-date:{since.date().isoformat()},{issn_filter}"

        url = "https://api.crossref.org/works"
        params = {
            "filter": filter_str,
            "sort": "created",
            "order": "desc",
            "rows": min(200, max_results),
            "mailto": self.config.mailto,
            "select": "DOI,title,author,abstract,container-title,created,URL,type,is-referenced-by-count,publisher,ISSN",
        }

        logger.info(
            "Fetching Crossref works since %s from %d journals (max %d)",
            since.date(),
            len(issns),
            max_results,
        )

        results: List[CandidateWork] = []
        journal_counts: Dict[str, int] = {}
        offset = 0

        while len(results) < max_results:
            params["offset"] = offset
            try:
                resp = self.session.get(url, params=params, timeout=30)
                resp.raise_for_status()
            except Exception as exc:
                logger.warning("Failed to fetch Crossref works: %s", exc)
                break

            message = resp.json().get("message", {})
            items = message.get("items", [])
            if not items:
                break

            for item in items:
                if len(results) >= max_results:
                    break
                work = self._parse_crossref_item(item)
                if work:
                    results.append(work)
                    # Track journal statistics
                    journal = (item.get("container-title") or ["Unknown"])[0]
                    journal_counts[journal] = journal_counts.get(journal, 0) + 1

            # Crossref pagination
            total = message.get("total-results", 0)
            offset += len(items)
            if offset >= total or offset >= max_results:
                break

        logger.info("Fetched %d Crossref works from whitelisted journals", len(results))
        # Log per-journal statistics (top 20)
        if journal_counts:
            sorted_journals = sorted(journal_counts.items(), key=lambda x: -x[1])[:20]
            for journal, count in sorted_journals:
                logger.info("  - %s: %d articles", journal, count)

        return results

    def _get_member_ids(self) -> List[int]:
        """Get Crossref member IDs for configured publishers."""
        member_ids = []
        for name in self.config.publishers:
            if name in MEMBER_IDS:
                member_ids.append(MEMBER_IDS[name])
            else:
                logger.warning("Unknown publisher '%s', skipping", name)
        return member_ids

    def _fetch_with_filters(
        self,
        days_back: int,
        member_ids: List[int],
        max_results: int,
    ) -> List[CandidateWork]:
        """Fetch works from specific publishers with abstract requirement."""
        since = datetime.now(timezone.utc) - timedelta(days=days_back)

        # Build filter string with member IDs (OR logic) and abstract requirement
        member_filter = ",".join(f"member:{m}" for m in member_ids)
        filter_str = f"from-created-date:{since.date().isoformat()},{member_filter},has-abstract:true"

        url = "https://api.crossref.org/works"
        params = {
            "filter": filter_str,
            "sort": "created",
            "order": "desc",
            "rows": min(200, max_results),
            "mailto": self.config.mailto,
            "select": "DOI,title,author,abstract,container-title,created,URL,type,is-referenced-by-count,publisher",
        }

        logger.info(
            "Fetching Crossref works since %s (max %d, has-abstract:true)",
            since.date(),
            max_results,
        )

        results: List[CandidateWork] = []
        publisher_counts: Dict[str, int] = {}
        offset = 0

        while len(results) < max_results:
            params["offset"] = offset
            try:
                resp = self.session.get(url, params=params, timeout=30)
                resp.raise_for_status()
            except Exception as exc:
                logger.warning("Failed to fetch Crossref works: %s", exc)
                break

            message = resp.json().get("message", {})
            items = message.get("items", [])
            if not items:
                break

            for item in items:
                if len(results) >= max_results:
                    break
                work = self._parse_crossref_item(item)
                if work:
                    results.append(work)
                    # Track publisher statistics
                    publisher = item.get("publisher", "Unknown")
                    publisher_counts[publisher] = publisher_counts.get(publisher, 0) + 1

            # Crossref pagination
            total = message.get("total-results", 0)
            offset += len(items)
            if offset >= total or offset >= max_results:
                break

        logger.info("Fetched %d Crossref works with abstracts", len(results))
        # Log per-publisher statistics
        if publisher_counts:
            for pub, count in sorted(publisher_counts.items(), key=lambda x: -x[1]):
                logger.info("  - %s: %d articles", pub, count)

        return results

    def _fetch_general(self, days_back: int) -> List[CandidateWork]:
        """Fetch general Crossref works."""
        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        url = "https://api.crossref.org/works"
        params = {
            "filter": f"from-created-date:{since.date().isoformat()}",
            "sort": "created",
            "order": "desc",
            "rows": 200,
            "mailto": self.config.mailto,
            "select": "DOI,title,author,abstract,container-title,created,URL,type,is-referenced-by-count",
        }

        logger.info("Fetching Crossref works since %s", since.date())
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        message = resp.json().get("message", {})

        results = []
        for item in message.get("items", []):
            work = self._parse_crossref_item(item)
            if work:
                results.append(work)

        logger.info("Fetched %d Crossref works", len(results))
        return results

    def _fetch_top_venues(self, days_back: int) -> List[CandidateWork]:
        """Fetch works from top venues."""
        if not self.top_venues:
            return []

        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        results: List[CandidateWork] = []

        for venue in self.top_venues:
            params = {
                "filter": f"from-created-date:{since.date().isoformat()},container-title:{venue}",
                "sort": "created",
                "order": "desc",
                "rows": 100,
                "mailto": self.config.mailto,
                "select": "DOI,title,author,abstract,container-title,created,URL,type,is-referenced-by-count",
            }
            try:
                resp = self.session.get(
                    "https://api.crossref.org/works",
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
            except Exception as exc:
                logger.warning("Failed to fetch Crossref top venue %s: %s", venue, exc)
                continue

            message = resp.json().get("message", {})
            for item in message.get("items", []):
                work = self._parse_crossref_item(item, venue_override=venue)
                if work:
                    work.extra["source"] = "top_venue"
                    results.append(work)

        if results:
            logger.info("Fetched %d additional works from top venues", len(results))
        return results

    def _parse_crossref_item(
        self,
        item: dict,
        venue_override: Optional[str] = None,
    ) -> Optional[CandidateWork]:
        """Parse Crossref API item to CandidateWork."""
        title = clean_title((item.get("title") or [""])[0])
        if not title:
            return None

        doi = item.get("DOI")
        authors = [" ".join(filter(None, [p.get("given"), p.get("family")])).strip() for p in item.get("author", [])]

        return CandidateWork(
            source="crossref",
            identifier=doi or item.get("URL", "unknown"),
            title=title,
            abstract=clean_html(item.get("abstract")),
            authors=[a for a in authors if a],
            doi=doi,
            url=item.get("URL"),
            published=parse_date(item.get("created", {}).get("date-time")),
            venue=venue_override or (item.get("container-title") or [None])[0],
            metrics={"is-referenced-by": float(item.get("is-referenced-by-count", 0))},
            extra={"type": item.get("type")},
        )


__all__ = ["CrossrefSource"]
