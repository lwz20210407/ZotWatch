"""Identifier-based author discovery, provenance labels and report selection."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests

from .models import CandidateWork, RankedWork
from .settings import AuthorWatchConfig, Settings
from .source_paging import iter_works

logger = logging.getLogger(__name__)


def work_key(work: CandidateWork) -> str:
    if work.doi:
        return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", work.doi.strip(), flags=re.I).casefold().rstrip(" .;")
    return work.identifier.casefold().strip()


def authorship_identifiers(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"author_id": (a.get("author", {}).get("id") or "").rsplit("/", 1)[-1],
         "name": a.get("author", {}).get("display_name") or "",
         "institution_ids": [i["id"].rsplit("/", 1)[-1] for i in a.get("institutions", []) if i.get("id")]}
        for a in item.get("authorships", [])
    ]


def mark_watched_authors(work: CandidateWork, config: AuthorWatchConfig) -> CandidateWork:
    # Recompute on every run so disabling an author also clears old cache labels.
    extra = {k: v for k, v in work.extra.items() if k != "watched_authors"}
    matches = []
    if config.enabled:
        identities = extra.get("openalex_authorships", [])
        for tracked in config.authors:
            if not tracked.enabled:
                continue
            for identity in identities:
                if identity.get("author_id") not in tracked.openalex_ids:
                    continue
                if tracked.institution_ids and not set(tracked.institution_ids).intersection(identity.get("institution_ids", [])):
                    continue
                matches.append({"name": tracked.name, "author_id": identity["author_id"], "reason": tracked.reason})
                break
    if matches:
        extra["watched_authors"] = matches
    return work.model_copy(update={"extra": extra})


def _abstract(item: Dict[str, Any]) -> str | None:
    inverted = item.get("abstract_inverted_index") or {}
    words = [(position, word) for word, positions in inverted.items() for position in positions]
    return " ".join(word for _, word in sorted(words)) or None


def candidate_from_openalex(item: Dict[str, Any]) -> CandidateWork | None:
    title = item.get("display_name") or ""
    if not title or item.get("is_retracted") or item.get("type") in {"dataset", "supplementary-material", "peer-review"}:
        return None
    location = item.get("primary_location") or {}
    source = location.get("source") or {}
    try:
        published = datetime.fromisoformat(item["publication_date"]).replace(tzinfo=timezone.utc)
    except (KeyError, ValueError, TypeError):
        published = None
    return CandidateWork(
        source="openalex", identifier=item.get("id") or item.get("doi") or title,
        title=title, abstract=_abstract(item),
        authors=[a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])],
        doi=item.get("doi"), url=location.get("landing_page_url") or item.get("doi") or item.get("id"),
        published=published, venue=source.get("display_name"),
        metrics={"cited_by": float(item.get("cited_by_count") or 0)},
        extra={"openalex_authorships": authorship_identifiers(item), "discovery_route": "author_watch",
               "referenced_works": item.get("referenced_works") or [], "work_type": item.get("type")},
    )


def fetch_author_works(session: requests.Session, settings: Settings, since: datetime) -> List[CandidateWork]:
    config = settings.author_watch
    ids = sorted({aid for a in config.authors if a.enabled for aid in a.openalex_ids}) if config.enabled else []
    results = []
    # Batch stable IDs, not names; other coauthors cannot create a name-match false positive.
    for start in range(0, len(ids), 10):
        batch = ids[start:start + 10]
        params = {
            "filter": f"authorships.author.id:{'|'.join(batch)},from_publication_date:{since.date().isoformat()}",
            "sort": "publication_date:desc", "per-page": settings.sources.page_size,
            "mailto": settings.sources.openalex.mailto,
        }
        for item in iter_works(session, "https://api.openalex.org/works", params, provider="openalex",
                               max_pages=config.max_pages, interval_seconds=settings.sources.request_interval_seconds,
                               logger=logger, context=f"Author watch batch {start // 10 + 1}"):
            work = candidate_from_openalex(item)
            if work is None:
                continue
            work = mark_watched_authors(work, config)
            if work.extra.get("watched_authors"):
                results.append(work)
    logger.info("Author watch fetched %d works from %d configured identifiers", len(results), len(ids))
    return results


def author_news(works: List[RankedWork], config: AuthorWatchConfig) -> List[RankedWork]:
    if not config.enabled or config.max_report_items == 0:
        return []
    candidates = [w for w in works if w.extra.get("watched_authors") and not w.extra.get("semantic_gate_failed")]
    candidates.sort(key=lambda w: (w.published.timestamp() if w.published else 0, w.score), reverse=True)
    seen, result = set(), []
    for work in candidates:
        key = work_key(work)
        if key not in seen:
            seen.add(key)
            result.append(work)
    return result[:config.max_report_items]


def merge_report_works(recommended: List[RankedWork], watched: List[RankedWork]) -> List[RankedWork]:
    seen, result = set(), []
    for work in [*recommended, *watched]:
        key = work_key(work)
        if key not in seen:
            seen.add(key)
            result.append(work)
    return result
