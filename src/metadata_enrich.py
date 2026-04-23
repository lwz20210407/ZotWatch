from __future__ import annotations

import logging
import re
from typing import Iterable
from urllib.parse import quote

import requests

from .models import RankedWork
from .settings import Settings

logger = logging.getLogger(__name__)

DOI_URL_RE = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)


def enrich_ranked_works(works: Iterable[RankedWork], settings: Settings) -> list[RankedWork]:
    session = requests.Session()
    session.headers.update({"User-Agent": "ZotWatcher/0.1 metadata enrichment"})
    enriched = []
    for work in works:
        updates: dict[str, object] = {}
        openalex_data = None
        if work.doi and (not work.abstract or _is_doi_url(work.url)):
            openalex_data = _fetch_openalex_by_doi(session, work.doi, settings)
        if not work.abstract and openalex_data:
            abstract = _extract_openalex_abstract(openalex_data)
            if abstract:
                updates["abstract"] = abstract
        landing_url = _best_landing_url(session, work, openalex_data)
        if landing_url:
            updates["url"] = landing_url
        enriched.append(work.copy(update=updates) if updates else work)
    return enriched


def _best_landing_url(
    session: requests.Session,
    work: RankedWork,
    openalex_data: dict | None,
) -> str | None:
    if work.url and not _is_doi_url(work.url):
        return work.url

    if openalex_data:
        primary_location = openalex_data.get("primary_location") or {}
        landing_page = primary_location.get("landing_page_url")
        if landing_page and not _is_doi_url(landing_page):
            return landing_page

    if work.doi:
        resolved = _resolve_doi(session, work.doi)
        if resolved:
            return resolved

    return work.url


def _fetch_openalex_by_doi(
    session: requests.Session,
    doi: str,
    settings: Settings,
) -> dict | None:
    doi_value = doi.strip()
    if doi_value.lower().startswith("http"):
        doi_url = doi_value
    else:
        doi_url = f"https://doi.org/{doi_value}"
    url = f"https://api.openalex.org/works/{quote(doi_url, safe=':/')}"
    params = {"mailto": settings.sources.openalex.mailto}
    try:
        response = session.get(url, params=params, timeout=20)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.debug("OpenAlex enrichment failed for DOI %s: %s", doi, exc)
        return None


def _resolve_doi(session: requests.Session, doi: str) -> str | None:
    doi_value = doi.strip()
    url = doi_value if doi_value.lower().startswith("http") else f"https://doi.org/{doi_value}"
    try:
        response = session.get(url, allow_redirects=True, timeout=20, stream=True)
        response.close()
    except requests.RequestException as exc:
        logger.debug("DOI resolution failed for %s: %s", doi, exc)
        return None
    if response.url and not _is_doi_url(response.url):
        return response.url
    return None


def _extract_openalex_abstract(item: dict) -> str | None:
    abstract = item.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        return abstract.strip()
    inverted = item.get("abstract_inverted_index")
    if not isinstance(inverted, dict) or not inverted:
        return None
    try:
        size = max(pos for positions in inverted.values() for pos in positions) + 1
    except ValueError:
        return None
    tokens = ["" for _ in range(size)]
    for word, positions in inverted.items():
        for pos in positions:
            if 0 <= pos < size:
                tokens[pos] = word
    text = " ".join(filter(None, tokens)).strip()
    return text or None


def _is_doi_url(url: str | None) -> bool:
    return bool(url and DOI_URL_RE.match(url))


__all__ = ["enrich_ranked_works"]
