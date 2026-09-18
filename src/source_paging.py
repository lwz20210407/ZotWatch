"""Bounded metadata pagination with explicit coverage warnings."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Literal, Optional

import requests

from .http_utils import request_with_retry


def iter_works(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    *,
    provider: Literal["openalex", "crossref"],
    max_pages: int,
    interval_seconds: float,
    logger: logging.Logger,
    context: str,
) -> Iterator[Dict[str, Any]]:
    request_params = dict(params)
    request_params["cursor"] = "*"
    size = int(params.get("rows", params.get("per-page", 100)))
    seen_cursors = {"*"}
    for page in range(1, max_pages + 1):
        if interval_seconds:
            time.sleep(interval_seconds)
        try:
            response = request_with_retry(
                session, "GET", url, params=dict(request_params), timeout=30,
                logger=logger, context=f"{context} page {page}",
            )
        except requests.RequestException as exc:
            logger.warning("Incomplete coverage for %s after %d pages: %s", context, page - 1, exc)
            return
        data = response.json()
        if provider == "crossref":
            message = data.get("message", {})
            items = message.get("items", [])
            cursor = message.get("next-cursor")
            total = message.get("total-results", 0)
        else:
            items = data.get("results", [])
            cursor = data.get("meta", {}).get("next_cursor")
            total = data.get("meta", {}).get("count", 0)
        yield from items
        if len(items) < size or (total and page * size >= total):
            return
        if page == max_pages:
            logger.warning("Coverage cap reached for %s: %d pages of %d; reported total=%s", context, page, size, total)
            return
        if not cursor or cursor in seen_cursors:
            logger.warning("Incomplete pagination for %s: missing/repeated cursor", context)
            return
        seen_cursors.add(cursor)
        request_params["cursor"] = cursor


def crossref_publication_date_precise(item: Dict[str, Any]) -> tuple:
    """Return (date, precision) where precision is "day", "month" or "year".

    Crossref date-parts may carry only a year, or a year and month. The missing
    components are filled with 1 so a datetime can be built, which means the
    stored value looks like 2026-01-01 for a record that really only says "2026".
    Printing that as a full date invents precision the source never had, so the
    precision travels with the date and the report formats accordingly.
    """
    dates = []
    for field in ("published", "published-online", "published-print", "issued"):
        parts = item.get(field, {}).get("date-parts", [])
        if not parts or not parts[0]:
            continue
        values = [v for v in parts[0] if v is not None]
        if not values:
            continue
        try:
            built = datetime(values[0], values[1] if len(values) > 1 else 1,
                             values[2] if len(values) > 2 else 1, tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
        dates.append((built, {1: "year", 2: "month"}.get(len(values), "day")))
    if not dates:
        return None, ""
    return min(dates, key=lambda row: row[0])


def crossref_publication_date(item: Dict[str, Any]) -> Optional[datetime]:
    """Use publication metadata, never the Crossref record creation timestamp."""
    return crossref_publication_date_precise(item)[0]
