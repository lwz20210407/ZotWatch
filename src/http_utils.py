from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable

import requests

RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def _retry_delay(header: str | None, fallback: float) -> float:
    if not header:
        return fallback
    try:
        return max(fallback, float(header))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(header)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(fallback, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return fallback


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    logger: logging.Logger,
    context: str,
    attempts: int = 3,
    backoff_seconds: float = 1.5,
    retryable_status_codes: Iterable[int] = RETRYABLE_STATUS_CODES,
    **kwargs,
) -> requests.Response:
    retryable_codes = set(retryable_status_codes)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = session.request(method, url, **kwargs)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt == attempts:
                raise
            logger.warning(
                "%s failed on attempt %d/%d with %s; retrying in %.1fs",
                context,
                attempt,
                attempts,
                exc.__class__.__name__,
                backoff_seconds * attempt,
            )
            time.sleep(backoff_seconds * attempt)
            continue

        if response.status_code in retryable_codes:
            if attempt == attempts:
                response.raise_for_status()
            delay = _retry_delay(response.headers.get("Retry-After"), backoff_seconds * attempt)
            logger.warning(
                "%s returned HTTP %s on attempt %d/%d; retrying in %.1fs",
                context,
                response.status_code,
                attempt,
                attempts,
                delay,
            )
            time.sleep(delay)
            continue

        response.raise_for_status()
        return response

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{context} failed without a response")


__all__ = ["RETRYABLE_STATUS_CODES", "request_with_retry"]
