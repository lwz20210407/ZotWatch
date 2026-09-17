from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def json_dumps(data: Any, *, indent: int | None = None) -> str:
    return json.dumps(data, ensure_ascii=False, indent=indent, sort_keys=True)


def hash_content(*parts: str) -> str:
    sha = hashlib.sha256()
    for part in parts:
        if part:
            sha.update(part.encode("utf-8"))
    return sha.hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def beijing_tz() -> timezone | ZoneInfo:
    """Asia/Shanghai, falling back to a fixed +08:00 offset.

    Windows and slim containers have no system tz database, and the `tzdata`
    wheel is not always present. Beijing time has had no DST since 1991, so the
    fixed offset is exact -- a report or an email must not fail over a timestamp.
    """
    try:
        return ZoneInfo("Asia/Shanghai")
    except (ZoneInfoNotFoundError, KeyError):  # pragma: no cover - platform dependent
        return timezone(timedelta(hours=8))


def beijing_now() -> datetime:
    return datetime.now(beijing_tz())


def ensure_isoformat(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def iso_to_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def chunk_dict(d: Dict[str, Any], *, max_len: int = 80) -> Dict[str, Any]:
    """Split long string values to keep JSON manageable (best-effort)."""
    result = {}
    for key, value in d.items():
        if isinstance(value, str) and len(value) > max_len:
            result[key] = value[:max_len] + "…"
        else:
            result[key] = value
    return result


__all__ = [
    "hash_content",
    "json_dumps",
    "utc_now",
    "beijing_tz",
    "beijing_now",
    "ensure_isoformat",
    "iso_to_datetime",
]
