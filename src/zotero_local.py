"""Read a Zotero library straight from its local zotero.sqlite.

The Zotero Web API only returns items that have been synced to zotero.org. On a
library where most items were never uploaded (items.version = 0), the API view is
a small and arbitrary slice, and an interest profile built from it does not
represent the user's actual reading. Reading the local database instead covers
the whole library.

Zotero holds an exclusive lock on the file while running, so the database is
copied before it is opened, and it is only ever opened read-only.
"""
from __future__ import annotations

import logging
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import ZoteroItem

logger = logging.getLogger(__name__)

# Child items carry no topical signal of their own.
EXCLUDED_TYPES = frozenset({"note", "attachment", "annotation"})

# Zotero field names mapped onto ZoteroItem fields.
WANTED_FIELDS = ("title", "abstractNote", "date", "DOI", "url", "publicationTitle")


@dataclass
class LocalIngestStats:
    total: int = 0
    documents: int = 0
    skipped_types: int = 0
    skipped_untitled: int = 0
    synced: int = 0
    local_only: int = 0
    removed: int = 0
    purge_refused: int = 0


def _copy_database(data_dir: Path, workdir: Path) -> Path:
    source = data_dir / "zotero.sqlite"
    if not source.exists():
        raise FileNotFoundError(
            f"zotero.sqlite not found in {data_dir}. Point zotero.local.data_dir at the "
            f"Zotero data directory (the one containing zotero.sqlite and storage/)."
        )
    target = workdir / "zotero.sqlite"
    # Copy rather than open in place: Zotero keeps an exclusive lock while running,
    # and we must never write to the user's library.
    shutil.copy2(source, target)
    for suffix in ("-wal", "-shm"):
        side = source.with_name(source.name + suffix)
        if side.exists():
            shutil.copy2(side, target.with_name(target.name + suffix))
    return target


def _field_values(conn: sqlite3.Connection) -> Dict[int, Dict[str, str]]:
    rows = conn.execute(
        """
        SELECT idata.itemID, f.fieldName, idv.value
        FROM itemData idata
        JOIN fields f ON f.fieldID = idata.fieldID
        JOIN itemDataValues idv ON idv.valueID = idata.valueID
        WHERE f.fieldName IN ({})
        """.format(",".join("?" for _ in WANTED_FIELDS)),
        WANTED_FIELDS,
    )
    out: Dict[int, Dict[str, str]] = defaultdict(dict)
    for item_id, field, value in rows:
        out[item_id][field] = value
    return out


def _creators(conn: sqlite3.Connection) -> Dict[int, List[str]]:
    rows = conn.execute(
        """
        SELECT ic.itemID, c.firstName, c.lastName
        FROM itemCreators ic JOIN creators c ON c.creatorID = ic.creatorID
        ORDER BY ic.itemID, ic.orderIndex
        """
    )
    out: Dict[int, List[str]] = defaultdict(list)
    for item_id, first, last in rows:
        name = " ".join(part for part in (first, last) if part).strip()
        if name:
            out[item_id].append(name)
    return out


def _tags(conn: sqlite3.Connection) -> Dict[int, List[str]]:
    rows = conn.execute(
        "SELECT it.itemID, t.name FROM itemTags it JOIN tags t ON t.tagID = it.tagID"
    )
    out: Dict[int, List[str]] = defaultdict(list)
    for item_id, name in rows:
        out[item_id].append(name)
    return out


def _collections(conn: sqlite3.Connection) -> Dict[int, List[str]]:
    rows = conn.execute(
        """
        SELECT ci.itemID, c.key FROM collectionItems ci
        JOIN collections c ON c.collectionID = ci.collectionID
        """
    )
    out: Dict[int, List[str]] = defaultdict(list)
    for item_id, key in rows:
        out[item_id].append(key)
    return out


def _year(date_value: Optional[str]) -> Optional[int]:
    if not date_value:
        return None
    for part in str(date_value).replace("/", "-").split("-"):
        token = part.strip()[:4]
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


def iter_local_items(data_dir: Path | str) -> Iterable[tuple[ZoteroItem, bool]]:
    """Yield (item, is_synced) for every non-child item in the local library."""
    data_dir = Path(data_dir)
    with tempfile.TemporaryDirectory(prefix="zotwatch-zotero-") as tmp:
        database = _copy_database(data_dir, Path(tmp))
        conn = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            fields = _field_values(conn)
            creators = _creators(conn)
            tags = _tags(conn)
            collections = _collections(conn)
            rows = conn.execute(
                """
                SELECT i.itemID, i.key, i.version, it.typeName
                FROM items i
                JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
                WHERE i.itemID NOT IN (SELECT itemID FROM deletedItems)
                ORDER BY i.itemID
                """
            ).fetchall()
            for item_id, key, version, type_name in rows:
                if type_name in EXCLUDED_TYPES:
                    continue
                data = fields.get(item_id, {})
                title = (data.get("title") or "").strip()
                if not title:
                    continue
                raw = {
                    "key": key,
                    "version": version or 0,
                    "data": {
                        "key": key,
                        "version": version or 0,
                        "itemType": type_name,
                        "title": title,
                        "abstractNote": data.get("abstractNote"),
                        "publicationTitle": data.get("publicationTitle"),
                        "date": data.get("date"),
                        "DOI": data.get("DOI"),
                        "url": data.get("url"),
                    },
                }
                yield (
                    ZoteroItem(
                        key=key,
                        version=version or 0,
                        title=title,
                        abstract=data.get("abstractNote"),
                        creators=creators.get(item_id, []),
                        tags=tags.get(item_id, []),
                        collections=collections.get(item_id, []),
                        year=_year(data.get("date")),
                        doi=data.get("DOI"),
                        url=data.get("url"),
                        raw=raw,
                    ),
                    bool(version),
                )
        finally:
            conn.close()


# Refuse to purge rather than delete more than this share of the store in one pass.
# A wrong data_dir, or a Zotero database mid-sync, reads as "almost everything was
# deleted"; losing that bet costs the embedding cache for the whole library.
MAX_PURGE_FRACTION = 0.2


def ingest_local(storage, data_dir: Path | str) -> LocalIngestStats:
    """Load the whole local library into the profile storage.

    This is a full snapshot, not a delta, so an item absent from the pass was
    trashed, merged away or hard-deleted in Zotero and must go. Without that, the
    store only ever grew: on 2026-09-19 it held 4399 items while the library had
    4391, and those 8 deleted papers were still pulling on the centroid, still
    turning up as nearest neighbours, and still able to suppress a genuinely new
    candidate through dedupe.
    """
    from .utils import hash_content

    storage.initialize()
    stats = LocalIngestStats()
    seen: set = set()
    for item, is_synced in iter_local_items(data_dir):
        storage.upsert_item(
            item,
            content_hash=hash_content(
                item.title, item.abstract or "", ",".join(item.creators), ",".join(item.tags)
            ),
        )
        seen.add(item.key)
        stats.documents += 1
        if is_synced:
            stats.synced += 1
        else:
            stats.local_only += 1

    existing = {row[0] for row in storage.connect().execute("SELECT key FROM items")}
    stale = existing - seen
    if stale and stats.documents:  # an empty read must never empty the store
        if len(stale) > MAX_PURGE_FRACTION * max(len(existing), 1):
            stats.purge_refused = len(stale)
            logger.warning(
                "Refusing to purge %d of %d items (>%.0f%%): this looks like a bad "
                "zotero.sqlite path or a partial read, not %d deletions. Nothing removed.",
                len(stale), len(existing), MAX_PURGE_FRACTION * 100, len(stale),
            )
        else:
            storage.remove_items(stale)
            stats.removed = len(stale)
            logger.info("Purged %d items no longer in the local Zotero library", len(stale))

    logger.info(
        "Local Zotero ingest: %d documents (%d synced to zotero.org, %d local-only, %d purged)",
        stats.documents, stats.synced, stats.local_only, stats.removed,
    )
    return stats


__all__ = ["ingest_local", "iter_local_items", "LocalIngestStats", "EXCLUDED_TYPES"]
