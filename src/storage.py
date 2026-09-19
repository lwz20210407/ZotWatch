from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .models import ZoteroItem


SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    key TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    creators TEXT,
    tags TEXT,
    collections TEXT,
    year INTEGER,
    doi TEXT,
    url TEXT,
    raw_json TEXT NOT NULL,
    content_hash TEXT,
    embedding BLOB,
    embedding_signature TEXT,
    embedding_version INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_version ON items(version);
"""


class ProfileStorage:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def initialize(self) -> None:
        conn = self.connect()
        conn.executescript(SCHEMA)
        # Databases created before embedding caching existed lack these columns.
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(items)")}
        for column, ddl in (
            ("embedding_signature", "ALTER TABLE items ADD COLUMN embedding_signature TEXT"),
            ("embedding_version", "ALTER TABLE items ADD COLUMN embedding_version INTEGER"),
        ):
            if column not in existing:
                conn.execute(ddl)
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # metadata helpers
    def get_metadata(self, key: str) -> Optional[str]:
        cur = self.connect().execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["value"] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        self.connect().execute(
            "REPLACE INTO metadata(key, value) VALUES(?, ?)",
            (key, value),
        )
        self.connect().commit()

    def last_modified_version(self) -> Optional[int]:
        value = self.get_metadata("last_modified_version")
        return int(value) if value else None

    def set_last_modified_version(self, version: int) -> None:
        self.set_metadata("last_modified_version", str(version))

    # item helpers
    def upsert_item(self, item: ZoteroItem, content_hash: Optional[str] = None) -> None:
        data = (
            item.key,
            item.version,
            item.title,
            item.abstract,
            json.dumps(item.creators),
            json.dumps(item.tags),
            json.dumps(item.collections),
            item.year,
            item.doi,
            item.url,
            json.dumps(item.raw),
            content_hash,
        )
        self.connect().execute(
            """
            INSERT INTO items(
                key, version, title, abstract, creators, tags, collections, year, doi, url, raw_json, content_hash
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                version=excluded.version,
                title=excluded.title,
                abstract=excluded.abstract,
                creators=excluded.creators,
                tags=excluded.tags,
                collections=excluded.collections,
                year=excluded.year,
                doi=excluded.doi,
                url=excluded.url,
                raw_json=excluded.raw_json,
                content_hash=excluded.content_hash,
                -- A moved content_hash drops the cached vector. Reuse used to be keyed
                -- on items.version alone, on the premise that Zotero bumps it whenever
                -- an item is edited. That is false for anything never synced to
                -- zotero.org: those rows sit at version 0 forever, and 3881 of this
                -- library's 4399 items are in exactly that state. Editing such an
                -- item's title or abstract left a vector computed from the old text
                -- permanently marked valid. Synced items are exposed too, in the
                -- window between a local edit and the next sync round-trip.
                -- `IS` rather than `=` so a NULL hash on both sides compares equal and
                -- keeps the cache (callers may omit content_hash).
                embedding=CASE WHEN items.content_hash IS excluded.content_hash
                          THEN items.embedding END,
                embedding_signature=CASE WHEN items.content_hash IS excluded.content_hash
                          THEN items.embedding_signature END,
                embedding_version=CASE WHEN items.content_hash IS excluded.content_hash
                          THEN items.embedding_version END,
                updated_at=CURRENT_TIMESTAMP
            """,
            data,
        )
        self.connect().commit()

    def remove_items(self, keys: Iterable[str]) -> None:
        keys = list(keys)
        if not keys:
            return
        placeholders = ",".join("?" for _ in keys)
        self.connect().execute(f"DELETE FROM items WHERE key IN ({placeholders})", keys)
        self.connect().commit()

    def set_embedding(self, key: str, vector: bytes, signature: str = "", version: int = 0) -> None:
        self.connect().execute(
            "UPDATE items SET embedding = ?, embedding_signature = ?, embedding_version = ?,"
            " updated_at=CURRENT_TIMESTAMP WHERE key = ?",
            (vector, signature, version, key),
        )
        self.connect().commit()

    def set_embeddings(self, rows: Iterable[Tuple[str, bytes, str, int]]) -> None:
        """Bulk variant; one commit instead of one per item."""
        conn = self.connect()
        conn.executemany(
            "UPDATE items SET embedding = ?, embedding_signature = ?, embedding_version = ?,"
            " updated_at=CURRENT_TIMESTAMP WHERE key = ?",
            [(vector, signature, version, key) for key, vector, signature, version in rows],
        )
        conn.commit()

    def cached_embeddings(self, signature: str) -> dict:
        """Return {key: embedding bytes} still valid for `signature`.

        An embedding is reusable only when it was produced by the same model at the
        same dimension and the item's text has not changed since. The text check is
        enforced in upsert_item, which clears these three columns whenever
        content_hash moves -- NOT by the items.version comparison below, which is
        kept only as a second line of defence. version cannot carry that
        responsibility: Zotero leaves it at 0 for every item never synced to
        zotero.org, which is 3881 of this library's 4399.
        """
        cur = self.connect().execute(
            "SELECT key, embedding FROM items"
            " WHERE embedding IS NOT NULL AND embedding_signature = ?"
            "   AND embedding_version IS NOT NULL AND embedding_version = version",
            (signature,),
        )
        return {row["key"]: row["embedding"] for row in cur}

    def iter_items(self) -> Iterable[ZoteroItem]:
        cur = self.connect().execute("SELECT * FROM items")
        for row in cur:
            yield _row_to_item(row)

    def fetch_items_without_embedding(self) -> List[Tuple[ZoteroItem, Optional[str]]]:
        cur = self.connect().execute(
            "SELECT * FROM items WHERE embedding IS NULL ORDER BY updated_at ASC"
        )
        rows = cur.fetchall()
        return [(_row_to_item(row), row["content_hash"]) for row in rows]

    def fetch_all_embeddings(self) -> List[Tuple[str, bytes]]:
        cur = self.connect().execute(
            "SELECT key, embedding FROM items WHERE embedding IS NOT NULL"
        )
        return [(row["key"], row["embedding"]) for row in cur]


def _row_to_item(row: sqlite3.Row) -> ZoteroItem:
    return ZoteroItem(
        key=row["key"],
        version=row["version"],
        title=row["title"],
        abstract=row["abstract"],
        creators=json.loads(row["creators"] or "[]"),
        tags=json.loads(row["tags"] or "[]"),
        collections=json.loads(row["collections"] or "[]"),
        year=row["year"],
        doi=row["doi"],
        url=row["url"],
        raw=json.loads(row["raw_json"]),
    )


__all__ = ["ProfileStorage"]
