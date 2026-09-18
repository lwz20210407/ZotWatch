"""Reading a Zotero library straight from zotero.sqlite."""

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.zotero_local import iter_local_items

SCHEMA = """
CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, valueID INTEGER);
CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT, itemTypeID INTEGER, version INTEGER);
CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT);
CREATE TABLE itemCreators (itemID INTEGER, creatorID INTEGER, orderIndex INTEGER);
CREATE TABLE tags (tagID INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE itemTags (itemID INTEGER, tagID INTEGER);
CREATE TABLE collections (collectionID INTEGER PRIMARY KEY, key TEXT);
CREATE TABLE collectionItems (collectionID INTEGER, itemID INTEGER);
CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
"""

TYPES = {1: "journalArticle", 2: "attachment", 3: "note", 4: "annotation", 5: "thesis"}
FIELDS = {1: "title", 2: "abstractNote", 3: "date", 4: "DOI", 5: "url", 6: "publicationTitle"}


def build_library(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO itemTypes VALUES(?,?)", TYPES.items())
    conn.executemany("INSERT INTO fields VALUES(?,?)", FIELDS.items())

    def add(item_id, key, type_id, version, values):
        conn.execute("INSERT INTO items VALUES(?,?,?,?)", (item_id, key, type_id, version))
        for field_id, value in values.items():
            conn.execute("INSERT INTO itemDataValues VALUES(?,?)", (item_id * 100 + field_id, value))
            conn.execute("INSERT INTO itemData VALUES(?,?,?)", (item_id, field_id, item_id * 100 + field_id))

    # Synced journal article with full metadata.
    add(1, "AAAA1111", 1, 4242, {1: "Ductile fracture of Ti-6Al-4V", 2: "An abstract.",
                                 3: "2024-05-01", 4: "10.1000/x", 5: "https://e.org/x", 6: "IJIE"})
    # Local-only thesis: version 0, invisible to the Web API.
    add(2, "BBBB2222", 5, 0, {1: "A local-only thesis", 3: "2019"})
    # Child items that carry no topical signal.
    add(3, "CCCC3333", 2, 7, {1: "some.pdf"})
    add(4, "DDDD4444", 3, 7, {1: "a note"})
    add(5, "EEEE5555", 4, 7, {1: "highlight"})
    # Untitled and trashed items.
    add(6, "FFFF6666", 1, 9, {2: "no title here"})
    add(7, "GGGG7777", 1, 9, {1: "Deleted paper"})
    conn.execute("INSERT INTO deletedItems VALUES(7)")

    conn.executemany("INSERT INTO creators VALUES(?,?,?)",
                     [(1, "Jane", "Doe"), (2, "", "Solo")])
    conn.executemany("INSERT INTO itemCreators VALUES(?,?,?)", [(1, 1, 0), (1, 2, 1)])
    conn.execute("INSERT INTO tags VALUES(1,'TC4')")
    conn.execute("INSERT INTO itemTags VALUES(1,1)")
    conn.execute("INSERT INTO collections VALUES(1,'COLLKEY1')")
    conn.execute("INSERT INTO collectionItems VALUES(1,1)")
    conn.commit()
    conn.close()


class ZoteroLocalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        build_library(self.dir / "zotero.sqlite")
        self.items = {item.key: (item, synced) for item, synced in iter_local_items(self.dir)}

    def test_returns_documents_only(self) -> None:
        self.assertEqual(set(self.items), {"AAAA1111", "BBBB2222"})

    def test_local_only_items_are_included(self) -> None:
        """The whole point: the Web API cannot see version 0 items."""
        item, synced = self.items["BBBB2222"]
        self.assertFalse(synced)
        self.assertEqual(item.version, 0)
        self.assertEqual(item.year, 2019)

    def test_synced_flag_and_metadata(self) -> None:
        item, synced = self.items["AAAA1111"]
        self.assertTrue(synced)
        self.assertEqual(item.title, "Ductile fracture of Ti-6Al-4V")
        self.assertEqual(item.abstract, "An abstract.")
        self.assertEqual(item.doi, "10.1000/x")
        self.assertEqual(item.year, 2024)
        self.assertEqual(item.creators, ["Jane Doe", "Solo"])
        self.assertEqual(item.tags, ["TC4"])
        self.assertEqual(item.collections, ["COLLKEY1"])
        self.assertEqual(item.raw["data"]["publicationTitle"], "IJIE")

    def test_trashed_and_untitled_are_dropped(self) -> None:
        self.assertNotIn("GGGG7777", self.items)
        self.assertNotIn("FFFF6666", self.items)

    def test_source_database_is_not_opened_in_place(self) -> None:
        """Zotero holds an exclusive lock; the reader must work on a copy."""
        source = self.dir / "zotero.sqlite"
        before = source.stat().st_mtime_ns
        list(iter_local_items(self.dir))
        self.assertEqual(source.stat().st_mtime_ns, before)

    def test_missing_database_reports_the_directory(self) -> None:
        with TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError) as ctx:
                list(iter_local_items(empty))
            self.assertIn("zotero.sqlite", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
