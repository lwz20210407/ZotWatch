"""The profile store must describe the library as it is now, with the encoder that ran.

Three defects found by external review on 2026-09-19, all of which let a run succeed,
send mail and pass the whole suite while working from wrong data:

  B1  an item edited without its Zotero version moving kept its old vector forever.
      3881 of this library's 4399 items sit at version 0 permanently.
  B2  an item deleted in Zotero was never removed from the store. Measured drift:
      the library held 4391 documents, the profile 4399.
  B3  with EMBEDDING_API_KEY unset the encoder silently becomes a 384-dim local
      model, but the bundle was stamped with the remote 1024-dim signature -- so it
      passed the workflow's compatibility gate and then queried the wrong index.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.models import ZoteroItem
from src.settings import EmbeddingConfig
from src.storage import ProfileStorage
from src.vectorizer import build_vectorizer
from src.zotero_local import MAX_PURGE_FRACTION, ingest_local


def item(key: str, title: str, version: int = 0) -> ZoteroItem:
    return ZoteroItem(key=key, version=version, title=title, abstract="abs",
                      creators=[], tags=[], collections=[], item_type="journalArticle")


class EmbeddingCacheTests(unittest.TestCase):
    """The cache must key on the text, not on a version Zotero often leaves alone."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProfileStorage(Path(self.tmp.name, "p.sqlite"))
        self.store.initialize()
        # Close before cleanup: Windows will not unlink a file with an open handle.
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.store.close)

    def seed(self, title: str, content_hash: str, version: int = 0) -> None:
        self.store.upsert_item(item("K1", title, version), content_hash=content_hash)

    def test_editing_an_unsynced_item_invalidates_its_vector(self):
        self.seed("Old title", "hash-old")
        self.store.set_embeddings([("K1", b"\x00" * 8, "sig", 0)])
        self.assertIn("K1", self.store.cached_embeddings("sig"))

        self.seed("New title", "hash-new")  # same version 0, different text
        self.assertNotIn("K1", self.store.cached_embeddings("sig"),
                         "an edited item must be re-encoded, not served from cache")

    def test_reingesting_unchanged_content_keeps_the_vector(self):
        """The whole point of the cache: no needless re-encoding, no needless bill."""
        self.seed("Same title", "hash-same")
        self.store.set_embeddings([("K1", b"\x11" * 8, "sig", 0)])
        self.seed("Same title", "hash-same")
        self.assertIn("K1", self.store.cached_embeddings("sig"))

    def test_a_synced_item_edited_before_its_next_sync_is_also_invalidated(self):
        """Zotero does not bump version in local SQLite until the sync round-trip."""
        self.seed("Old title", "hash-old", version=10)
        self.store.set_embeddings([("K1", b"\x22" * 8, "sig", 10)])
        self.seed("New title", "hash-new", version=10)
        self.assertNotIn("K1", self.store.cached_embeddings("sig"))

    def test_a_caller_omitting_content_hash_does_not_lose_the_cache(self):
        """NULL on both sides must compare equal -- hence `IS`, not `=`."""
        self.store.upsert_item(item("K1", "T"))
        self.store.set_embeddings([("K1", b"\x33" * 8, "sig", 0)])
        self.store.upsert_item(item("K1", "T"))
        self.assertIn("K1", self.store.cached_embeddings("sig"))


class SnapshotPurgeTests(unittest.TestCase):
    """A full snapshot is authoritative: absent means deleted."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProfileStorage(Path(self.tmp.name, "p.sqlite"))
        self.store.initialize()
        # Close before cleanup: Windows will not unlink a file with an open handle.
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.store.close)

    def keys(self):
        return {r[0] for r in self.store.connect().execute("SELECT key FROM items")}

    def ingest(self, pairs):
        with patch("src.zotero_local.iter_local_items", return_value=iter(pairs)):
            return ingest_local(self.store, Path(self.tmp.name))

    def test_an_item_gone_from_the_library_is_removed_from_the_profile(self):
        library = [(item(k, f"Paper {k}"), True) for k in "ABCDEFGHIJ"]
        self.ingest(library)
        self.assertEqual(len(self.keys()), 10)

        stats = self.ingest(library[:-1])  # "J" deleted in Zotero
        self.assertNotIn("J", self.keys(), "deleted papers still steer the centroid")
        self.assertEqual(stats.removed, 1)

    def test_an_empty_read_never_empties_the_store(self):
        self.ingest([(item(k, k), True) for k in "ABCDE"])
        stats = self.ingest([])
        self.assertEqual(len(self.keys()), 5, "a failed read must not wipe the library")
        self.assertEqual(stats.removed, 0)

    def test_an_implausibly_large_purge_is_refused_and_reported(self):
        """A wrong data_dir reads as 'almost everything was deleted'."""
        self.ingest([(item(k, k), True) for k in "ABCDEFGHIJ"])
        stats = self.ingest([(item("A", "A"), True)])  # 9 of 10 would vanish
        self.assertEqual(len(self.keys()), 10)
        self.assertEqual(stats.removed, 0)
        self.assertEqual(stats.purge_refused, 9)
        self.assertGreater(9, MAX_PURGE_FRACTION * 10)

    def test_a_purge_within_the_guard_still_happens(self):
        self.ingest([(item(k, k), True) for k in "ABCDEFGHIJ"])
        stats = self.ingest([(item(k, k), True) for k in "ABCDEFGHI"])  # 1 of 10
        self.assertEqual(stats.removed, 1)
        self.assertNotIn("J", self.keys())


class EncoderSignatureTests(unittest.TestCase):
    """A bundle may not claim an encoder it was not built with."""

    def test_local_fallback_does_not_borrow_the_remote_signature(self):
        config = EmbeddingConfig(provider="openai-compatible",
                                 model_name="Qwen/Qwen3-Embedding-8B", dimensions=1024,
                                 api_key_env="ZOTWATCH_TEST_ABSENT_KEY",
                                 local_fallback_model="sentence-transformers/all-MiniLM-L6-v2")
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("ZOTWATCH_TEST_ABSENT_KEY", None)
            vec = build_vectorizer(config)
        self.assertTrue(vec.signature.startswith("local:"), vec.signature)
        self.assertNotEqual(vec.signature, config.cache_signature(),
                            "the fallback must not pass the workflow's compatibility gate")

    def test_remote_keeps_the_config_signature_so_existing_caches_stay_valid(self):
        config = EmbeddingConfig(provider="openai-compatible",
                                 model_name="Qwen/Qwen3-Embedding-8B", dimensions=1024,
                                 api_key_env="ZOTWATCH_TEST_PRESENT_KEY")
        with patch.dict("os.environ", {"ZOTWATCH_TEST_PRESENT_KEY": "k"}):
            vec = build_vectorizer(config)
        self.assertEqual(vec.signature, config.cache_signature())

    def test_the_profile_records_the_signature_of_the_encoder_that_ran(self):
        from src.build_profile import ProfileBuilder
        from src.settings import load_settings

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        settings = load_settings(Path(__file__).resolve().parents[1])
        settings.embedding = EmbeddingConfig(provider="openai-compatible",
                                             model_name="Qwen/Qwen3-Embedding-8B",
                                             dimensions=1024)
        stub = SimpleNamespace(model_name="all-MiniLM-L6-v2", text_separator="\n",
                               signature="local:all-MiniLM-L6-v2",
                               encode=lambda texts: np.ones((len(texts), 384), dtype=np.float32))
        builder = ProfileBuilder(Path(tmp.name), ProfileStorage(Path(tmp.name, "p.sqlite")),
                                 settings, vectorizer=stub)
        self.assertEqual(builder.embedding_signature, "local:all-MiniLM-L6-v2")
        self.assertNotEqual(builder.embedding_signature, settings.embedding.cache_signature())


if __name__ == "__main__":
    unittest.main()
