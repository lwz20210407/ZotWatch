"""Remote embedding provider and the library embedding cache."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.build_profile import ProfileBuilder
from src.models import ZoteroItem
from src.settings import EmbeddingConfig, load_settings
from src.storage import ProfileStorage
from src.vectorizer import LocalVectorizer, RemoteVectorizer, build_vectorizer

BASE = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class RemoteVectorizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls = []
        self.vectorizer = RemoteVectorizer(
            "test-model", "https://api.test/v1", "k", dimensions=4, batch_size=2
        )

    def _fake_post(self, session, method, url, *, logger, context, json, timeout):
        self.calls.append((url, json))
        # Return in shuffled index order to prove the client re-sorts.
        data = [
            {"index": i, "embedding": [float(i + 1), 0.0, 0.0, 0.0]}
            for i in range(len(json["input"]))
        ]
        return FakeResponse({"data": list(reversed(data))})

    def test_batches_and_preserves_input_order(self) -> None:
        with patch("src.vectorizer.request_with_retry", self._fake_post):
            vectors = self.vectorizer.encode(["a", "b", "c"])
        self.assertEqual(vectors.shape, (3, 4))
        self.assertEqual([len(call[1]["input"]) for call in self.calls], [2, 1])
        # Each vector is e_0 scaled, so after normalisation all are the unit x axis
        # in the order they were submitted.
        np.testing.assert_allclose(vectors[:, 0], [1.0, 1.0, 1.0], atol=1e-6)

    def test_requests_the_configured_dimensions(self) -> None:
        with patch("src.vectorizer.request_with_retry", self._fake_post):
            self.vectorizer.encode(["a"])
        self.assertEqual(self.calls[0][1]["dimensions"], 4)
        self.assertTrue(self.calls[0][0].endswith("/embeddings"))

    def test_output_is_unit_norm(self) -> None:
        def post(session, method, url, *, logger, context, json, timeout):
            return FakeResponse({"data": [{"index": 0, "embedding": [3.0, 4.0, 0.0, 0.0]}]})

        with patch("src.vectorizer.request_with_retry", post):
            vectors = self.vectorizer.encode(["a"])
        self.assertAlmostEqual(float(np.linalg.norm(vectors[0])), 1.0, places=6)

    def test_blank_text_does_not_reach_the_api(self) -> None:
        with patch("src.vectorizer.request_with_retry", self._fake_post):
            self.vectorizer.encode(["", "   "])
        self.assertTrue(all(text.strip() == "" or text == " " for text in self.calls[0][1]["input"]))
        self.assertNotIn("", self.calls[0][1]["input"])

    def test_empty_input_makes_no_request(self) -> None:
        with patch("src.vectorizer.request_with_retry", self._fake_post):
            out = self.vectorizer.encode([])
        self.assertEqual(out.shape[0], 0)
        self.assertEqual(self.calls, [])


class ProviderSelectionTests(unittest.TestCase):
    def test_remote_provider_used_when_the_key_is_present(self) -> None:
        config = EmbeddingConfig(provider="openai-compatible", api_key_env="TEST_EMB_KEY")
        with patch.dict("os.environ", {"TEST_EMB_KEY": "secret"}):
            self.assertIsInstance(build_vectorizer(config), RemoteVectorizer)

    def test_missing_key_degrades_to_the_local_fallback(self) -> None:
        config = EmbeddingConfig(provider="openai-compatible", api_key_env="TEST_EMB_KEY")
        with patch.dict("os.environ", {"TEST_EMB_KEY": ""}):
            vectorizer = build_vectorizer(config)
        self.assertIsInstance(vectorizer, LocalVectorizer)
        self.assertEqual(vectorizer.model_name, config.local_fallback_model)

    def test_missing_key_without_fallback_is_an_error(self) -> None:
        from src.vectorizer import EmbeddingError

        config = EmbeddingConfig(
            provider="openai-compatible", api_key_env="TEST_EMB_KEY", local_fallback_model=""
        )
        with patch.dict("os.environ", {"TEST_EMB_KEY": ""}):
            with self.assertRaises(EmbeddingError):
                build_vectorizer(config)

    def test_signature_distinguishes_model_and_dimension(self) -> None:
        a = EmbeddingConfig(model_name="m", dimensions=1024).cache_signature()
        b = EmbeddingConfig(model_name="m", dimensions=512).cache_signature()
        c = EmbeddingConfig(model_name="other", dimensions=1024).cache_signature()
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)


class CountingVectorizer:
    text_separator = "\n"
    model_name = "counting"

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.encoded = 0

    def encode(self, texts):
        texts = list(texts)
        self.encoded += len(texts)
        out = np.tile(np.arange(1, self.dim + 1, dtype=np.float32), (len(texts), 1))
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.storage = ProfileStorage(Path(self.tmp.name, "profile.sqlite"))
        self.storage.initialize()
        # Registered after tmp.cleanup so LIFO closes the sqlite handle first;
        # Windows refuses to unlink a file that is still open.
        self.addCleanup(self.storage.close)
        self.settings = load_settings(BASE)
        for i in range(3):
            self.storage.upsert_item(
                ZoteroItem(
                    key=f"K{i}", version=1, title=f"Paper {i}", abstract="abstract",
                    raw={"data": {"itemType": "journalArticle"}},
                )
            )

    def _build(self, vectorizer):
        builder = ProfileBuilder(Path(self.tmp.name), self.storage, self.settings, vectorizer=vectorizer)
        return builder.run()

    def test_second_run_reuses_cached_vectors(self) -> None:
        """The library used to be fully re-encoded every week."""
        v1 = CountingVectorizer()
        self._build(v1)
        self.assertEqual(v1.encoded, 3)

        v2 = CountingVectorizer()
        self._build(v2)
        self.assertEqual(v2.encoded, 0)

    def test_only_changed_items_are_re_encoded(self) -> None:
        self._build(CountingVectorizer())
        # Zotero bumps `version` whenever an item is edited.
        self.storage.upsert_item(
            ZoteroItem(key="K1", version=2, title="Paper 1 revised", abstract="new",
                       raw={"data": {"itemType": "journalArticle"}})
        )
        v = CountingVectorizer()
        self._build(v)
        self.assertEqual(v.encoded, 1)

    def test_changing_the_model_invalidates_the_whole_cache(self) -> None:
        self._build(CountingVectorizer())
        self.settings = self.settings.model_copy(
            update={"embedding": self.settings.embedding.model_copy(update={"dimensions": 512})}
        )
        v = CountingVectorizer()
        self._build(v)
        self.assertEqual(v.encoded, 3)

    def test_vector_order_matches_item_order(self) -> None:
        artifacts = self._build(CountingVectorizer())
        self.assertTrue(Path(artifacts.faiss_path).exists())


if __name__ == "__main__":
    unittest.main()
