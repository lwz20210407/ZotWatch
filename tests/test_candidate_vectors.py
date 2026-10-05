import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import requests

from src.candidate_vectors import CandidateVectors
from src.models import CandidateWork
from src.score_rank import rank_with_optional_aminer
from src.vectorizer import EmbeddingError
from src.vectorizer import RemoteVectorizer


class Encoder:
    text_separator = "\n"
    def __init__(self): self.calls = []
    def encode(self, texts):
        self.calls.append(list(texts))
        return np.asarray([[len(t) or 1, 1] for t in texts], dtype=np.float32)


class CandidateVectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name); self.encoder = Encoder()

    def cache(self, identity="model-a", encoder=None, **kwargs):
        return CandidateVectors(encoder or self.encoder, self.path, identity, 2, **kwargs)

    def test_restart_reuses_vectors_and_preserves_duplicates_and_order(self):
        first = self.cache().encode(["bb", "a", "bb"])
        second = self.cache().encode(["a", "bb"])
        self.assertEqual(len(self.encoder.calls), 1)
        np.testing.assert_equal(first[[1, 0]], second)
        self.assertNotIn('"bb"', ''.join(p.read_text() for p in self.path.glob('*.json')))

    def test_model_and_text_changes_do_not_reuse_old_vectors(self):
        self.cache().encode(["a"]); self.cache(identity="model-b").encode(["a"]); self.cache().encode(["changed"])
        self.assertEqual(len(self.encoder.calls), 3)

    def test_corrupt_nan_and_wrong_dimension_cache_are_refreshed(self):
        cache = self.cache(); cache.encode(["a"])
        path = next(self.path.glob('*.json')); data = json.loads(path.read_text()); data['vector'] = [float('nan'), 1]
        path.write_text(json.dumps(data)); cache.encode(["a"])
        self.assertEqual(len(self.encoder.calls), 2)
        with self.assertRaises(EmbeddingError): cache.store('b', [1])

    def test_successful_batches_survive_later_failure(self):
        encoder = Mock(text_separator='\n'); encoder.encode.side_effect = [np.ones((1, 2)), requests.Timeout()]
        cache = self.cache(encoder=encoder, batch_size=1)
        with self.assertRaises(requests.Timeout): cache.encode(['a', 'b'])
        self.assertIsNotNone(cache.cached('a')); self.assertIsNone(cache.cached('b'))
        resumed = self.cache(); resumed.encode(['a', 'b'])
        self.assertEqual(self.encoder.calls, [['b']])

    def test_timeout_split_is_bounded_and_does_not_hide_rate_limit(self):
        encoder = Mock(text_separator='\n')
        encoder.encode.side_effect = [requests.Timeout(), np.ones((1, 2)), np.ones((1, 2))]
        cache = self.cache(encoder=encoder, split_budget=1)
        self.assertEqual(cache.encode(['a', 'b']).shape, (2, 2)); self.assertEqual(cache.stats['splits'], 1)
        encoder.encode.side_effect = requests.HTTPError('rate limit')
        with self.assertRaises(requests.HTTPError): cache.encode(['c', 'd'])

    def test_optional_ranking_failure_preserves_baseline_and_is_not_retried_in_run(self):
        baseline = CandidateWork(source='crossref', identifier='a', title='a')
        optional = CandidateWork(source='aminer', identifier='aminer:b', title='b', extra={'aminer_id':'b'})
        ranker = Mock(); ranked = SimpleNamespace(score=0.6)
        ranker.rank.side_effect = [[ranked], requests.Timeout(), [ranked]]
        deferred = set()
        self.assertEqual(rank_with_optional_aminer(ranker, [baseline, optional], True, deferred), [ranked])
        self.assertEqual(deferred, {'aminer:b'})
        self.assertEqual(rank_with_optional_aminer(ranker, [baseline, optional], True, deferred), [ranked])
        self.assertEqual(ranker.rank.call_count, 3)

    def test_required_source_failure_is_not_silently_ignored(self):
        ranker = Mock(); ranker.rank.side_effect = requests.Timeout()
        with self.assertRaises(requests.Timeout): rank_with_optional_aminer(ranker, [], True, set())

    def test_malformed_provider_indices_are_not_silently_reordered(self):
        vectorizer = RemoteVectorizer('m', 'https://example.test', 'secret', dimensions=2)
        response = Mock(); response.json.return_value = {'data': [{'index': 1, 'embedding': [1, 2]}]}
        with patch('src.vectorizer.request_with_retry', return_value=response):
            with self.assertRaises(EmbeddingError): vectorizer.encode(['a'])


class ConcurrentBatchTests(unittest.TestCase):
    """~90 serial batches cost ~6 minutes of the 2026-10-05 run; they now overlap."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def test_batches_overlap_and_order_is_preserved(self):
        import time

        class Slow(Encoder):
            def encode(self, texts):
                time.sleep(0.2)
                return super().encode(texts)
        texts = [f"text {i}" * (i + 1) for i in range(32)]
        vectors = CandidateVectors(Slow(), self.path, "m", 2, batch_size=8, workers=4)
        started = time.monotonic()
        result = vectors.encode(texts)
        self.assertLess(time.monotonic() - started, 0.6, "4 batches of 0.2 s should overlap")
        np.testing.assert_equal(result[:, 0], [len(t) for t in texts])

    def test_a_failing_batch_still_raises(self):
        class Broken(Encoder):
            def encode(self, texts):
                raise EmbeddingError("bad")
        with self.assertRaises(EmbeddingError):
            CandidateVectors(Broken(), self.path, "m", 2, batch_size=2, workers=4).encode(["a", "b", "c", "d"])

    def test_the_split_budget_is_shared_across_workers(self):
        class TimesOut(Encoder):
            def encode(self, texts):
                if len(texts) > 1:
                    raise requests.Timeout()
                return super().encode(texts)
        vectors = CandidateVectors(TimesOut(), self.path, "m", 2, batch_size=2, split_budget=1, workers=4)
        with self.assertRaises(requests.Timeout):
            vectors.encode(["a", "bb", "ccc", "dddd"])
        self.assertEqual(vectors.stats["splits"], 1, "one split in total, not one per worker")

    def test_remote_encoders_get_a_session_per_thread(self):
        import threading
        remote = RemoteVectorizer("m", "https://example.invalid", "k", dimensions=2)
        vectors = CandidateVectors(remote, self.path, "m", 2)
        seen = []
        worker = threading.Thread(target=lambda: seen.append(vectors._thread_encoder()._session))
        worker.start(); worker.join()
        mine = vectors._thread_encoder()._session
        self.assertIsNot(seen[0], mine)
        self.assertEqual(mine.headers.get("Authorization"), "Bearer k")


if __name__ == '__main__': unittest.main()
