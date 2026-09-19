"""Chinese enrichment must degrade one paper at a time, never a whole batch.

On 2026-09-19 run 35416092270 delivered 27 papers of which only 19 carried a Chinese
title, TLDR and abstract. The other 8 were a single batch that timed out three times
and was then dropped whole, so those cards rendered with no translation and no
"翻译" button while their neighbours were complete. The timeout was not bad luck:
the prompt asks for a full abstract translation per paper, so a batch of eight
requests roughly 7.7k output tokens and simply cannot finish inside 180 s. Retrying
the identical request could only hit the identical wall.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from src.enrich_zh import ChineseEnricher
from src.settings import TranslationConfig


def work(i: int):
    return SimpleNamespace(title=f"Paper {i}", abstract=f"Abstract {i}", extra={})


def reply(indices):
    """A well-formed model reply covering exactly the given slots."""
    return [{"i": i, "title_zh": f"标题{i}", "tldr": f"结论{i}", "abstract_zh": f"摘要{i}"}
            for i in indices]


class EnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name, "zh-cache.json")
        self.config = TranslationConfig(batch_size=8, timeout_seconds=1)
        patcher = patch.dict("os.environ", {"EMBEDDING_API_KEY": "test-key"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_a_timing_out_batch_is_split_rather_than_retried(self):
        """Eight papers that fail as one batch still all get translated."""
        works = [work(i) for i in range(8)]
        enricher = ChineseEnricher(self.config, self.cache)
        sizes = []

        def post(batch, attempts=3):
            sizes.append(len(batch))
            if len(batch) > 4:
                raise requests.Timeout("read timed out")
            return reply(range(len(batch)))

        with patch.object(ChineseEnricher, "_post", side_effect=post, autospec=False):
            enricher.enrich(works)

        # 8 fails, then 4 + 4 succeed. No paper is left behind.
        self.assertEqual(sizes, [8, 4, 4])
        for w in works:
            self.assertTrue(w.extra.get("title_zh"), f"{w.title} lost its Chinese title")
            self.assertTrue(w.extra.get("tldr_zh"))
            self.assertTrue(w.extra.get("abstract_zh"))

    def test_an_oversized_batch_is_not_retried_but_a_single_paper_is(self):
        """Repeating an oversized request is pointless; repeating one paper is not."""
        seen = []

        def post(batch, attempts=3):
            seen.append((len(batch), attempts))
            raise requests.Timeout("read timed out")

        with patch.object(ChineseEnricher, "_post", side_effect=post, autospec=False):
            ChineseEnricher(self.config, self.cache).enrich([work(i) for i in range(2)])

        self.assertEqual(seen[0], (2, 1), "a multi-item chunk must not burn retries")
        self.assertTrue(all(attempts == 3 for size, attempts in seen if size == 1),
                        "a single paper should still get the full retry budget")

    def test_one_unrecoverable_paper_does_not_cost_its_neighbours(self):
        works = [work(0), work(1)]

        def post(batch, attempts=3):
            if len(batch) > 1:
                raise requests.Timeout("read timed out")
            if batch[0]["title"] == "Paper 0":
                raise requests.Timeout("read timed out")
            return reply([0])

        with patch.object(ChineseEnricher, "_post", side_effect=post, autospec=False):
            ChineseEnricher(self.config, self.cache).enrich(works)

        self.assertNotIn("title_zh", works[0].extra)
        self.assertEqual(works[1].extra["title_zh"], "标题0")

    def test_a_short_reply_costs_only_the_rows_it_dropped(self):
        """The model echoes an index per row; match on it rather than on position."""
        works = [work(i) for i in range(3)]

        with patch.object(ChineseEnricher, "_post", side_effect=lambda b, attempts=3: reply([0, 2])):
            ChineseEnricher(self.config, self.cache).enrich(works)

        self.assertEqual(works[0].extra["title_zh"], "标题0")
        self.assertNotIn("title_zh", works[1].extra)
        self.assertEqual(works[2].extra["title_zh"], "标题2")

    def test_cache_survives_a_restart_so_a_failed_batch_is_repaired_next_run(self):
        """Without this the weekly run re-translated everything and never recovered."""
        works = [work(0)]
        with patch.object(ChineseEnricher, "_post", side_effect=lambda b, attempts=3: reply([0])):
            ChineseEnricher(self.config, self.cache).enrich(works)
        self.assertTrue(self.cache.exists())
        self.assertEqual(len(json.loads(self.cache.read_text(encoding="utf-8"))), 1)

        # A second enricher must not call the model again for the same paper.
        fresh = [work(0)]
        with patch.object(ChineseEnricher, "_post",
                          side_effect=AssertionError("should have been cached")):
            ChineseEnricher(self.config, self.cache).enrich(fresh)
        self.assertEqual(fresh[0].extra["title_zh"], "标题0")


if __name__ == "__main__":
    unittest.main()
