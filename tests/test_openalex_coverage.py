"""OpenAlex runs every topic query each week once a key is set; Crossref keeps rotating.

Before the key (2026-10-05) both providers ran 40 of 179 queries a run, so a query came
round every ~4.5 weeks against a 30-day window, and the keyless $0.08/day budget stopped
the OpenAlex loop after roughly 40 searches anyway.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.fetch_new import CandidateFetcher
from src.settings import load_settings

REPO = Path(__file__).resolve().parents[1]


class CoverageCapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = load_settings(REPO)
        self.queries = [f"query {i}" for i in range(179)]

    def scheduled(self, provider, **network):
        settings = self.settings.model_copy(deep=True)
        settings.sources.queries = self.queries
        for key, value in network.items():
            setattr(settings.network, key, value)
        with mock.patch("src.fetch_new.load_registry", return_value={"entries": []}):
            fetcher = CandidateFetcher(settings, Path(self.tmp.name))
        return list(fetcher._scheduled_queries(provider))

    def test_openalex_runs_every_query_with_the_configured_cap(self):
        self.assertEqual(len(self.scheduled("openalex", openalex_topic_queries_per_run=200)), 179)

    def test_crossref_keeps_its_own_rotation(self):
        self.assertEqual(len(self.scheduled("crossref", openalex_topic_queries_per_run=200)), 40)

    def test_zero_falls_back_to_the_shared_cap(self):
        self.assertEqual(len(self.scheduled("openalex", openalex_topic_queries_per_run=0)), 40)

    def test_the_shipped_config_covers_all_queries_with_room_for_the_requests(self):
        network = self.settings.network
        self.assertGreaterEqual(network.openalex_topic_queries_per_run, len(self.settings.sources.queries))
        # two pages per query plus semantic, citation, lineage and metadata calls
        self.assertGreaterEqual(network.openalex_max_requests, 2 * len(self.settings.sources.queries) + 100)


if __name__ == "__main__":
    unittest.main()
