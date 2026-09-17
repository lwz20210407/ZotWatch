import json
import tempfile
import unittest
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import requests

from src.citation_watch import CitationDiscovery, citation_strength, merge_candidates, recommendation_reasons
from src.fetch_new import CandidateFetcher
from src.models import CandidateWork, RankedWork
from src.report_html import render_html
from src.rss_writer import write_rss
from src.settings import CitationSeed, load_settings
from src.watch_history import WatchHistory


def raw(ident, refs=(), date=None, **extra):
    return {"id": f"https://openalex.org/{ident}", "display_name": "Steel plasticity " + ident,
            "doi": "https://doi.org/10.1234/" + ident, "type": "article",
            "publication_date": date or datetime.now(timezone.utc).date().isoformat(),
            "referenced_works": ["https://openalex.org/" + r for r in refs], **extra}


def ranked(title="Steel plasticity", **kwargs):
    return RankedWork(source="test", identifier="W2", title=title, score=0.7, similarity=0.6,
                      recency_score=0, metric_score=0, author_bonus=0, venue_bonus=0, label="consider", **kwargs)


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(Path(__file__).resolve().parents[1])
        self.settings.sources.request_interval_seconds = 0
        self.settings.citation_watch.seeds = [CitationSeed(openalex_id="W1", title="Seed")]
        self.discovery = CitationDiscovery(self.settings, Mock())

    def test_forward_backward_direction_and_dates(self):
        seed = raw("W1", ["W9"], "2020-01-01")
        def resolve(ids, field="openalex"):
            return [seed] if ids == ["W1"] else [raw("W9", date="1980-01-01")]
        self.discovery.resolve = Mock(side_effect=resolve)
        self.discovery.query = Mock(return_value=iter([
            raw("W2", ["W1"]), raw("W3", ["W8"]), raw("W4", ["W1"], "2000-01-01"),
            raw("W5", ["W1"], "2999-01-01"), raw("W6", ["W1"], is_retracted=True)]))
        works = self.discovery.fetch()
        self.assertEqual({w.identifier.rsplit('/', 1)[-1] for w in works}, {"W2", "W9"})
        self.assertTrue(works[0].extra["cites_seeds"])
        self.assertTrue(works[1].extra["referenced_by"])
        self.assertIn("cites:W1", self.discovery.query.call_args.args[0])

    def test_missing_or_retracted_seed_warns(self):
        self.discovery.resolve = Mock(return_value=[raw("W1", is_retracted=True)])
        self.assertEqual(self.discovery.fetch(), [])
        self.assertTrue(self.discovery.warnings)

    def test_disabled_no_requests(self):
        self.settings.citation_watch.enabled = False
        self.assertEqual(self.discovery.fetch(), [])
        self.discovery.session.request.assert_not_called()

    def test_requests_capped_and_api_failure_warns(self):
        self.settings.citation_watch.max_requests = 1
        response = Mock(); response.json.return_value = {"results": []}
        with patch("src.citation_watch.request_with_retry", return_value=response) as req:
            list(self.discovery.query("openalex:W1"))
            list(self.discovery.query("openalex:W2"))
            self.assertEqual(req.call_count, 1)
        self.assertTrue(self.discovery.warnings)
        self.discovery.requests = 0
        with patch("src.citation_watch.request_with_retry", side_effect=requests.Timeout):
            self.assertEqual(list(self.discovery.query("openalex:W1")), [])

    def test_shared_references_and_bounded_bonus(self):
        self.discovery.seed_records = {"W1": raw("W1", ["W8", "W9"])}
        work = CandidateWork(source="test", identifier="w", title="Steel", extra={"referenced_works": ["W8", "W9"]})
        out = self.discovery.annotate([work])[0]
        self.assertEqual(out.extra["bibliographic_coupling"]["count"], 2)
        self.assertEqual(out.extra["bibliographic_coupling"]["score"], 1)
        self.assertLessEqual(citation_strength({**out.extra, "cites_seeds": [1]}), 1)
        self.assertNotIn("bibliographic_coupling", work.extra)

    def test_merge_retains_both_routes_and_does_not_mutate(self):
        a = CandidateWork(source="test", identifier="a", doi="10.1234/a", title="A", extra={"cites_seeds": [{"id": "W1"}]})
        b = a.model_copy(update={"identifier": "b", "doi": "https://doi.org/10.1234/A",
                                 "extra": {"referenced_by": [{"id": "W3"}]}})
        merged = merge_candidates([a, b])
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0].extra["cites_seeds"])
        self.assertTrue(merged[0].extra["referenced_by"])
        self.assertNotIn("referenced_by", a.extra)

    def test_graph_still_requires_mechanical_topic(self):
        fetcher = object.__new__(CandidateFetcher); fetcher.settings = self.settings
        good = CandidateWork(source="test", identifier="good", title="Polycrystalline plasticity mechanisms",
                             extra={"cites_seeds": [{"id": "W1"}]})
        bad = good.model_copy(update={"identifier": "bad", "title": "Commodity price prediction"})
        self.assertEqual(len(fetcher._filter_by_topic([good, bad])), 1)

    def test_history_staged_not_sent_until_commit_and_persistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = ranked(doi="10.1234/a")
            history = WatchHistory(tmp)
            history.stage([work])
            self.assertEqual(len(WatchHistory(tmp).filter([work])), 1)
            history.commit()
            self.assertEqual(WatchHistory(tmp).filter([work]), [])
            variant = work.model_copy(update={"doi": "https://doi.org/10.1234/A", "title": "Different title"})
            self.assertEqual(WatchHistory(tmp).filter([variant]), [])
            no_doi = work.model_copy(update={"doi": None, "identifier": "different"})
            self.assertEqual(WatchHistory(tmp).filter([no_doi]), [])

    def test_corrupt_history_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "state.json").write_text('{"version":999}', encoding="utf-8")
            with self.assertRaises(ValueError): WatchHistory(tmp)

    def test_classic_report_and_rss_evidence_are_escaped(self):
        work = ranked(title="Steel <script>bad</script>", extra={"report_channel": "经典文献补漏",
                      "nearest_library_work": {"title": "Library <b>paper</b>"},
                      "referenced_by": [{"id": "W1", "title": "Seed", "url": "https://openalex.org/W1"}]})
        with tempfile.TemporaryDirectory() as tmp:
            render_html([], Path(tmp, "report.html"), classic_works=[work], coverage_warnings=["partial"])
            html = Path(tmp, "report.html").read_text(encoding="utf-8")
            self.assertIn("经典文献补漏", html); self.assertIn("partial", html)
            self.assertIn("Library &lt;b&gt;paper", html); self.assertNotIn("<script>bad", html)
            write_rss([work], Path(tmp, "feed.xml"))
            self.assertIn("被相关论文引用", Path(tmp, "feed.xml").read_text(encoding="utf-8"))

    def test_ranker_returns_nearest_library_evidence_and_graph_bonus(self):
        with patch.dict(sys.modules, {"src.vectorizer": SimpleNamespace(TextVectorizer=object)}):
            from src.score_rank import WorkRanker
        ranker = object.__new__(WorkRanker)
        ranker.settings = self.settings; ranker.journal_metrics = {}
        ranker.profile = {"index_items": [{"title": "Library paper", "doi": "10.1/x"}]}
        ranker.vectorizer = SimpleNamespace(encode=lambda texts: np.zeros((len(texts), 2)))
        ranker.index = SimpleNamespace(search=lambda vectors, top_k: (np.ones((len(vectors), 1)), np.zeros((len(vectors), 1), dtype=int)))
        work = CandidateWork(source="test", identifier="w", title="Steel plasticity", extra={"cites_seeds": [{"id": "W1"}]})
        out = ranker.rank([work])[0]
        self.assertEqual(out.extra["nearest_library_work"]["title"], "Library paper")
        self.assertAlmostEqual(out.extra["citation_bonus"], self.settings.citation_watch.score_bonus * 2 / 3)

    def test_pipeline_old_reference_separate_and_history_commit_deferred(self):
        # Full orchestration; external network, model and Zotero are mocked, output files are real.
        with patch.dict(sys.modules, {"src.vectorizer": SimpleNamespace(TextVectorizer=object)}):
            import src.cli as cli
            run_watch = cli.run_watch
        now = datetime.now(timezone.utc)
        new = CandidateWork(source="test", identifier="new", title="Dynamic titanium alloy experiments", published=now)
        old = CandidateWork(source="test", identifier="old", title="Nonlocal regularization fundamentals",
                            published=now - timedelta(days=3650),
                            extra={"referenced_by": [{"id": "W1", "title": "Seed", "url": "https://openalex.org/W1"}]})
        weak = old.model_copy(update={"identifier": "weak", "title": "Unrelated low similarity", "extra": {**old.extra, "weak": True}})
        future = new.model_copy(update={"identifier": "future", "title": "Future work", "published": now + timedelta(days=10)})
        def rank(works):
            return [RankedWork(**w.model_dump(), score=0.7, similarity=0.1 if w.extra.get('weak') else 0.6,
                               recency_score=0, metric_score=0, author_bonus=0, venue_bonus=0, label="consider") for w in works]
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(cli, "ZoteroIngestor"), patch.object(cli, "ProfileBuilder"), \
             patch.object(cli, "CandidateFetcher") as fetch, patch.object(cli, "WorkRanker") as ranker, \
             patch.object(cli, "DedupeEngine") as dedupe, patch.object(cli, "CitationDiscovery") as graph, \
             patch.object(cli, "enrich_ranked_works", side_effect=lambda works, settings: works):
            fetch.return_value.fetch_all.return_value = [new, future]
            fetch.return_value._filter_by_topic.side_effect = lambda works: works
            dedupe.return_value.filter.side_effect = lambda works: list(works)
            ranker.return_value.rank.side_effect = rank
            graph.return_value.fetch.return_value = [old, weak]
            graph.return_value.annotate.side_effect = lambda works: works
            graph.return_value.warnings = []
            args = dict(rss=True, report=True, top=20, push=False, defer_history=True)
            run_watch(Path(tmp), self.settings, Mock(), **args)
            report = next(Path(tmp, "reports").glob("*.html"))
            html = report.read_text(encoding="utf-8")
            self.assertIn(old.title, html); self.assertIn(new.title, html)
            self.assertNotIn(weak.title, html); self.assertNotIn(future.title, html)
            history = WatchHistory(Path(tmp, "data", "watch-state"))
            self.assertEqual(len(history.filter([old, new])), 2)
            history.commit()
            run_watch(Path(tmp), self.settings, Mock(), **args)
            html = report.read_text(encoding="utf-8")
            self.assertNotIn(old.title, html); self.assertNotIn(new.title, html)
            self.assertIn("本轮无通过筛选", html)

    def test_request_pagination_and_truncation_warning(self):
        response = Mock()
        response.json.return_value = {"results": [raw("W2")] * 100, "meta": {"count": 300, "next_cursor": "next"}}
        with patch("src.citation_watch.request_with_retry", return_value=response) as req:
            self.assertEqual(len(list(self.discovery.query("cites:W1", max_pages=2))), 200)
            self.assertEqual(req.call_args.kwargs["params"]["cursor"], "next")
        self.assertTrue(self.discovery.warnings)


if __name__ == "__main__":
    unittest.main()
