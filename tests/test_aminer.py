import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from pydantic import ValidationError
from src.aminer_client import AMinerClient, AMinerError
from src.aminer_source import AMinerSource, align_identities, normalize_paper, query_plan
from src.aminer_compare import compare, bounded_rank
from src.citation_watch import merge_candidates
from src.cli import _filter_recent
from src.fetch_new import CandidateFetcher
from src.models import CandidateWork
from src.report_html import published_label, citation_label
from src.settings import AMinerConfig, load_settings

ROOT = Path(__file__).resolve().parents[1]


def response(rows=None, status=200, **body):
    r = Mock(status_code=status, headers={})
    r.json.return_value = {"success": True, "code": 200, "data": rows or [], **body}
    return r


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = Mock()
        self.config = AMinerConfig(interval_seconds=0, max_attempts=2)
        self.client = AMinerClient(self.config, Path(self.temp.name), session=self.session, token="private-token")

    def test_free_allowlist_and_no_request_without_credentials(self):
        with self.assertRaises(AMinerError): self.client.query("paper_detail", {"id": "x"})
        self.client._token = ""
        with self.assertRaises(AMinerError): self.client.query("search", {"title": "x"})
        self.session.request.assert_not_called()
        with self.assertRaises(ValidationError): AMinerConfig(free_only=False)

    def test_post_cache_canonical_body_and_secret_absent(self):
        self.session.request.return_value = response([{"id": "p", "title": "A"}])
        first = self.client.query("recommend", {"topics": ["fracture"], "size": 1})
        self.assertEqual(self.client.query("recommend", {"size": 1, "topics": ["fracture"]}), first)
        self.assertEqual(self.session.request.call_count, 1)
        self.assertFalse(self.session.request.call_args.kwargs["allow_redirects"])
        self.assertNotIn("private-token", "".join(p.read_text() for p in Path(self.temp.name).glob("*.json")))
        self.client.query("recommend", {"size": 2, "topics": ["fracture"]})
        self.assertEqual(self.session.request.call_count, 2)

    def test_live_rec5_envelope_is_unwrapped_including_cached_responses(self):
        self.session.request.return_value = response([{"analyzed_topics": ["fracture"], "papers": [{"paper_id": "p", "title": "TC4 fracture"}], "size": 1}])
        rows = self.client.query("recommend", {"topics": ["fracture"], "size": 1})
        self.assertEqual(rows[0]["paper_id"], "p")
        self.assertEqual(self.client.query("recommend", {"topics": ["fracture"], "size": 1}), rows)

    def test_retry_counts_and_cap(self):
        self.client.config.max_requests = 2
        self.session.request.side_effect = [requests.Timeout(), response()]
        self.assertEqual(self.client.query("search", {"title": "a"}), [])
        self.assertEqual(self.client.calls, 2)
        with self.assertRaisesRegex(AMinerError, "request_cap"):
            self.client.query("search", {"title": "b"})

    def test_deadline_reserves_metadata_time(self):
        self.client.config.max_run_seconds = 100
        self.client.started -= 85
        with self.assertRaisesRegex(AMinerError, "discovery_time_cap"):
            self.client.query("search", {"title": "a"}, reserve=1)
        self.session.request.assert_not_called()
        self.session.request.return_value = response([{"id": "a", "abstract_slice": "partial"}])
        self.assertEqual(len(self.client.query("info", {"ids": ["a"]})), 1)

    def test_business_auth_error_even_http500_is_not_retried_or_cached(self):
        self.session.request.return_value = response(status=500, code=40308, success=False, msg="private-token")
        with self.assertRaisesRegex(AMinerError, "credential_or_permission_failure"):
            self.client.query("search", {"title": "a"})
        self.assertEqual(self.client.calls, 1)
        self.assertFalse(list(Path(self.temp.name).glob("*.json")))

    def test_rate_limit_persists_without_sleeping(self):
        self.session.request.return_value = response(status=200, code=40306, success=False)
        with patch("src.aminer_client.time.sleep") as sleep:
            with self.assertRaisesRegex(AMinerError, "rate_limited"):
                self.client.query("search", {"title": "a"})
            sleep.assert_not_called()
        other = AMinerClient(self.config, Path(self.temp.name), token="t")
        with self.assertRaisesRegex(AMinerError, "cooldown"): other.check_cooldown()

    def test_invalid_shape_and_redirect_not_cached(self):
        for r in [response(status=302), response(data="unexpected"), response(code=40001, success=False)]:
            self.session.request.return_value = r
            with self.assertRaises(AMinerError): self.client.query("search", {"title": "a"})
        self.assertFalse(list(Path(self.temp.name).glob("*.json")))


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(ROOT)
        self.settings.sources.aminer.enabled = True
        self.settings.sources.aminer.interval_seconds = 0

    def test_year_and_generated_summary_never_become_date_or_abstract(self):
        w = normalize_paper({"id": "123", "title": "TC4 fracture", "year": 2026, "summary": "Generated text"}, route="recommend")
        self.assertIsNone(w.published)
        self.assertIsNone(w.abstract)
        self.assertEqual(w.identifier, "aminer:123")
        self.assertIn("2026", published_label(w))
        self.assertEqual(_filter_recent([w], days=30), [])
        self.assertEqual(citation_label(w), "未提供")
        w.extra["aminer_citation_bucket"] = "51-200"
        self.assertEqual(citation_label(w), "51-200")
        self.assertEqual(w.metrics, {})

    def test_merge_preserves_routes_and_full_abstract_dates_metrics(self):
        a = normalize_paper({"id": "p", "title": "TC4 fracture", "doi": "10.1234/x", "year": 2026, "abstract_slice": "partial"}, route="search", facet="fracture")
        b = CandidateWork(source="openalex", identifier="W2", doi="https://doi.org/10.1234/X", title=a.title,
            abstract="Full original abstract", published=datetime.now(timezone.utc), metrics={"citations": 4})
        c = merge_candidates([a, b])[0]
        self.assertEqual(c.abstract, b.abstract)
        self.assertFalse(c.extra["abstract_is_partial"])
        self.assertEqual(c.published, b.published)
        self.assertEqual(c.extra["date_precision"], "day")
        self.assertEqual(c.extra["external_ids"]["aminer"], "p")
        self.assertEqual(len(c.extra["provenance"]), 2)
        self.assertEqual(c.extra["source_metrics"]["openalex"], {"citations": 4})
        self.assertEqual(a.abstract, "partial")
        other = b.model_copy(update={"doi": "10.1234/y"})
        self.assertEqual(len(merge_candidates([a, other])), 2)

    def test_info_is_joined_by_id_and_applied_before_topic_gate(self):
        self.settings.research.facets = self.settings.research.facets[:1]
        self.settings.sources.aminer.phrases_per_facet = 0
        self.settings.sources.aminer.max_identity_lookups = 0
        client = Mock()
        client.stopped = False
        client.query.side_effect = [[{"id": "p", "title": "TC4 experiments", "year": 2026}],
            [{"id": "unexpected", "title": "wrong", "abstract_slice": "incorrect"},
             {"id": "p", "title": "TC4 experiments", "abstract_slice": "Ductile fracture plasticity of Ti-6Al-4V under high strain rate."}]]
        client.summary.return_value = {}
        with tempfile.TemporaryDirectory() as tmp:
            out = AMinerSource(self.settings, Path(tmp), client=client).fetch()
        self.assertEqual(len(out), 1)
        self.assertIn("Ductile fracture", out[0].abstract)
        self.assertTrue(out[0].extra["abstract_is_partial"])
        f = object.__new__(CandidateFetcher); f.settings = self.settings
        self.assertEqual(len(f._filter_by_topic(out)), 1)

    def test_query_plan_uses_short_phrases_before_recommendations(self):
        plan = query_plan(self.settings)
        self.assertEqual({f for f, endpoint, _ in plan[:11]}, {f.id for f in self.settings.research.facets})
        self.assertTrue(all(endpoint == "search" for _, endpoint, _ in plan[:11]))
        self.assertTrue(all(endpoint == "recommend" for _, endpoint, _ in plan[11:22]))
        self.assertEqual(plan[0][2]["title"], "Ti6Al4V")
        self.assertTrue(all(len(p.get("title", "")) < 100 for _, e, p in plan if e == "search"))

    def test_aminer_route_does_not_bypass_professional_gate(self):
        w = normalize_paper({"id": "p", "title": "Microstructure of LPBF Ti-6Al-4V", "abstract_slice": "EBSD TEM precipitation and texture."}, route="recommend", facet="lpbf_tc4")
        f = object.__new__(CandidateFetcher); f.settings = self.settings
        self.assertEqual(f._filter_by_topic([w]), [])

    def test_source_off_and_shadow_never_change_baseline(self):
        base = CandidateWork(source="test", identifier="1", title="TC4 ductile fracture")
        new = normalize_paper({"id": "p", "title": "TC4 plasticity and ductile fracture"}, route="search")
        with tempfile.TemporaryDirectory() as tmp:
            f = CandidateFetcher(self.settings, Path(tmp))
            f._fetch_existing = Mock(return_value=[base])
            with patch("src.fetch_new.AMinerSource") as source:
                source.return_value.fetch.return_value = [new]
                source.return_value.stats = {}
                self.settings.sources.aminer.mode = "shadow"
                self.assertEqual(f.fetch_all(), [base])
                self.assertEqual(len(f.discovery_comparison["aminer"]), 1)
                self.settings.sources.aminer.enabled = False
                source.reset_mock()
                self.assertEqual(f.fetch_all(), [base])
                source.assert_not_called()

    def test_live_failure_keeps_existing_candidates(self):
        self.settings.sources.aminer.mode = "live"
        base = CandidateWork(source="test", identifier="1", title="TC4 ductile fracture")
        with tempfile.TemporaryDirectory() as tmp:
            f = CandidateFetcher(self.settings, Path(tmp))
            f._fetch_existing = Mock(return_value=[base])
            with patch("src.fetch_new.AMinerSource") as source:
                source.return_value.fetch.side_effect = AMinerError("service_failure")
                source.return_value.stats = {"warnings": ["service_failure"]}
                self.assertEqual(f.fetch_all()[0].title, base.title)
                self.assertEqual(f.aminer_summary["warnings"], ["service_failure"])

    def test_optional_aminer_cache_failure_does_not_break_baseline(self):
        base = CandidateWork(source="test", identifier="1", title="TC4 ductile fracture")
        with tempfile.TemporaryDirectory() as tmp:
            f = CandidateFetcher(self.settings, Path(tmp))
            f._fetch_existing = Mock(return_value=[base])
            with patch("src.fetch_new.AMinerSource") as source:
                source.return_value.fetch.return_value = []
                source.return_value.stats = {}
                source.return_value.record_topic_results.side_effect = OSError('cache unavailable')
                self.assertEqual(f.fetch_all(), [base])
                self.assertEqual(f.discovery_comparison['combined'], [base])
                self.assertIn('source_processing_failure', f.aminer_summary['warnings'])
            with patch('src.fetch_new.AMinerSource', side_effect=OSError('constructor failure')):
                self.assertEqual(f.fetch_all(), [base])

    def test_baseline_day_precision_survives_year_only_overlay(self):
        a = CandidateWork(source="crossref", identifier="x", doi="10.1234/x", title="TC4 fracture", published=datetime.now(timezone.utc))
        b = normalize_paper({"id": "p", "doi": a.doi, "title": a.title, "year": 2026}, route="search")
        c = merge_candidates([a, b])[0]
        self.assertEqual(c.extra["date_precision"], "day")
        self.assertEqual(_filter_recent([c], days=30), [c])

    def test_year_precision_not_recent_even_with_synthetic_day(self):
        w = CandidateWork(source="crossref", identifier="x", title="x", published=datetime.now(timezone.utc), extra={"date_precision": "year"})
        self.assertEqual(_filter_recent([w], days=30), [])

    def test_same_aminer_id_and_conservative_cross_source_identity(self):
        a = normalize_paper({"id": "p", "title": "TC4 fracture", "year": 2026, "authors": ["John Smith"]}, route="recommend")
        b = a.model_copy(update={"doi": "10.1234/x"})
        self.assertEqual(len(merge_candidates([a, b])), 1)
        self.assertEqual(align_identities([a], [b])[0].doi, b.doi)
        self.assertIsNone(align_identities([a], [b, b.model_copy(update={"doi": "10.1234/y"})])[0].doi)
        self.assertIsNone(align_identities([a], [b.model_copy(update={"authors": ["Jane Smith"]})])[0].doi)

    def test_cached_config_changes_invalidate_legacy_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = CandidateFetcher(self.settings, Path(tmp))
            f._save_cache([])
            self.assertIsNotNone(f._load_cache())
            self.settings.sources.aminer.mode = "live"
            self.assertIsNotNone(f._load_cache())
            self.settings.sources.include_keywords.append("new-term")
            self.assertIsNone(f._load_cache())

    def test_exact_doi_metadata_checks_identity_and_preserves_unknown_date(self):
        a = normalize_paper({"id": "p", "doi": "10.1234/x", "title": "TC4 ductile fracture", "year": 2026}, route="search")
        with tempfile.TemporaryDirectory() as tmp:
            source = AMinerSource(self.settings, Path(tmp), client=Mock())
            source.client.warnings = []
            source.stats = {}
            session = Mock()
            session.request.return_value = response()
            session.request.return_value.json.return_value = {
                "id": "https://openalex.org/W1", "doi": "https://doi.org/10.1234/other",
                "display_name": a.title, "publication_date": "2026-09-20"}
            self.assertIsNone(source.resolve_dates([a], session)[0].published)
            session.request.return_value.json.return_value["doi"] = "https://doi.org/10.1234/x"
            b = source.resolve_dates([a], session)[0]
            self.assertEqual(b.published.day, 20)
            self.assertEqual(b.extra["date_precision"], "day")
            self.assertEqual(b.extra["external_ids"]["aminer"], "p")
            self.assertEqual(source.stats["doi_metadata_requests"], 1)

    def test_optional_ranking_timeout_preserves_candidate_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "candidates.json").write_text('{"preserved": true}', "utf-8")
            with patch("multiprocessing.get_context") as context:
                worker = context.return_value.Process.return_value
                worker.is_alive.return_value = True
                status = bounded_rank(ROOT, self.settings, {}, out, 1)
                worker.terminate.assert_called_once()
                self.assertEqual(status["status"], "timeout")
            self.assertTrue(json.loads((out / "candidates.json").read_text())["preserved"])


if __name__ == "__main__": unittest.main()
