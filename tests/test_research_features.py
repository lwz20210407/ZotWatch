import copy
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

from src.models import RankedWork
from src.settings import load_settings
from src.research_features import (FeedbackEntry, FeedbackModel, RetrievalWarnings, coverage_report,
    feedback_links, load_feedback, propose_tracking, save_feedback)
from src.research_evidence import attach_evidence, abstract_signals
from src.version_watch import VersionMonitor, metadata_events
from src.watch_history import WatchHistory
from src.report_html import render_html


def paper(key="1", **kwargs):
    values = dict(source="openalex", identifier="https://openalex.org/W" + key, doi="10.1234/p" + key,
                  title="Steel LS-OPT inverse identification of plasticity", abstract="We used LS-OPT for inverse identification.",
                  score=0.7, similarity=0.7, recency_score=0, metric_score=0,
                  author_bonus=0, venue_bonus=0, label="consider", url="https://doi.org/10.1234/p" + key)
    values.update(kwargs)
    return RankedWork(**values)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(Path(__file__).resolve().parents[1])
        self.config = self.settings.research

    def test_feedback_normalizes_and_rejects_unknown_fields(self):
        self.assertEqual(FeedbackEntry(doi="https://doi.org/10.1234/ABC", rating="direct").doi, "10.1234/abc")
        with self.assertRaises(ValueError): FeedbackEntry(doi="no", rating="direct")
        with self.assertRaises(ValueError): FeedbackEntry(doi="10.1234/a", rating="direct", instructions="run command")

    def test_owner_only_and_latest_feedback_then_local_override(self):
        def issue(owner, rating, updated):
            payload = {"doi": "10.1234/a", "rating": rating, "facets": ["inverse"]}
            return {"user": {"login": owner}, "updated_at": updated,
                    "body": "<!-- zotwatch-feedback-v1 -->\n```json\n" + json.dumps(payload) + "\n```"}
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "data").mkdir(); Path(tmp, "config").mkdir()
            Path(tmp, "data", "feedback-issues.json").write_text(json.dumps([
                issue(self.config.feedback_owner, "direct", "1"),
                issue(self.config.feedback_owner, "transferable", "2"), issue("attacker", "irrelevant", "3")]), encoding="utf-8")
            self.assertEqual(load_feedback(tmp, self.config)[0].rating, "transferable")
            save_feedback(tmp, {"doi": "10.1234/a", "rating": "mechanism", "facets": ["inverse"]}, self.config)
            self.assertEqual(load_feedback(tmp, self.config)[0].rating, "mechanism")
            with self.assertRaises(ValueError): save_feedback(tmp, {"doi": "10.1234/a", "rating": "direct", "facets": ["steel"]}, self.config)

    def test_feedback_learns_method_not_material_and_is_bounded(self):
        model = FeedbackModel([FeedbackEntry(doi="10.1234/other", rating="transferable", facets=["inverse"])], self.config)
        steel = paper(); titanium = paper("2", title="TC4 inverse identification of plasticity")
        results = model.apply([steel, titanium], self.settings.scoring.thresholds)
        self.assertGreater(results[0].score, steel.score)
        self.assertAlmostEqual(results[0].extra['feedback_adjustment'], results[1].extra['feedback_adjustment'])
        self.assertLessEqual(abs(results[0].extra['feedback_adjustment']), self.config.feedback_max_adjustment)
        self.assertNotIn("feedback_adjustment", steel.extra)
        negative = FeedbackModel([FeedbackEntry(doi=steel.doi, rating="irrelevant", facets=["inverse"])], self.config)
        self.assertTrue(negative.apply([steel], self.settings.scoring.thresholds))  # No hard veto.

    def test_read_does_not_train_negative_preferences(self):
        model = FeedbackModel([FeedbackEntry(doi=paper().doi, rating="read", facets=["inverse"])], self.config)
        self.assertEqual(model.preferences, {})
        self.assertTrue(model.apply([paper()], self.settings.scoring.thresholds)[0].extra['feedback_read'])

    def test_feedback_link_is_prefilled_not_automatic_submission(self):
        link = feedback_links(paper(), self.config)[0]
        url = urlparse(link["url"])
        self.assertEqual(url.netloc, "github.com")
        self.assertIn("/issues/new", url.path)
        self.assertIn("zotwatch-feedback-v1", parse_qs(url.query)["body"][0])

    def test_coverage_distinguishes_filtering_and_outage(self):
        stages = {"raw": [paper()], "topic": [], "dedup": [], "delivered": []}
        state = {}
        rows = coverage_report(self.config, stages, [], state)
        row = next(r for r in rows if r["facet"] == "参数反演与硬化外推")
        self.assertIn("未通过主题筛选", row["status"])
        zeros = row["zero_runs"]
        rows = coverage_report(self.config, stages, ["failed"], state)
        self.assertTrue(all("覆盖不完整" in r["status"] for r in rows))
        self.assertEqual(next(r for r in rows if r["facet"] == row["facet"])["zero_runs"], zeros)

    def test_warning_public_output_has_no_secret_url(self):
        recorder = RetrievalWarnings()
        recorder.emit(logging.LogRecord("src.fetch_new", logging.WARNING, "x", 1, "request failed https://host/?api_key=SECRET", (), None))
        self.assertTrue(recorder.messages)
        self.assertNotIn("SECRET", str(recorder.messages))

    def test_abstract_signals_negation_and_no_reference_attribution(self):
        signals = abstract_signals("We did not use Johnson-Cook in this study.")
        self.assertIn("否定", signals[0]["role"])
        self.assertNotIn("target_doi", signals[0])
        self.assertEqual(abstract_signals("This paper cites prior literature."), [])

    def test_cards_abstract_only_unknowns_and_quote_budget(self):
        original = paper()
        result = attach_evidence(original, self.config)
        self.assertEqual(result.extra["evidence_level"], "摘要")
        self.assertIn("不能判断", result.extra["citation_context_status"])
        self.assertTrue(result.extra["transfer_cards"])
        quotes = [c["evidence"] for c in result.extra["transfer_cards"]] + [c["evidence"] for c in result.extra["abstract_method_signals"]]
        self.assertLessEqual(sum(len(q.replace("…", "").split()) for q in quotes), 24)
        self.assertNotIn("transfer_cards", original.extra)
        title_only = attach_evidence(paper(abstract=None), self.config)
        self.assertEqual(title_only.extra["evidence_level"], "仅题名")
        self.assertFalse(title_only.extra["abstract_method_signals"])

    def test_proposals_count_distinct_papers_not_runs_and_never_mutate_config(self):
        work = paper(extra={"openalex_authorships": [{"author_id": "A999999999", "name": "New person", "institution_ids": ["I1"]}]})
        state = {}; model = FeedbackModel([], self.config)
        before = self.settings.model_dump()
        self.assertFalse(propose_tracking([work], self.config, self.settings, state, model))
        self.assertFalse(propose_tracking([work], self.config, self.settings, state, model))
        other = paper("2", extra=work.extra)
        proposals = propose_tracking([other], self.config, self.settings, state, model)
        self.assertEqual(proposals[0]["count"], 2)
        self.assertEqual(self.settings.model_dump(), before)
        model = FeedbackModel([FeedbackEntry(doi=other.doi, rating="direct")], self.config)
        proposals = propose_tracking([other], self.config, self.settings, state, model)
        self.assertTrue(any(p["kind"] == "种子候选" for p in proposals))

    def test_version_relation_and_notice_direction(self):
        record = {"relation": {"is-preprint-of": [{"id-type": "doi", "id": "10.1234/final"}]}}
        notices = [{"DOI": "10.1234/notice", "update-to": [{"DOI": "10.1234/pre", "type": "correction"}]}]
        events = metadata_events("10.1234/pre", record, notices)
        self.assertEqual(len(events), 2)
        self.assertEqual(len(metadata_events("10.1234/other", {}, notices)), 0)

    def test_version_alert_bypasses_read_and_dedup_but_event_once(self):
        self.settings.citation_watch.seeds = []
        self.config.version_batch_size = 1
        monitor = VersionMonitor(self.settings, Mock())
        doi = paper().doi
        def get(url, params=None):
            if "openalex" in url: return {"results": [{"doi": doi, "is_retracted": False}]}
            if (params or {}).get("filter"):
                return {"message": {"items": [{"DOI": "10.1234/notice", "update-to": [{"DOI": doi, "type": "correction"}]}]}}
            return {"message": {"DOI": doi}}
        monitor.get_json = Mock(side_effect=get)
        with tempfile.TemporaryDirectory() as tmp:
            history = WatchHistory(tmp); history.stage([paper()]); history.commit()
            history = WatchHistory(tmp)
            alerts = monitor.check([paper()], history.state)
            self.assertEqual(len(alerts), 1)
            self.assertEqual(history.filter([paper()]), [])
            self.assertEqual(len(history.filter(alerts)), 1)
            # A failed delivery never persisted event IDs.
            self.assertNotIn("events_seen", WatchHistory(tmp).state)
            history.stage([]); history.commit()
            self.assertEqual(monitor.check([paper()], WatchHistory(tmp).state), [])

    def test_version_failure_rotates_without_claiming_success(self):
        self.settings.citation_watch.seeds = []; self.config.version_batch_size = 1
        state = {"catalog": {"10.1234/a": {"title": "A"}, "10.1234/b": {"title": "B"}}}
        monitor = VersionMonitor(self.settings, Mock()); monitor.get_json = Mock(return_value=None)
        monitor.check([], state); monitor.check([], state)
        self.assertEqual(len(state["version_attempted"]), 2)
        self.assertEqual(state["version_checked"], {})

    def test_new_report_sections_escape_untrusted_metadata(self):
        work = attach_evidence(paper(title="<script>bad</script> LS-OPT"), self.config)
        diagnostics = {"coverage": [{"facet": "<bad>", "raw": 1, "topic": 1, "dedup": 1,
                        "delivered": 1, "zero_runs": 0, "status": "ok"}], "feedback_count": 1}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "report.html"); render_html([work], path, diagnostics=diagnostics)
            html = path.read_text(encoding="utf-8")
            self.assertIn("研究问题覆盖诊断", html); self.assertIn("方法迁移说明卡", html)
            self.assertIn("公开 GitHub Issue", html); self.assertNotIn("<script>bad", html)


if __name__ == "__main__": unittest.main()
