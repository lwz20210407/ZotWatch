import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from unittest.mock import Mock

import numpy as np
from pydantic import ValidationError
from src.aminer_policy import applicability
from src.aminer_entities import lookup
from src.models import RankedWork, ZoteroItem
from src.research_features import FeedbackEntry, FeedbackModel, feedback_links, load_feedback, save_feedback
from src.problem_ranking import build_problem_profiles
from src.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


def work(**updates):
    params = dict(source="aminer", identifier="aminer:p1", title="Hydrogen-assisted ductile fracture of steel",
        abstract="Ductile fracture damage evolution and calibration of steel.", extra={"aminer_id": "p1"},
        score=0.6, similarity=0.8, recency_score=0, metric_score=0, author_bonus=0, venue_bonus=0, label="consider")
    params.update(updates)
    return RankedWork(**params)


class FeedbackV2Tests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(ROOT)
        self.config = self.settings.research

    def test_requires_valid_identity_and_positive_approval(self):
        for payload in [{"rating": "direct"}, {"work_id": "http://bad", "rating": "direct"},
                        {"work_id": "aminer:p1", "rating": "irrelevant", "applicability": "approve"}]:
            with self.assertRaises(ValidationError): FeedbackEntry(**payload)

    def test_no_doi_read_feedback_survives_doi_enrichment(self):
        model = FeedbackModel([FeedbackEntry(work_id="aminer:p1", rating="read")], self.config)
        for w in [work(), work(doi="10.1234/a")]:
            self.assertTrue(model.apply([w], self.settings.scoring.thresholds)[0].extra["feedback_read"])
        self.assertFalse(model.apply([work(identifier="aminer:p2", extra={"aminer_id": "p2"})], self.settings.scoring.thresholds)[0].extra["feedback_read"])

    def test_explicit_doi_reset_wins_over_legacy_aminer_feedback(self):
        model = FeedbackModel([FeedbackEntry(work_id="aminer:p1", rating="read"),
                               FeedbackEntry(doi="10.1234/a", rating="reset")], self.config)
        self.assertFalse(model.apply([work(doi="10.1234/a")], self.settings.scoring.thresholds)[0].extra["feedback_read"])

    def test_verified_identity_pair_merges_votes_and_reset(self):
        model = FeedbackModel([FeedbackEntry(work_id="aminer:p1", rating="direct", facets=["ductile_fracture"]),
            FeedbackEntry(doi="10.1234/a", work_id="aminer:p1", rating="reset")], self.config)
        self.assertEqual(len(model.entries), 1)
        self.assertEqual(model.preferences, {})

    def test_owner_approval_changes_conditional_route_and_hold_reverses_it(self):
        for decision, expected in [("approve", "suitable"), ("hold", "conditional"), ("auto", "conditional")]:
            model = FeedbackModel([FeedbackEntry(work_id="aminer:p1", rating="transferable", applicability=decision)], self.config)
            tagged = model.apply([work()], self.settings.scoring.thresholds)[0]
            self.assertEqual(applicability(tagged, self.config)["status"], expected)
            self.assertLessEqual(abs(tagged.score - 0.6), self.config.feedback_max_adjustment)

    def test_approval_never_overrides_editorial_or_missing_evidence(self):
        model = FeedbackModel([FeedbackEntry(work_id="aminer:p1", rating="direct", applicability="approve")], self.config)
        for w in [work(title="Preface to metal damage"), work(abstract=None)]:
            tagged = model.apply([w], self.settings.scoring.thresholds)[0]
            self.assertNotEqual(applicability(tagged, self.config)["status"], "suitable")

    def test_v1_and_v2_owner_only_issue_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "data").mkdir()
            def issue(owner, payload, version):
                return {"user": {"login": owner}, "body": f"<!-- zotwatch-feedback-v{version} -->\n```json\n{json.dumps(payload)}\n```"}
            (root / "data/feedback-issues.json").write_text(json.dumps([
                issue(self.config.feedback_owner, {"doi": "10.1234/a", "rating": "direct"}, 1),
                issue(self.config.feedback_owner, {"work_id": "aminer:p1", "rating": "later", "reason": "wrong_conditions"}, 2),
                issue("attacker", {"work_id": "aminer:p1", "rating": "direct", "applicability": "approve"}, 2)]), "utf-8")
            entries = load_feedback(root, self.config)
            self.assertEqual(len(entries), 2)
            self.assertEqual(next(e for e in entries if e.work_id).rating, "later")

    def test_local_no_doi_save_reset_and_reason_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config").mkdir()
            save_feedback(tmp, {"work_id": "aminer:p1", "rating": "irrelevant", "reason": "wrong_material", "facets": ["lpbf_tc4"]}, self.config)
            model = FeedbackModel(load_feedback(tmp, self.config), self.config)
            self.assertEqual(model.reason_summary()["lpbf_tc4"]["wrong_material"], 1)
            save_feedback(tmp, {"work_id": "aminer:p1", "rating": "reset"}, self.config)
            self.assertEqual(FeedbackModel(load_feedback(tmp, self.config), self.config).reason_summary(), {})

    def test_no_doi_links_use_v2_without_fake_doi(self):
        links = feedback_links(work(), self.config)
        body = parse_qs(urlparse(links[0]["url"]).query)["body"][0]
        self.assertIn("zotwatch-feedback-v2", body)
        self.assertIn('"work_id": "aminer:p1"', body)
        self.assertNotIn('"doi"', body)
        self.assertTrue(any(l["name"] == "确认方法可迁移" for l in links))

    def test_no_doi_feedback_does_not_match_all_no_doi_library_items(self):
        cfg = self.config.model_copy(deep=True); cfg.facets = cfg.facets[:1]
        items = [ZoteroItem(key="a", version=1, title="Unrelated subject")]
        result = build_problem_profiles(items, np.eye(1), cfg,
            [FeedbackEntry(work_id="aminer:p1", rating="direct", facets=[cfg.facets[0].id])])
        self.assertEqual(result, {})

    def test_entity_results_remain_ambiguous_and_projected(self):
        client = Mock()
        client.query.return_value = [{"id": "a", "name": "Smith", "org": "A", "private_unexpected": "ignored"},
                                     {"id": "b", "name": "Smith", "org": "B"}]
        result = lookup(client, "person", "Smith", organization="A")
        self.assertIsNone(result["selected_id"])
        self.assertEqual(len(result["candidates"]), 2)
        self.assertNotIn("private_unexpected", result["candidates"][0])
        self.assertEqual(client.query.call_args.args[1]["org"], "A")


if __name__ == "__main__": unittest.main()
