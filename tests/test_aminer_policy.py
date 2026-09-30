import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.aminer_source import AMinerSource, query_plan
from src.aminer_policy import applicability, screen_aminer_delivery, recent_delivery, select_backfill
from src.cli import _filter_recent
from src.models import RankedWork
from src.settings import load_settings
from src.report_html import render_html
from src.rss_writer import write_rss
from src.watch_history import WatchHistory, history_keys

ROOT = Path(__file__).resolve().parents[1]


def work(title="A metal ductile fracture model", **changes):
    fields = dict(source="aminer", identifier="aminer:p", title=title,
        abstract="Ductile fracture and damage evolution of metals using constitutive calibration.",
        score=0.7, similarity=0.8, label="consider", recency_score=0, metric_score=0,
        author_bonus=0, venue_bonus=0, extra={"aminer_id": "p", "publication_year": 2026})
    fields.update(changes)
    return RankedWork(**fields)


class FakeClient:
    def __init__(self):
        self.warnings = []
        self.stopped = False
        self.calls = []
    def check_cooldown(self): pass
    def warn(self, warning): self.warnings.append(warning)
    def summary(self): return {"calls": len(self.calls)}
    def query(self, endpoint, params, **kwargs):
        self.calls.append((endpoint, params))
        if endpoint == "recommend":
            return [{"id": str(i), "title": "Metal ductile fracture " + str(i), "abstract": "Fracture model"} for i in range(5)]
        return []


class AMinerRefinementTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(ROOT)
        cfg = self.settings.sources.aminer
        cfg.enabled = True
        cfg.max_recommendation_queries = 4  # Exercise redundancy independently of production rollout quota.
        cfg.max_enrich_items = 0
        cfg.max_identity_lookups = 0

    def test_repeated_recommendations_stop_but_search_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient()
            source = AMinerSource(self.settings, Path(tmp), client=client)
            rows = source.fetch()
            self.assertEqual(sum(e == "recommend" for e, _ in client.calls), 3)
            self.assertEqual(sum(e == "search" for e, _ in client.calls), 22)
            self.assertEqual(len(rows), 5)
            self.assertTrue(source.stats["recommendation_redundancy_stop"])
            self.assertEqual(source.stats["recommendation_skipped"], 8)
            # Even duplicate responses preserve the discovery routes that were tried.
            self.assertEqual(len(rows[0].extra["aminer_facets"]), 3)

    def test_recommendation_rotation_reaches_other_directions_next_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = FakeClient(), FakeClient()
            AMinerSource(self.settings, Path(tmp), client=a).fetch()
            AMinerSource(self.settings, Path(tmp), client=b).fetch()
            first = [p["topics"] for e, p in a.calls if e == "recommend"]
            second = [p["topics"] for e, p in b.calls if e == "recommend"]
            self.assertNotEqual(first, second)
            self.assertFalse(any(t in first for t in second))

    def test_recommendations_can_be_disabled_without_disabling_search(self):
        self.settings.sources.aminer.max_recommendation_queries = 0
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient()
            AMinerSource(self.settings, Path(tmp), client=client).fetch()
            self.assertTrue(client.calls)
            self.assertTrue(all(e == "search" for e, _ in client.calls))

    def test_distinct_recommendations_still_respect_query_cap(self):
        class Distinct(FakeClient):
            def query(self, endpoint, params, **kwargs):
                rows = super().query(endpoint, params, **kwargs)
                for row in rows:
                    row["id"] += "_" + str(len(self.calls))
                return rows
        with tempfile.TemporaryDirectory() as tmp:
            client = Distinct()
            source = AMinerSource(self.settings, Path(tmp), client=client)
            source.fetch()
            self.assertEqual(source.stats["recommendation_queries"], 4)
            self.assertFalse(source.stats["recommendation_redundancy_stop"])
            self.assertEqual(sum(e == "search" for e, _ in client.calls), 22)

    def test_aminer_phrases_do_not_invalidate_library_embedding_profile(self):
        before = self.settings.research.derived_fingerprint()
        plan_before = query_plan(self.settings)
        self.settings.research.facets[0].aminer_phrases = ["TC4"]
        self.assertEqual(before, self.settings.research.derived_fingerprint())
        self.assertNotEqual(plan_before, query_plan(self.settings))

    def test_other_metals_same_chain_are_suitable(self):
        for title in ["Ductile fracture of aluminum under combined tension and torsion",
                      "Strain rate dependence of strengthening mechanisms in lath martensite",
                      "Inverse identification of anisotropic plasticity with non-linear VFM"]:
            self.assertEqual(applicability(work(title), self.settings.research)["status"], "suitable")

    def test_conditional_regimes_are_retained_for_review_not_auto_delivered(self):
        for title in ["Hydrogen-assisted cracking of martensitic steels",
                      "Inverse identification of Johnson-Cook parameters for turning 304 steel",
                      "Ballistic limit of sandwich plates with a metal foam core"]:
            original = work(title)
            allowed, review = screen_aminer_delivery([original], self.settings.research)
            self.assertEqual(allowed, [])
            self.assertEqual(review["review"][0]["status"], "conditional")
            self.assertEqual(original.score, 0.7)
            self.assertNotIn("aminer_applicability", original.extra)

    def test_existing_source_is_not_penalized_by_aminer_metadata(self):
        original = work("Hydrogen-assisted cracking", source="openalex")
        allowed, review = screen_aminer_delivery([original], self.settings.research)
        self.assertEqual(allowed, [original])
        self.assertEqual(review["held_count"], 0)

    def test_preface_and_nonmetal_examples_are_excluded(self):
        for title in ["Preface to metallurgy and ductile fracture", "Particle breakage of calcareous sand",
                      "Strain rate and rock acoustic emission characteristics"]:
            self.assertEqual(applicability(work(title), self.settings.research)["status"], "exclude")

    def test_generated_summary_is_not_applicability_evidence(self):
        item = work(abstract=None, extra={"aminer_id": "p", "generated_summary": "A useful Ti6Al4V fracture study"})
        self.assertEqual(applicability(item, self.settings.research)["status"], "insufficient_evidence")

    def test_default_delivery_does_not_enter_recent_even_with_exact_date(self):
        now = datetime.now(timezone.utc)
        item = work(published=now)
        self.assertEqual(_filter_recent([item], days=30), [item])
        self.assertEqual(recent_delivery([item], self.settings.sources.aminer), [])
        self.settings.sources.aminer.delivery = "recent_and_backfill"
        self.assertEqual(recent_delivery([item], self.settings.sources.aminer), [item])

    def test_backfill_quota_future_dates_and_conditions(self):
        now = datetime.now(timezone.utc)
        items = [work(identifier="aminer:" + str(i)) for i in range(6)]
        items.insert(0, work("Hydrogen-assisted cracking"))
        items.insert(0, work(published=now + timedelta(days=10)))
        selected = select_backfill(items, set(), self.settings, now)
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(w.title == "A metal ductile fracture model" for w in selected))
        self.assertTrue(all("方法补漏" in w.extra["report_channel"] for w in selected))

    def test_review_is_visible_but_not_delivered_or_recorded_as_sent(self):
        held = work("Hydrogen-assisted cracking of steel", identifier="aminer:held")
        suitable = work(identifier="aminer:good")
        allowed, review = screen_aminer_delivery([held, suitable], self.settings.research)
        selected = select_backfill(allowed, set(), self.settings, datetime.now(timezone.utc))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render_html([], root / "review.html", exploration_works=selected,
                        diagnostics={"aminer": {"applicability": review}})
            write_rss(selected, root / "feed.xml")
            page = (root / "review.html").read_text("utf-8")
            feed = (root / "feed.xml").read_text("utf-8")
            self.assertIn("AMiner 条件参考与待补证", page)
            self.assertIn(held.title, page)
            self.assertNotIn(held.title, feed)
            history = WatchHistory(root / "history")
            history.stage(selected)
            sent = json.loads(history.pending.read_text("utf-8"))["sent"]
            self.assertFalse(history_keys(held).intersection(sent))
            self.assertTrue(history_keys(suitable).intersection(sent))


if __name__ == "__main__": unittest.main()
