"""What the score means, and what a direction label claims.

Three defects from the 2026-09-19 review, all in the same family: a value that was
only ever a proximity or a bonus got treated as a fact.

  B4  author and citation bonuses were added after the bounded weighted sum and after
      the priority multiplier, so a total reached 1.09 -- and 1.17 once feedback
      applied -- while the design asserted twice that it stayed in [0,1] and that those
      channels stayed out of the main ranking.
  B5  primary_problem is an argmax over facet centroids, i.e. "nearest direction in
      embedding space". Shown as a card heading it read as a classification, and filed
      an LPBF 316L steel paper and a Ti/CeO2 composite under "增材 TC4".
  B6  dedupe treated matching DOIs as evidence FOR duplication but never treated
      differing DOIs as evidence against, so a library "... Part I" suppressed a
      candidate "... Part II".
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.dedupe import DedupeEngine
from src.models import CandidateWork, RankedWork, ZoteroItem
from src.report_html import direction
from src.research_features import facet_ids
from src.settings import load_settings
from src.storage import ProfileStorage

BASE = Path(__file__).resolve().parents[1]


def candidate(title, abstract="", **extra):
    return CandidateWork(source="t", identifier=extra.pop("identifier", "x"),
                         title=title, abstract=abstract, **extra)


class MaterialFactTests(unittest.TestCase):
    """A material claim needs the material named, not a shared process word."""

    def setUp(self):
        self.settings = load_settings(BASE)

    def facets(self, title, abstract=""):
        return facet_ids(candidate(title, abstract), self.settings.research)

    def test_a_steel_paper_is_not_labelled_as_additive_tc4(self):
        ids = self.facets("LPBF 316L steel plasticity and ductile fracture",
                          "Laser powder bed fusion 316L stainless steel; porosity and "
                          "build orientation effects on ductile fracture.")
        self.assertNotIn("lpbf_tc4", ids)

    def test_a_composite_paper_is_not_labelled_as_additive_tc4(self):
        ids = self.facets("Dual-scale Ti/CeO2 hybrid reinforcement improves fracture energy",
                          "Additively manufactured Ti/CeO2 composite; porosity.")
        self.assertNotIn("lpbf_tc4", ids)

    def test_a_genuine_lpbf_titanium_paper_still_is(self):
        ids = self.facets("Build orientation effects on the dynamic tensile fracture of LPBF Ti-6Al-4V",
                          "Split Hopkinson tension on laser powder bed fusion Ti-6Al-4V; "
                          "lack-of-fusion defects control the fracture strain.")
        self.assertIn("lpbf_tc4", ids)

    def test_directions_are_multi_label(self):
        """Conditions, phenomena and methods are not mutually exclusive."""
        ids = self.facets(
            "Temperature and strain-rate dependent ductile fracture of Ti-6Al-4V "
            "calibrated by inverse identification and implemented as a LS-DYNA UMAT",
            "Split Hopkinson bar, stress triaxiality and Lode angle, GISSMO damage, "
            "ballistic perforation validation.")
        self.assertGreater(len(ids), 2, f"expected several directions, got {ids}")

    def test_a_facet_without_requires_stays_a_plain_or(self):
        without = [f for f in self.settings.research.facets if not f.requires]
        self.assertTrue(without, "all facets became conjunctions; that is too strict")
        facet = without[0]
        self.assertIn(facet.id, self.facets(f"A study of {facet.terms[0]}"))


class DirectionLabelTests(unittest.TestCase):
    """The card heading must prefer evidence over vector proximity."""

    NAMES = {"ductile_fracture": "延性断裂准则", "lpbf_tc4": "增材 TC4 缺陷与力学响应"}

    def work(self, **extra):
        return RankedWork(source="t", identifier="x", title="T", authors=[], metrics={},
                          score=0.5, similarity=0.5, recency_score=0, metric_score=0,
                          author_bonus=0, venue_bonus=0, label="consider", extra=extra)

    def test_an_evidenced_direction_wins_over_the_nearest_centroid(self):
        shown = direction(self.work(primary_problem="lpbf_tc4",
                                    research_facets=["ductile_fracture"]), self.NAMES)
        self.assertEqual(shown, "延性断裂准则")

    def test_nearest_centroid_is_the_last_resort_only(self):
        shown = direction(self.work(primary_problem="lpbf_tc4"), self.NAMES)
        self.assertEqual(shown, "增材 TC4 缺陷与力学响应")

    def test_the_priority_label_beats_the_nearest_centroid(self):
        shown = direction(self.work(primary_problem="lpbf_tc4",
                                    research_priority="跨金属方法参考"), self.NAMES)
        self.assertEqual(shown, "跨金属方法参考")


class ScoreRangeTests(unittest.TestCase):
    """Every combination of bonuses and feedback must stay inside [0,1]."""

    def ranker(self, settings):
        with patch.dict("sys.modules",
                        {"src.vectorizer": SimpleNamespace(TextVectorizer=object)}):
            from src.score_rank import WorkRanker
        r = object.__new__(WorkRanker)
        r.settings = settings
        r.journal_metrics = {}
        r.profile = {}
        r.vectorizer = SimpleNamespace(encode=lambda texts: np.zeros((len(texts), 2)),
                                       text_separator="\n")
        # Perfect similarity, so the bounded sum is at its ceiling.
        r.index = SimpleNamespace(search=lambda vectors, top_k: (np.ones((len(vectors), 1)), None),
                                  ntotal=10)
        return r

    def test_a_watched_cited_top_priority_paper_does_not_exceed_one(self):
        settings = load_settings(BASE)
        work = candidate(
            "Ti-6Al-4V ductile fracture under dynamic loading: constitutive calibration",
            "Johnson-Cook, stress triaxiality, Lode, ballistic validation, UMAT.",
            extra={"watched_authors": [{"name": "X"}],
                   "cites_seeds": [{"title": "S"}], "referenced_by": [{"title": "R"}]},
        )
        out = self.ranker(settings).rank([work])[0]
        self.assertLessEqual(out.score, 1.0, f"score escaped the declared range: {out.score}")
        self.assertGreaterEqual(out.score, 0.0)

    def test_feedback_cannot_push_a_score_past_one(self):
        from src.research_features import FeedbackModel
        settings = load_settings(BASE)
        top = RankedWork(source="t", identifier="x", title="T", authors=[], metrics={},
                         doi="10.1/a", score=1.0, similarity=1.0, recency_score=1,
                         metric_score=1, author_bonus=0, venue_bonus=0, label="must_read",
                         extra={})
        model = FeedbackModel([], settings.research)
        model.preferences = {"": 1.0}  # strongest possible positive preference
        out = model.apply([top], settings.scoring.thresholds)[0]
        self.assertLessEqual(out.score, 1.0, f"feedback escaped the range: {out.score}")


class DedupeIdentityTests(unittest.TestCase):
    """A differing DOI is evidence AGAINST duplication."""

    def engine(self, library):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = ProfileStorage(Path(tmp.name, "p.sqlite"))
        store.initialize()
        self.addCleanup(store.close)
        for key, title, doi in library:
            store.upsert_item(ZoteroItem(key=key, version=0, title=title, abstract="",
                                         creators=[], tags=[], collections=[], doi=doi,
                                         item_type="journalArticle"))
        return DedupeEngine(store)

    def test_part_two_survives_part_one_being_in_the_library(self):
        engine = self.engine([("L1", "Ductile fracture of titanium alloys Part I: experiments",
                               "10.1016/j.a.2025.1")])
        kept = engine.filter([candidate(
            "Ductile fracture of titanium alloys Part II: modelling",
            identifier="new", doi="10.1016/j.a.2025.2")])
        self.assertEqual(len(kept), 1, "a different DOI must override a fuzzy title match")

    def test_the_same_doi_is_still_a_duplicate(self):
        engine = self.engine([("L1", "Ductile fracture of titanium alloys",
                               "10.1016/j.a.2025.1")])
        kept = engine.filter([candidate("Ductile fracture of titanium alloys",
                                        identifier="new", doi="10.1016/j.a.2025.1")])
        self.assertEqual(kept, [])

    def test_without_a_doi_the_title_match_still_decides(self):
        """No identity evidence either way, so the conservative rule stands."""
        engine = self.engine([("L1", "Ductile fracture of titanium alloys", "")])
        kept = engine.filter([candidate("Ductile fracture of titanium alloys",
                                        identifier="new")])
        self.assertEqual(kept, [])


if __name__ == "__main__":
    unittest.main()
