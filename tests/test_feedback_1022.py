"""Issues #18-#22 (2026-10-15 / 10-22): five papers marked not relevant.

Two lessons are pinned here. First, each paper slipped through a hole in the demotion
vocabulary, and closing those holes must not catch what the owner actually reads -- the
guard titles below are real library papers kept on purpose. Second, "not relevant" used
to vote against every research direction the paper touched, so clicks on a gyroid
lattice and a laminated Ti/Mg plate penalised lpbf_tc4 and ductile_fracture across the
board. It now judges the paper only.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

from src.models import RankedWork
from src.research_features import FeedbackEntry, FeedbackModel
from src.settings import load_settings
from src.topic_matching import research_priority

REPO = Path(__file__).resolve().parents[1]

MARKED_IRRELEVANT = [
    "In-situ resistance heating-ultrasonic vibration composite forming of TC4 alloy sheet: Experimental and numerical study",
    "Beyond Composition: A Geometry-Aware Framework for Functionally Graded Material Design Demonstrated in Monel 400 to Inconel 625 Gradients",
    "Microstructural Design, Processing, and Dynamic Deformation of Advanced High-Strength Steels for Energy-Absorbing Applications",
    "Strength-ductility matching of hot-rolled pure titanium/ pure magnesium/magnesium alloy laminated composite plates regulated by low-temperature anneal",
    "Relative Density Effects on Deformation Mechanisms and Failure Modes in Gyroid Lattices",
]

# Library papers that near-synonyms of the new terms would have caught.
KEPT_IN_LIBRARY = [
    "On fracture mechanism of additively manufactured triply periodic minimal surface structures using an explicit phase field model",
    "Numerical and experimental investigation of ductile fracture in AA6061-T6 sheets under ultrasonic vibration loading",
    "Electrically-assisted anisotropic evolution behavior and constitutive model of AZ31 magnesium alloy sheets considering microstructure",
    "Ultrauniform, strong, and ductile 3D-printed titanium alloy through bifunctional alloy design",
    "Numerical simulation of deformation and damage of lattice structures made of austenitic stainless steel manufactured via laser powder bed fusion",
]


class DemotionVocabularyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scoring = load_settings(REPO).scoring

    def test_every_paper_marked_irrelevant_is_now_demoted(self):
        for title in MARKED_IRRELEVANT:
            name, multiplier = research_priority(SimpleNamespace(title=title, abstract=""), self.scoring)
            self.assertLessEqual(multiplier, 0.7, title)

    def test_the_new_terms_leave_kept_library_papers_alone(self):
        new_terms = ["gyroid", "laminated composite", "composite plate", "composite forming", "resistance heating",
                     "material design", "materials design", "microstructural design", "microstructure design"]
        from src.topic_matching import matches_any
        for title in KEPT_IN_LIBRARY:
            self.assertFalse(matches_any(title, new_terms), title)


def paper(**kwargs):
    values = dict(source="openalex", identifier="x", doi="10.1016/j.ijplas.2026.000001",
                  title="Ductile fracture of LPBF Ti-6Al-4V under combined stress states", abstract="",
                  score=0.72, similarity=0.72, recency_score=0, metric_score=0, author_bonus=0,
                  venue_bonus=0, label="must_read")
    values.update(kwargs)
    return RankedWork(**values)


class IrrelevantJudgesThePaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = load_settings(REPO)
        cls.config, cls.thresholds = settings.research, settings.scoring.thresholds

    def test_not_relevant_does_not_penalise_a_research_direction(self):
        clicks = [FeedbackEntry(doi=f"10.1016/j.mtla.2026.10292{i}", rating="irrelevant",
                                facets=["lpbf_tc4", "ductile_fracture", "damage_evolution"]) for i in range(5)]
        model = FeedbackModel(clicks, self.config)
        self.assertEqual(model.preferences, {})
        other = model.apply([paper()], self.thresholds)[0]
        self.assertEqual(other.extra["feedback_adjustment"], 0)
        self.assertEqual(other.label, "must_read", "an unrelated core paper keeps its label")

    def test_the_marked_paper_itself_is_still_pushed_down(self):
        target = paper(doi="10.1016/j.mtla.2026.102929")
        model = FeedbackModel([FeedbackEntry(doi=target.doi, rating="irrelevant", facets=["lpbf_tc4"])], self.config)
        self.assertLess(model.apply([target], self.thresholds)[0].score, target.score)

    def test_positive_feedback_still_teaches_a_direction(self):
        model = FeedbackModel([FeedbackEntry(doi="10.1016/j.x.2026.1", rating="direct", facets=["ductile_fracture"])],
                              self.config)
        self.assertGreater(model.preferences["ductile_fracture"], 0)


if __name__ == "__main__":
    unittest.main()
