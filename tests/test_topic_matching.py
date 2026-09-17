"""Regression cases for research recall, false positives and priority scoring."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

from src.fetch_new import CandidateFetcher
from src.models import CandidateWork, RankedWork
from src.settings import load_settings
from src.topic_matching import matches_term, research_priority


BASE = Path(__file__).resolve().parents[1]


class TopicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = load_settings(BASE)
        # Filtering needs settings only; bypass network/cache initialization.
        cls.fetcher = object.__new__(CandidateFetcher)
        cls.fetcher.settings = cls.settings

    def kept(self, title: str, venue: str = "") -> bool:
        work = CandidateWork(source="test", identifier="test", title=title, venue=venue)
        return bool(self.fetcher._filter_by_topic([work]))

    def test_boundaries(self) -> None:
        for text, term in [
            ("prediction", "DIC"), ("generalized model", "gene"),
            ("cellular metal", "cell"), ("modelling", "Lode"),
            ("nickname", "nickel"), ("shielding", "Hill"),
        ]:
            with self.subTest(text=text, term=term):
                self.assertFalse(matches_term(text, term))
        for text, term in [("DIC measurements", "DIC"), ("genes", "gene"),
                           ("cells", "cell"), ("constitutive models", "constitutive model")]:
            with self.subTest(text=text, term=term):
                self.assertTrue(matches_term(text, term))

    def test_aliases(self) -> None:
        for alias in ["Ti-6Al4V", "Ti6Al-4V", "Ti–6Al–4V", "Ti 6Al 4V", "Ti6Al4V"]:
            with self.subTest(alias=alias):
                self.assertTrue(matches_term(alias, "Ti-6Al-4V"))
                self.assertTrue(self.kept(f"Ductile fracture of {alias}"))
        for alias in ["SLM", "L-PBF", "PBF-LB/M", "selective laser melting",
                      "laser beam powder bed fusion"]:
            with self.subTest(alias=alias):
                self.assertTrue(matches_term(alias, "LPBF"))
        self.assertTrue(matches_term("Johnson–Cook model", "Johnson Cook"))
        self.assertTrue(matches_term("MAT_224", "MAT 224"))

    def test_relevant_recall(self) -> None:
        titles = [
            "Generalized constitutive model for Ti-6Al-4V",
            "Constitutive modelling parameters of additively manufactured copper",
            "Electroplasticity-based constitutive modeling of Inconel 718",
            "LYP steel dampers: Constitutive modeling and structural implications",
            "Cellular metal plasticity with GISSMO damage",
            "Crystal plasticity and backstress in titanium alloys",
            "Ductile-brittle transition of Ti6Al4V",
            "Mechanical properties of heat-treated Ti-6Al-4V: ductile fracture",
            "Process-microstructure relationships and damage in titanium alloys",
            "Steel ductile damage mesh objectivity regularization",
            "Aluminum alloy fracture energy and characteristic length",
            "Bayesian calibration and parameter identifiability for steel plasticity",
            "Finite element model updating for copper hardening",
            "Ti-6Al-4V post-necking identification and plastic hardening",
            "Full-field identification of aluminum alloy flow stress",
            "Virtual fields method for magnesium alloy plasticity",
            "LPBF Ti-6Al-4V lack-of-fusion defects and fracture",
            "Ti-6Al-4V martensite decomposition and ductility",
            "Ti-6Al-4V hot isostatic pressing and damage",
            "LS-OPT inverse identification of Johnson-Cook parameters for steel",
            "GISSMO damage calibration for aluminum alloy",
            "Molecular dynamics of fracture in titanium alloy",
            "Fatigue performance and damage in steel",
            "Creep damage in nickel alloy",
            "Lattice structure plasticity in Ti-6Al-4V",
        ]
        for title in titles:
            with self.subTest(title=title):
                self.assertTrue(self.kept(title))

    def test_unrelated_rejection(self) -> None:
        for title in [
            "Gene expression and protein analysis",
            "Machine learning prediction of metal commodity prices",
            "LS-OPT optimization of titanium alloy manufacturing process parameters",
            "Ti-6Al-4V surface roughness optimization",
            "Steel corrosion evaluation",
            "Bayesian calibration of financial models",
            "Thermal conductivity prediction for aluminum alloy",
            "Stable diffusion image restoration",
        ]:
            with self.subTest(title=title):
                self.assertFalse(self.kept(title))
        self.assertFalse(self.kept("Steel commodity price prediction", "Engineering Fracture Mechanics"))

    def test_research_facets(self) -> None:
        for title in [
            "Ti64 cryogenic tensile fracture",
            "Ti-6Al-4V intermediate strain rate plasticity",
            "Thermomechanical coupling of titanium alloy flow stress",
            "Titanium alloy combined tension shear fracture strain",
            "Titanium alloy nonproportional loading damage evolution",
            "Ti-6Al-4V Hockett-Sherby hardening",
            "Steel post-necking hardening extrapolation",
            "Aluminium damage regularisation and mesh sensitivity",
            "Stress integration and return mapping for anisotropic plasticity",
            "Physics-informed neural network for constitutive plasticity",
            "Pulse shaping for split Hopkinson high strain rate testing",
            "High speed DIC for dynamic tension stress strain measurement",
            "PBF-LB/M Ti-6Al-4V build direction and mechanical properties",
            "Ti64 prior beta grains and fracture",
            "Ti-6Al-4V heat treatment and tensile strength",
            "X-ray computed tomography of titanium alloy void growth and damage",
            "Titanium plate ballistic limit and residual velocity",
            "Steel projectile perforation and petalling failure",
            "Titanium alloy Taylor impact model validation",
        ]:
            with self.subTest(title=title):
                self.assertTrue(self.kept(title))
        # Mentioning a peripheral method only in the abstract must not demote core research.
        work = CandidateWork(source="test", identifier="p", title="TC4 high strain rate ductile fracture",
                             abstract="Experiments are compared with molecular dynamics.")
        self.assertEqual(research_priority(work, self.settings.scoring), ("TC4核心研究", 1.0))

    def test_priority_and_ranker(self) -> None:
        # These tests exercise scoring, not the optional torch/model runtime.
        with patch.dict(sys.modules, {"src.vectorizer": SimpleNamespace(TextVectorizer=object)}):
            from src.score_rank import WorkRanker

        examples = [
            ("TC4 ductile fracture", "TC4核心研究", 1.0),
            ("Steel LS-OPT constitutive model calibration", "跨金属方法参考", 0.95),
            ("Titanium alloy crystal plasticity", "机制参考", 0.85),
            ("Steel fatigue performance damage", "外围方法参考", 0.70),
        ]
        candidates = [CandidateWork(source="test", identifier=str(i), title=row[0])
                      for i, row in enumerate(examples)]
        ranker = object.__new__(WorkRanker)
        ranker.settings = self.settings
        ranker.vectorizer = SimpleNamespace(encode=lambda texts: np.zeros((len(texts), 2)))
        ranker.index = SimpleNamespace(search=lambda vectors, top_k: (np.ones((len(vectors), 1)), None))
        ranker.journal_metrics = {}
        ranked = {work.identifier: work for work in ranker.rank(candidates)}
        for i, (title, label, multiplier) in enumerate(examples):
            with self.subTest(title=title):
                self.assertEqual(research_priority(candidates[i], self.settings.scoring), (label, multiplier))
                work = ranked[str(i)]
                self.assertAlmostEqual(work.score, work.extra['base_score'] * multiplier)
                self.assertEqual(work.extra['research_priority'], label)
                self.assertEqual(candidates[i].extra, {})
        self.assertGreater(ranked['0'].score, ranked['2'].score)
        self.assertGreater(ranked['2'].score, ranked['3'].score)

    def test_report_and_rss_priority(self) -> None:
        from src.report_html import render_html
        from src.rss_writer import write_rss
        from xml.etree import ElementTree

        work = RankedWork(
            source="test", identifier="1", title="Steel fracture <test>",
            score=0.6, similarity=0.8, recency_score=1.0, metric_score=0.0,
            author_bonus=0.0, venue_bonus=0.0, label="consider",
            extra={"research_priority": "跨金属方法参考"},
        )
        with patch.object(Path, "mkdir"), patch.object(Path, "write_text") as write:
            render_html([work], BASE / "reports" / "test-unused.html")
        html = write.call_args.args[0]
        self.assertIn("跨金属方法参考", html)
        self.assertIn("&lt;test&gt;", html)
        with patch.object(Path, "mkdir"), patch.object(ElementTree.ElementTree, "write", autospec=True) as write:
            write_rss([work], BASE / "reports" / "test-unused.xml")
        root = write.call_args.args[0].getroot()
        self.assertIn("跨金属方法参考", root.findtext("channel/item/description"))


if __name__ == "__main__":
    unittest.main()
