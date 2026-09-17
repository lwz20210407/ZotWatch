"""Regression cases for the bounded scoring model.

Before these changes the score was a sum of terms on incompatible scales: the
semantic term was capped at its weight while log1p(citations) was unbounded, so a
well-cited but half-relevant older paper could outrank an on-topic new one -- in a
feed whose entire purpose is new work, which has no citations yet.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from src.models import CandidateWork
from src.score_rank import WorkRanker, _compute_metric, _compute_recency, _journal_quality_score
from src.settings import ScoreScales, load_settings

BASE = Path(__file__).resolve().parents[1]


def _ranker(settings, similarity: float) -> WorkRanker:
    """A ranker with the embedding and index layers stubbed to a fixed similarity."""
    ranker = object.__new__(WorkRanker)
    ranker.base_dir = BASE
    ranker.settings = settings
    ranker.profile = {"index_items": [], "problem_profiles": {}}
    ranker.journal_metrics = {}
    ranker.vectorizer = SimpleNamespace(
        text_separator="[SEP]", encode=lambda texts: np.zeros((len(texts), 2), dtype=np.float32)
    )
    ranker.index = SimpleNamespace(
        ntotal=100,
        search=lambda vectors, top_k: (np.full((len(vectors), top_k), similarity), None),
    )
    return ranker


def _work(title: str, *, days_old: int, citations: float) -> CandidateWork:
    return CandidateWork(
        source="test",
        identifier=title,
        title=title,
        published=datetime.now(timezone.utc) - timedelta(days=days_old),
        metrics={"cited_by": citations},
    )


class ScoringBoundsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = load_settings(BASE)
        cls.scales = ScoreScales()

    def test_weights_sum_to_one(self) -> None:
        total = sum(self.settings.scoring.weights.model_dump().values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_every_component_is_bounded(self) -> None:
        # Extreme inputs must not push any component outside [0, 1].
        citation, altmetric = _compute_metric(
            _work("x", days_old=0, citations=1_000_000), self.scales
        )
        self.assertLessEqual(citation, 1.0)
        self.assertLessEqual(altmetric, 1.0)
        self.assertLessEqual(_compute_recency(datetime.now(timezone.utc), self.scales), 1.0)
        quality, _ = _journal_quality_score("x", {"x": 50.0}, self.scales)
        self.assertLessEqual(quality, 1.0)

    def test_score_stays_within_unit_interval(self) -> None:
        ranker = _ranker(self.settings, similarity=1.0)
        work = _work("Ti-6Al-4V ductile fracture", days_old=0, citations=100_000)
        ranked = ranker.rank([work])[0]
        self.assertGreaterEqual(ranked.score, 0.0)
        self.assertLessEqual(ranked.score, 1.0)

    def test_citations_cannot_outrank_topical_relevance(self) -> None:
        """The defect this fix targets, stated as an assertion.

        A highly cited but less relevant older paper must not beat a fresh, more
        relevant one. Under the previous unbounded citation term it did.
        """
        relevant = _ranker(self.settings, similarity=0.90).rank(
            [_work("Ti-6Al-4V ductile fracture under dynamic loading", days_old=2, citations=0)]
        )[0]
        cited = _ranker(self.settings, similarity=0.55).rank(
            [_work("Ti-6Al-4V ductile fracture review", days_old=1100, citations=2000)]
        )[0]
        self.assertGreater(relevant.score, cited.score)

    def test_recency_has_no_cliff(self) -> None:
        """Adjacent days must differ by a small amount, not by a factor of four."""
        now = datetime.now(timezone.utc)
        day30 = _compute_recency(now - timedelta(days=30), self.scales)
        day31 = _compute_recency(now - timedelta(days=31), self.scales)
        self.assertLess(day31, day30)
        self.assertGreater(day31 / day30, 0.9)

    def test_unknown_venue_scores_below_a_strong_venue(self) -> None:
        metrics = {"international journal of impact engineering": 1.5}
        known, sjr = _journal_quality_score(
            "International Journal of Impact Engineering", metrics, self.scales
        )
        unknown, missing = _journal_quality_score("Some Unindexed Venue", metrics, self.scales)
        self.assertEqual(sjr, 1.5)
        self.assertIsNone(missing)
        self.assertGreater(known, unknown)

    def test_similarity_averages_over_neighbours(self) -> None:
        """One lucky neighbour must not decide the similarity on its own."""
        ranker = _ranker(self.settings, similarity=0.5)
        ranker.index = SimpleNamespace(
            ntotal=100,
            search=lambda vectors, top_k: (
                np.array([[0.95] + [0.10] * (top_k - 1)] * len(vectors)),
                None,
            ),
        )
        ranked = ranker.rank([_work("y", days_old=1, citations=0)])[0]
        self.assertEqual(ranked.extra["nearest_similarity"], 0.95)
        self.assertLess(ranked.similarity, 0.5)


class PriorityOrderTests(unittest.TestCase):
    def test_core_work_is_not_demoted_by_a_peripheral_keyword(self) -> None:
        """Rule order regression.

        "外围方法参考" used to be evaluated first, so a core TC4 paper whose title
        mentioned a peripheral keyword was demoted by 30% before the core rule was
        ever reached.
        """
        from src.topic_matching import research_priority

        settings = load_settings(BASE)
        work = CandidateWork(
            source="test",
            identifier="t",
            title="Ductile fracture of coated Ti-6Al-4V under dynamic loading",
            abstract="Johnson-Cook constitutive model and damage evolution.",
        )
        name, multiplier = research_priority(work, settings.scoring)
        self.assertEqual(name, "TC4核心研究")
        self.assertEqual(multiplier, 1.0)

    def test_multipliers_are_non_increasing(self) -> None:
        settings = load_settings(BASE)
        multipliers = [rule.multiplier for rule in settings.scoring.research_priorities]
        self.assertEqual(multipliers, sorted(multipliers, reverse=True))


if __name__ == "__main__":
    unittest.main()
