"""Web report rendering: date precision, direction colour, layout helpers."""

from datetime import datetime, timezone
import unittest

from src.models import RankedWork
from src.report_html import (direction_hue, published_label, scatter,
                             split_abstract, venue_counts)
from src.source_paging import crossref_publication_date_precise


def work(**kwargs) -> RankedWork:
    base = dict(source="openalex", identifier="W1", title="A paper",
                published=datetime(2026, 9, 14, tzinfo=timezone.utc),
                score=0.7, similarity=0.6, recency_score=0.8, metric_score=0.1,
                author_bonus=0.0, venue_bonus=0.0, label="consider", extra={})
    base.update(kwargs)
    return RankedWork(**base)


class DatePrecisionTests(unittest.TestCase):
    """Crossref may give only a year; the report must not invent a day."""

    def _crossref(self, parts):
        return crossref_publication_date_precise({"published": {"date-parts": [parts]}})

    def test_precision_follows_the_source(self) -> None:
        self.assertEqual(self._crossref([2026, 9, 14])[1], "day")
        self.assertEqual(self._crossref([2026, 9])[1], "month")
        self.assertEqual(self._crossref([2026])[1], "year")

    def test_missing_components_are_still_filled_for_sorting(self) -> None:
        date, _ = self._crossref([2026])
        self.assertEqual(date.date().isoformat(), "2026-01-01")

    def test_label_never_claims_more_precision_than_the_source(self) -> None:
        self.assertEqual(published_label(work(extra={"date_precision": "year"})), "2026")
        self.assertEqual(published_label(work(extra={"date_precision": "month"})), "2026-09")
        self.assertEqual(published_label(work(extra={"date_precision": "day"})), "2026-09-14")

    def test_unmarked_dates_default_to_full(self) -> None:
        """OpenAlex publication_date is day-precise, so an unset flag means day."""
        self.assertEqual(published_label(work()), "2026-09-14")

    def test_no_date_renders_nothing(self) -> None:
        self.assertEqual(published_label(work(published=None)), "")

    def test_no_crossref_date_is_reported_as_absent(self) -> None:
        self.assertEqual(crossref_publication_date_precise({}), (None, ""))


class DirectionColourTests(unittest.TestCase):
    def test_same_direction_always_gets_the_same_hue(self) -> None:
        a = direction_hue("延性断裂与损伤演化")
        b = direction_hue("延性断裂与损伤演化")
        self.assertEqual(a, b)

    DIRECTIONS = ["延性断裂与损伤演化", "参数反演与硬化外推", "增材组织—缺陷—失效",
                  "数值实现与损伤正则化", "温度—应变率耦合", "冲击与结构验证",
                  "应力状态依赖塑性"]

    def test_ordered_assignment_gives_every_direction_its_own_colour(self) -> None:
        """The report always passes `order`, which is what makes this a guarantee.

        Hashing the name alone collided: seven directions over an eight-colour
        palette produced only five distinct hues, and two directions sharing a
        colour defeats the purpose of colouring them at all.
        """
        order = {name: i for i, name in enumerate(self.DIRECTIONS)}
        hues = {direction_hue(n, order)["fg"] for n in self.DIRECTIONS}
        self.assertEqual(len(hues), len(self.DIRECTIONS))

    def test_hash_fallback_is_deterministic(self) -> None:
        """Without `order` colours may collide, but must never change run to run."""
        first = [direction_hue(n)["fg"] for n in self.DIRECTIONS]
        second = [direction_hue(n)["fg"] for n in self.DIRECTIONS]
        self.assertEqual(first, second)

    def test_unknown_direction_falls_back(self) -> None:
        self.assertIn("fg", direction_hue(""))


class LayoutHelperTests(unittest.TestCase):
    def test_abstract_splits_on_a_word_boundary(self) -> None:
        lead, rest = split_abstract("word " * 200)
        self.assertFalse(lead.endswith("wor"), "must not cut mid-word")
        self.assertTrue(lead.endswith("word"))
        self.assertTrue(rest)

    def test_short_abstract_is_not_split(self) -> None:
        self.assertEqual(split_abstract("short"), ("short", ""))

    def test_scatter_maps_recency_and_relevance(self) -> None:
        recent = work(identifier="new", published=datetime.now(timezone.utc), score=0.9)
        chart = scatter([recent, work(identifier="old", score=0.5)], {})
        self.assertEqual(len(chart["points"]), 2)
        newest, oldest = chart["points"]
        # More recent sits further right, higher score sits higher (smaller y).
        self.assertGreater(newest["x"], oldest["x"])
        self.assertLess(newest["y"], oldest["y"])

    def test_scatter_without_dates_is_empty(self) -> None:
        self.assertEqual(scatter([work(published=None)], {}), {})

    def test_venue_counts_are_ranked(self) -> None:
        works = [work(identifier=str(i), venue=v) for i, v in
                 enumerate(["IJIE", "IJIE", "Acta", "IJIE", "Acta", "Strain"])]
        self.assertEqual(venue_counts(works)[:2], [("IJIE", 3), ("Acta", 2)])


if __name__ == "__main__":
    unittest.main()
