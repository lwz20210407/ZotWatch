"""AMiner abstract fill: the matching must err towards missing, and failures must stop it.

Measured on 2026-10-05 against the dry-run page: 9 bare cards, 6 filled (all by DOI),
2 matched records without an abstract, 1 not indexed, 0.08 yuan.
"""

import unittest

from src.abstract_fill import fill_abstracts, match, needs_abstract
from src.models import CandidateWork

FULL = "A full abstract about Lode-dependent ductile fracture of Ti-6Al-4V sheets. " * 5


def work(title, doi=None, abstract=None):
    return CandidateWork(source="t", identifier=title, title=title, doi=doi, abstract=abstract)


class MatchTests(unittest.TestCase):
    def test_doi_agreement_wins(self):
        row, how = match(work("Anything", doi="https://doi.org/10.1016/J.X.1"),
                         [{"id": "a", "title": "Different", "doi": "10.1016/j.x.1"}])
        self.assertEqual((row["id"], how), ("a", "doi"))

    def test_two_different_dois_are_two_papers_however_alike_the_titles(self):
        row, _ = match(work("Fracture of Ti-6Al-4V sheets", doi="10.1/a"),
                       [{"id": "b", "title": "Fracture of Ti-6Al-4V sheets", "doi": "10.1/b"}])
        self.assertIsNone(row)

    def test_without_a_doi_only_a_near_identical_title_counts(self):
        rows = [{"id": "c", "title": "Fracture of Ti-6Al-4V sheets at elevated temperature"}]
        self.assertEqual(match(work("Fracture of Ti-6Al-4V Sheets at Elevated Temperature."), rows)[1], "title")
        self.assertIsNone(match(work("Fracture of Ti-6Al-4V sheets at room temperature"), rows)[0])


class FillTests(unittest.TestCase):
    def test_fills_only_bare_papers_and_marks_the_provenance(self):
        bare, full = work("Bare", doi="10.1/bare"), work("Has one", doi="10.1/full", abstract=FULL)
        report = fill_abstracts([bare, full], search=lambda t: [{"id": "x", "doi": "10.1/bare"}],
                                detail=lambda i: {"data": [{"abstract": FULL}]})
        self.assertEqual(bare.abstract, FULL.strip())
        self.assertEqual(bare.extra["abstract_filled_from"], "AMiner")
        self.assertNotIn("abstract_filled_from", full.extra)
        self.assertEqual((report["missing"], report["filled"]), (1, 1))

    def test_a_record_without_an_abstract_leaves_the_card_alone(self):
        bare = work("Bare", doi="10.1/bare")
        report = fill_abstracts([bare], search=lambda t: [{"id": "x", "doi": "10.1/bare"}],
                                detail=lambda i: {"data": {"abstract": ""}})
        self.assertIsNone(bare.abstract)
        self.assertEqual(report["items"][0]["result"], "matched_without_abstract")

    def test_a_network_failure_stops_the_remaining_calls(self):
        """From the US runner AMiner often fails; ten slow failures would add minutes."""
        calls = []

        def search(title):
            calls.append(title)
            raise RuntimeError("network_failure")
        report = fill_abstracts([work(f"P{i}", doi=f"10.1/{i}") for i in range(5)], search=search,
                                detail=lambda i: {})
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["stopped"], "network_failure")
        self.assertEqual(report["missing"], 5)

    def test_the_budget_stop_is_respected(self):
        detail_calls = []

        def detail(ident):
            detail_calls.append(ident)
            raise RuntimeError("budget_exhausted")
        fill_abstracts([work(f"P{i}", doi=f"10.1/{i}") for i in range(3)],
                       search=lambda t: [{"id": t, "doi": "10.1/" + t[1:]}], detail=detail)
        self.assertEqual(len(detail_calls), 1)

    def test_the_limit_caps_attempts(self):
        report = fill_abstracts([work(f"P{i}") for i in range(5)], search=lambda t: [],
                                detail=lambda i: {}, limit=2)
        self.assertEqual((report["missing"], report["attempted"]), (5, 2))

    def test_a_short_abstract_counts_as_missing(self):
        self.assertTrue(needs_abstract(work("x", abstract="Too short.")))
        self.assertFalse(needs_abstract(work("x", abstract=FULL)))


class OffByDefaultTests(unittest.TestCase):
    def test_the_paid_fill_is_off_unless_configured(self):
        from pathlib import Path
        from src.settings import load_settings
        settings = load_settings(Path(__file__).resolve().parents[1])
        self.assertFalse(settings.research.aminer_abstract_fill)
        self.assertLessEqual(settings.research.aminer_abstract_budget_yuan, 1.0)


if __name__ == "__main__":
    unittest.main()
