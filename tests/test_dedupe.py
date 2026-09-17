"""Regression cases for candidate deduplication.

The previous implementation scored titles with `token_set_ratio`, which returns
100 for a strict subset. A conference abstract already in the library therefore
suppressed the full journal article that followed it -- silently, at DEBUG level.
"""

from types import SimpleNamespace
import unittest

from src.dedupe import DedupeEngine, _closest_title
from src.models import CandidateWork, ZoteroItem


def _storage(titles, dois=(), urls=()):
    items = [
        ZoteroItem(
            key=f"k{i}",
            version=1,
            title=title,
            doi=dois[i] if i < len(dois) else None,
            url=urls[i] if i < len(urls) else None,
        )
        for i, title in enumerate(titles)
    ]
    return SimpleNamespace(iter_items=lambda: iter(items))


def _candidate(title: str, *, doi: str | None = None) -> CandidateWork:
    return CandidateWork(source="test", identifier=title, title=title, doi=doi)


class DedupeTests(unittest.TestCase):
    def test_longer_paper_survives_a_shorter_library_stub(self) -> None:
        """The silent-drop bug, stated as an assertion."""
        engine = DedupeEngine(_storage(["Dynamic tensile behavior of Ti-6Al-4V"]))
        candidate = _candidate(
            "Dynamic tensile behavior of Ti-6Al-4V at elevated temperatures and high strain rates"
        )
        self.assertEqual([w.title for w in engine.filter([candidate])], [candidate.title])

    def test_true_duplicate_is_still_removed(self) -> None:
        engine = DedupeEngine(_storage(["Dynamic tensile behavior of Ti-6Al-4V alloy"]))
        candidate = _candidate("Dynamic tensile behaviour of Ti-6Al-4V alloy")
        self.assertEqual(engine.filter([candidate]), [])

    def test_word_order_variation_is_treated_as_duplicate(self) -> None:
        engine = DedupeEngine(_storage(["Ductile fracture of titanium alloys"]))
        self.assertEqual(engine.filter([_candidate("Of titanium alloys ductile fracture")]), [])

    def test_doi_duplicate_is_removed(self) -> None:
        engine = DedupeEngine(_storage(["Something else"], dois=["10.1000/abc"]))
        self.assertEqual(engine.filter([_candidate("A new paper", doi="10.1000/ABC")]), [])

    def test_duplicates_within_the_same_batch_are_collapsed(self) -> None:
        engine = DedupeEngine(_storage([]))
        batch = [_candidate("Ductile fracture of Ti-6Al-4V"), _candidate("Ductile fracture of Ti-6Al-4V")]
        self.assertEqual(len(engine.filter(batch)), 1)

    def test_part_one_and_part_two_are_kept_apart(self) -> None:
        engine = DedupeEngine(
            _storage(["Constitutive modelling of Ti-6Al-4V Part I experiments"])
        )
        candidate = _candidate("Constitutive modelling of Ti-6Al-4V Part II simulations")
        self.assertEqual(len(engine.filter([candidate])), 1)

    def test_suppression_is_reported_at_info_level(self) -> None:
        """Title suppression is the only rule that can drop a genuinely new paper."""
        engine = DedupeEngine(_storage(["Ductile fracture of titanium alloys"]))
        with self.assertLogs("src.dedupe", level="INFO") as captured:
            engine.filter([_candidate("Ductile fracture of titanium alloy")])
        self.assertTrue(any("Title-similarity suppressed" in line for line in captured.output))

    def test_length_guard_blocks_incomparable_titles(self) -> None:
        self.assertIsNone(_closest_title("short title", ["short title " + "x" * 200], 0.9))


if __name__ == "__main__":
    unittest.main()
