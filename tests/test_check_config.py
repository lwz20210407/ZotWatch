"""The config validator must catch the two mistakes that only showed up in a live run."""

from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
import unittest

import yaml

from src.check_config import check

BASE = Path(__file__).resolve().parents[1]


class CheckConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name, "repo")
        copytree(BASE / "config", self.root / "config")

    def _scoring(self) -> dict:
        return yaml.safe_load((self.root / "config" / "scoring.yaml").read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        (self.root / "config" / "scoring.yaml").write_text(
            yaml.safe_dump(data, allow_unicode=True), encoding="utf-8"
        )

    def test_shipped_config_is_valid(self) -> None:
        self.assertEqual(check(BASE), [])

    def test_detects_weights_that_do_not_sum_to_one(self) -> None:
        data = self._scoring()
        data["weights"]["similarity"] = 0.9
        self._write(data)
        problems = check(self.root)
        self.assertTrue(any("sum to" in problem for problem in problems), problems)

    def test_rule_order_no_longer_changes_the_outcome(self) -> None:
        """Reordering the table must be a no-op now, including the original defect.

        Priority used to be "first match wins", so putting the x0.70 peripheral rule at
        the top demoted core work by 30% whenever a title mentioned a coating. The check
        that policed this required non-increasing multipliers -- a workaround that also
        failed, passing happily while a demotion rule sat below a x1.0 rule that always
        claimed the same papers first, so it never fired at all. Resolution is now
        explicit (authoritative wins, else lowest multiplier), so order is free.
        """
        data = self._scoring()
        rules = data["research_priorities"]
        data["research_priorities"] = [rules[-1]] + rules[:-1]
        self._write(data)
        self.assertEqual(check(self.root), [])

    def test_detects_a_rule_that_can_never_apply(self) -> None:
        """Same groups, higher multiplier: the lowest-wins policy always beats it."""
        data = self._scoring()
        first = data["research_priorities"][0]
        clone = dict(first)
        clone["name"] = "永远轮不到"
        clone["multiplier"] = min(1.0, float(first["multiplier"]))
        clone.pop("authoritative", None)
        first_copy = dict(first)
        first_copy["multiplier"] = 0.5
        first_copy["name"] = "更低的同规则"
        first_copy.pop("authoritative", None)
        data["research_priorities"] = [clone, first_copy] + data["research_priorities"][1:]
        self._write(data)
        problems = check(self.root)
        self.assertTrue(any("never apply" in problem for problem in problems), problems)

    def test_detects_one_label_meaning_two_different_multipliers(self) -> None:
        data = self._scoring()
        rules = data["research_priorities"]
        twin = dict(rules[0])
        twin["multiplier"] = 0.5
        twin.pop("authoritative", None)
        data["research_priorities"] = rules + [twin]
        self._write(data)
        problems = check(self.root)
        self.assertTrue(any("two different things" in problem for problem in problems), problems)

    def test_detects_inverted_thresholds(self) -> None:
        data = self._scoring()
        data["thresholds"] = {"must_read": 0.3, "consider": 0.8}
        self._write(data)
        self.assertTrue(any("must_read" in problem for problem in check(self.root)))


if __name__ == "__main__":
    unittest.main()
