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

    def test_detects_a_demoting_rule_placed_above_a_core_rule(self) -> None:
        data = self._scoring()
        rules = data["research_priorities"]
        # Put the 0.70 peripheral rule back at the top, the original defect.
        data["research_priorities"] = [rules[-1]] + rules[:-1]
        self._write(data)
        problems = check(self.root)
        self.assertTrue(any("shadows it" in problem for problem in problems), problems)

    def test_detects_inverted_thresholds(self) -> None:
        data = self._scoring()
        data["thresholds"] = {"must_read": 0.3, "consider": 0.8}
        self._write(data)
        self.assertTrue(any("must_read" in problem for problem in check(self.root)))


if __name__ == "__main__":
    unittest.main()
