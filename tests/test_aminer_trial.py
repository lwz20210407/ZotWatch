"""Shadow mode must be able to graduate.

Until 2026-10-05 it could not. The AMiner-vs-baseline comparison was overwritten every
run and never committed, and the labelling CSV beside it was never filled in, so
export_shadow reported human_labels: 0 every week and the decision to go live had
nothing to stand on. These tests pin the loop that replaces it: AMiner-only papers are
shown with the ordinary feedback buttons, recorded per week, and scored across weeks.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.aminer_shadow import persist_week, scoreboard, trial_candidates
from src.models import CandidateWork


def work(title, doi="", aminer_id=""):
    extra = {"aminer_id": aminer_id} if aminer_id else {}
    return CandidateWork(source="t", identifier=title, title=title, doi=doi or None, extra=extra)


def entry(rating, doi="", work_id=""):
    return SimpleNamespace(rating=rating, doi=doi, work_id=work_id)


class TrialSelectionTests(unittest.TestCase):
    def test_only_papers_the_baseline_missed_are_offered(self):
        shared = work("Shared paper", doi="10.1/shared")
        cohorts = {"baseline": [shared],
                   "aminer": [shared, work("AMiner only", doi="10.1/new")]}
        got = trial_candidates(cohorts, keep_unseen=lambda ws: ws)
        self.assertEqual([w.title for w in got], ["AMiner only"])

    def test_papers_the_owner_has_already_seen_are_not_offered(self):
        """The shadow cohort is captured before library and history filters."""
        seen = work("Already in the library", doi="10.1/lib")
        fresh = work("Genuinely new", doi="10.1/fresh")
        got = trial_candidates({"baseline": [], "aminer": [seen, fresh]},
                               keep_unseen=lambda ws: [w for w in ws if w is not seen])
        self.assertEqual([w.title for w in got], ["Genuinely new"])

    def test_duplicates_within_aminer_collapse_and_the_list_is_capped(self):
        dup = [work(f"P{i}", doi=f"10.1/{i % 3}") for i in range(9)]
        got = trial_candidates({"baseline": [], "aminer": dup}, keep_unseen=lambda ws: ws, limit=2)
        self.assertEqual(len(got), 2)


class ScoreboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_weeks_accumulate_instead_of_being_overwritten(self):
        persist_week([work("A", doi="10.1/a")], self.dir, "2026-W41")
        persist_week([work("B", doi="10.1/b")], self.dir, "2026-W42")
        board = scoreboard(self.dir, [])
        self.assertEqual(board["weeks"], 2)
        self.assertEqual(board["aminer_only_total"], 2)

    def test_ordinary_feedback_clicks_count_as_labels(self):
        persist_week([work("A", doi="10.1/a"), work("B", doi="10.1/b"),
                      work("C", aminer_id="abc123")], self.dir, "2026-W41")
        board = scoreboard(self.dir, [entry("direct", doi="10.1/a"),
                                      entry("irrelevant", doi="10.1/b"),
                                      entry("transferable", work_id="aminer:abc123")])
        self.assertEqual((board["useful"], board["irrelevant"], board["labelled"]), (2, 1, 3))

    def test_no_verdict_on_a_handful_of_clicks(self):
        persist_week([work("A", doi="10.1/a")], self.dir, "2026-W41")
        board = scoreboard(self.dir, [entry("direct", doi="10.1/a")], min_labels=20)
        self.assertIn("证据不足", board["verdict"])

    def test_verdict_follows_the_configured_ratio(self):
        works = [work(f"P{i}", doi=f"10.1/{i}") for i in range(10)]
        persist_week(works, self.dir, "2026-W41")
        useful = [entry("direct", doi=f"10.1/{i}") for i in range(4)]
        noise = [entry("irrelevant", doi=f"10.1/{i}") for i in range(4, 10)]
        self.assertIn("转为 live", scoreboard(self.dir, useful + noise,
                                              min_labels=10, graduate_ratio=0.3)["verdict"])
        self.assertIn("维持影子", scoreboard(self.dir, useful + noise,
                                            min_labels=10, graduate_ratio=0.5)["verdict"])

    def test_a_corrupt_week_file_does_not_break_the_board(self):
        (self.dir / "2026-W40.json").write_text("{not json", encoding="utf-8")
        persist_week([work("A", doi="10.1/a")], self.dir, "2026-W41")
        self.assertEqual(scoreboard(self.dir, [])["aminer_only_total"], 1)


class TrialRenderTests(unittest.TestCase):
    """The section must render, carry the feedback buttons, and stay out of the email."""

    def render(self, trial, diagnostics=None):
        from src.report_html import render_html
        out = Path(tempfile.mkdtemp()) / "report.html"
        render_html([], out, aminer_trial_works=trial, diagnostics=diagnostics or {})
        return out.read_text(encoding="utf-8")

    def test_section_shows_titles_buttons_and_scoreboard(self):
        paper = work("A paper only AMiner found", doi="10.1016/j.x.2026.0001")
        paper.extra["feedback_links"] = [
            {"name": "直接有用", "url": "https://github.com/o/r/issues/new?useful"},
            {"name": "不相关", "url": "https://github.com/o/r/issues/new?noise"},
            {"name": "稍后看", "url": "https://github.com/o/r/issues/new?later"},
        ]
        html = self.render([paper], {"aminer_trial": {
            "weeks": 3, "aminer_only_total": 17, "labelled": 5, "useful": 2,
            "irrelevant": 3, "verdict": "证据不足：已标注 5 篇，至少 20 篇再判断"}})
        self.assertIn("AMiner 试运行", html)
        self.assertIn("A paper only AMiner found", html)
        self.assertIn("issues/new?useful", html)
        self.assertIn("issues/new?noise", html)
        self.assertNotIn("issues/new?later", html, "only the two labelling buttons belong here")
        self.assertIn("证据不足", html)

    def test_no_section_when_there_is_nothing_to_label(self):
        self.assertNotIn("AMiner 试运行", self.render([]))

    def test_the_email_digest_never_carries_the_trial(self):
        from src.digest_email import render_digest
        self.assertNotIn("AMiner 试运行", render_digest([]))


if __name__ == "__main__":
    unittest.main()
