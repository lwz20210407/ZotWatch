"""The page must say when the library profile is old.

The profile is rebuilt on the owner's PC by a scheduled task. If the task stops, nothing
fails: the digest keeps running on an ageing library, re-recommending papers already
added since. The only place that can notice is the page itself.
"""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.report_html import profile_age, render_html

NOW = datetime(2026, 10, 5, 8, 0, tzinfo=timezone.utc)


class ProfileAgeTests(unittest.TestCase):
    def test_age_in_days_and_beijing_date(self):
        age = profile_age("2026-09-19T07:15:50.364554+00:00", now=NOW)
        self.assertEqual((age["date"], age["days"], age["stale"]), ("2026-09-19", 16, True))

    def test_a_fresh_profile_is_not_flagged(self):
        self.assertFalse(profile_age("2026-10-01T05:00:00+00:00", now=NOW)["stale"])

    def test_garbage_is_ignored_not_fatal(self):
        self.assertIsNone(profile_age("not a date", now=NOW))
        self.assertIsNone(profile_age(None, now=NOW))

    def render(self, generated):
        out = Path(tempfile.mkdtemp()) / "r.html"
        render_html([], out, library_size="4391 篇", profile_generated_at=generated)
        return out.read_text(encoding="utf-8")

    def test_the_page_shows_the_date_and_warns_when_stale(self):
        html = self.render("2020-01-01T00:00:00+00:00")
        self.assertIn("生成于 2020-01-01", html)
        self.assertIn("本机每周四的画像更新任务可能没有运行", html)

    def test_no_warning_for_a_fresh_profile(self):
        html = self.render(datetime.now(timezone.utc).isoformat())
        self.assertIn("（0 天前）", html)
        self.assertNotIn("可能没有运行", html)


if __name__ == "__main__":
    unittest.main()
