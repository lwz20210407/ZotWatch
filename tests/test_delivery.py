"""Regression cases for the RSS feed and the email digest."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

from src.models import RankedWork
from src.notify_email import SmtpConfig, build_message, latest_report
from src.rss_writer import site_url, write_rss


def _work(title: str, *, published: datetime, identifier: str = "W123") -> RankedWork:
    return RankedWork(
        source="openalex",
        identifier=identifier,
        title=title,
        url="https://example.org/paper",
        published=published,
        score=0.8,
        similarity=0.7,
        recency_score=0.5,
        metric_score=0.1,
        author_bonus=0.0,
        venue_bonus=0.0,
        label="must_read",
    )


class RssTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name, "feed.xml")
        self.addCleanup(self.tmp.cleanup)

    def _channel(self) -> ET.Element:
        return ET.parse(self.path).getroot().find("channel")

    def test_channel_link_is_not_the_placeholder(self) -> None:
        """write_rss was called without `link=`, so every reader showed example.com."""
        with patch.dict("os.environ", {"GITHUB_REPOSITORY": "someone/ZotWatch"}):
            write_rss([], self.path)
        link = self._channel().findtext("link")
        self.assertNotIn("example.com", link)
        self.assertEqual(link, "https://someone.github.io/ZotWatch/")

    def test_pubdate_is_discovery_date_not_publication_date(self) -> None:
        """A 2014 paper in the classics channel must not enter the feed dated 2014.

        Readers sort and mark-as-read by pubDate, which buried the entire
        "经典文献补漏" section.
        """
        old = datetime(2014, 3, 1, tzinfo=timezone.utc)
        write_rss([_work("A classic", published=old)], self.path)
        item = self._channel().find("item")
        self.assertNotIn("2014", item.findtext("pubDate"))
        # The real publication date is still exposed, just not as the sort key.
        self.assertEqual(item.findtext("{http://purl.org/dc/elements/1.1/}date"), "2014-03-01")

    def test_guid_is_marked_as_a_non_permalink(self) -> None:
        write_rss([_work("x", published=datetime.now(timezone.utc))], self.path)
        self.assertEqual(self._channel().find("item/guid").get("isPermaLink"), "false")

    def test_site_url_falls_back_without_repository_env(self) -> None:
        with patch.dict("os.environ", {"GITHUB_REPOSITORY": "", "ZOTWATCH_SITE_URL": "https://x.test/"}):
            self.assertEqual(site_url(), "https://x.test/")


REPORT = """<!DOCTYPE html><html><body>
<h2>综合推荐</h2>
<article><h2>1. <a href="https://example.org/a">First paper &amp; notes</a></h2></article>
<article><h2>2. <a href="https://example.org/b">Second paper</a></h2></article>
</body></html>"""


class EmailTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.config = SmtpConfig(
            host="smtp.test", port=587, username="u@test", password="p",
            sender="u@test", recipients=["a@test", "b@test"],
        )

    def test_latest_report_uses_the_filename_date(self) -> None:
        """`cp -r` rewrites mtimes, so mtime ordering picked an arbitrary report."""
        for name in ("report-20260422.html", "report-20260521.html", "report-20260114.html"):
            (self.dir / name).write_text("x", encoding="utf-8")
        # Make the oldest report the most recently touched file.
        (self.dir / "report-20260114.html").touch()
        self.assertEqual(latest_report(self.dir).name, "report-20260521.html")

    def test_digest_is_the_body_not_an_attachment(self) -> None:
        """Exchange and mobile clients routinely refuse to preview HTML attachments."""
        report = self.dir / "report-20260521.html"
        report.write_text(REPORT, encoding="utf-8")
        msg = build_message(self.config, report_path=report, feed_path=None)
        html_part = msg.get_body(preferencelist=("html",))
        self.assertIsNotNone(html_part)
        self.assertIn("First paper", html_part.get_content())
        attachments = [part.get_filename() for part in msg.iter_attachments()]
        self.assertNotIn(report.name, attachments)

    def test_plain_text_part_lists_the_top_papers(self) -> None:
        report = self.dir / "report-20260521.html"
        report.write_text(REPORT, encoding="utf-8")
        msg = build_message(self.config, report_path=report, feed_path=None)
        text = msg.get_body(preferencelist=("plain",)).get_content()
        self.assertIn("First paper & notes", text)
        self.assertIn("https://example.org/b", text)

    def test_headers_thread_the_digests_and_address_all_recipients(self) -> None:
        msg = build_message(self.config, report_path=None, feed_path=None)
        self.assertEqual(msg["To"], "a@test, b@test")
        self.assertTrue(msg["Message-ID"])
        self.assertEqual(msg["References"], msg["In-Reply-To"])

    def test_recipients_are_parsed_from_a_delimited_list(self) -> None:
        env = {
            "SMTP_HOST": "h", "SMTP_USERNAME": "u@test", "SMTP_PASSWORD": "p",
            "EMAIL_TO": "a@test, b@test;c@test",
        }
        with patch.dict("os.environ", env, clear=False):
            self.assertEqual(list(SmtpConfig.from_env().recipients), ["a@test", "b@test", "c@test"])

    def test_missing_report_still_produces_a_message(self) -> None:
        msg = build_message(self.config, report_path=None, feed_path=None)
        self.assertIn("本轮无通过筛选", msg.get_body(preferencelist=("plain",)).get_content())


if __name__ == "__main__":
    unittest.main()
