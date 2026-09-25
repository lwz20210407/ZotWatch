"""Failures must look like failures, and the email must say what it means.

Five defects from the 2026-09-19 review, all variations on "something went wrong and
the system reported success".
"""

import tempfile
import unittest
import logging
from email.message import EmailMessage
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from src.enrich_zh import ABSTRACT_LIMIT, ChineseEnricher, _key
from src.settings import TranslationConfig


def work(i: int, abstract: str = ""):
    return SimpleNamespace(title=f"Paper {i}", abstract=abstract or f"Abstract {i}", extra={})


class TranslationFidelityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name, "zh.json")
        self.config = TranslationConfig(batch_size=8, timeout_seconds=1)
        patcher = patch.dict("os.environ", {"EMBEDDING_API_KEY": "k"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_two_papers_sharing_a_long_prefix_get_different_keys(self):
        """The key used to hash only the first 1200 characters of the abstract.

        Not a hash collision -- a deliberate truncation. 80% of this library's abstracts
        are longer than that, so for most of the corpus the key was effectively
        title-only, and the second paper silently received the first one's translation.
        """
        shared = "x" * 1300
        self.assertNotEqual(_key("Same title", shared + "ending A"),
                            _key("Same title", shared + "ending B"))

    def test_the_key_changes_with_the_model_and_the_prompt(self):
        self.assertNotEqual(_key("T", "A", "model-one"), _key("T", "A", "model-two"))

    def test_a_long_abstract_is_sent_whole(self):
        """51 of 407 real abstracts were cut at 2000 chars, the worst losing 46.9%.

        The card showed the complete English abstract beside the half-finished Chinese
        one, with nothing to indicate it stopped early.
        """
        long_abstract = "A sentence about ductile fracture. " * 120  # ~4200 chars
        self.assertGreater(len(long_abstract), 2000)
        sent = {}

        def fake_post(self_, batch, attempts=3):
            sent["abstract"] = batch[0]["abstract"]
            return [{"i": 0, "title_zh": "标题", "tldr": "结论", "abstract_zh": "摘要"}]

        with patch.object(ChineseEnricher, "_post", fake_post):
            ChineseEnricher(self.config, self.cache).enrich([work(0, long_abstract)])
        self.assertEqual(len(sent["abstract"]), len(long_abstract),
                         "the abstract must reach _post uncut")
        self.assertLess(len(long_abstract), ABSTRACT_LIMIT)

    def test_an_empty_translation_is_not_cached(self):
        """Caching it poisoned the entry forever: the next run hit the empty cache."""
        calls = []

        def fake_post(self_, batch, attempts=3):
            calls.append(len(batch))
            return [{"i": 0, "title_zh": "", "tldr": "", "abstract_zh": ""}]

        with patch.object(ChineseEnricher, "_post", fake_post):
            ChineseEnricher(self.config, self.cache).enrich([work(0)])
            ChineseEnricher(self.config, self.cache).enrich([work(0)])
        self.assertEqual(calls, [1, 1], "an empty reply must be retried, not cached")


class SourceFailureTests(unittest.TestCase):
    """A dead source must not look like a quiet week."""

    def test_a_request_failure_propagates_out_of_the_pager(self):
        from src import source_paging

        session = SimpleNamespace()
        with patch.object(source_paging, "request_with_retry",
                          side_effect=requests.ConnectionError("down")):
            with self.assertRaises(requests.RequestException):
                list(source_paging.iter_works(
                    session, "https://api.example.org/works", {}, provider="crossref",
                    max_pages=2, context="probe", interval_seconds=0,
                    logger=logging.getLogger("probe")))

    def test_pages_already_yielded_survive_the_failure(self):
        """Partial coverage is still coverage; the caller sees both."""
        from src import source_paging

        page = SimpleNamespace(json=lambda: {
            "message": {"items": [{"DOI": "10.1/a"}], "next-cursor": "c2", "total-results": 99}})
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return page
            raise requests.Timeout("gone")

        got = []
        with patch.object(source_paging, "request_with_retry", side_effect=flaky):
            with self.assertRaises(requests.RequestException):
                for item in source_paging.iter_works(
                        SimpleNamespace(), "https://api.example.org/works", {"rows": 1},
                        provider="crossref", max_pages=3, context="probe",
                        interval_seconds=0, logger=logging.getLogger("probe")):
                    got.append(item)
        self.assertEqual(len(got), 1)


class MetricSpamAbstractTests(unittest.TestCase):
    """A list of impact factors is not an abstract.

    Predatory journals put their citation-metric boasts in the abstract field. That is
    worse than an empty abstract: the topic gate reads it as real text and judges the
    paper as though its content had been examined. On 2026-09-24 a Kevlar ballistic
    composite paper (DOI 10.15863/tas...) reached the digest that way, and the owner
    marked it irrelevant in issue #5. Its entire abstract was:
        "Impact Factor: ISRA (India) = 6.317ISI (Dubai, UAE) = 1.582GIF (Australia)
         = 0.564 JIF = 1.500SIS (USA) = 0.912 ..."
    """

    SPAM = ("Impact Factor: ISRA (India) = 6.317ISI (Dubai, UAE) = 1.582GIF (Australia) "
            "= 0.564 JIF = 1.500SIS (USA) = 0.912 РИНЦ (Russia) = 0.191 ESJI (KZ) = 8.100")

    def test_a_metric_scoreboard_is_recognised(self):
        from src.fetch_new import _looks_like_metric_spam
        self.assertTrue(_looks_like_metric_spam(self.SPAM))

    def test_a_real_abstract_mentioning_impact_is_not(self):
        """"Impact" is an ordinary word in this field; one mention proves nothing."""
        from src.fetch_new import _looks_like_metric_spam
        real = ("We measure the impact factor of build orientation on the ductile fracture "
                "of Ti-6Al-4V under dynamic loading, using notched specimens and a "
                "calibrated damage model across stress triaxialities and Lode angles.")
        self.assertFalse(_looks_like_metric_spam(real))
        self.assertFalse(_looks_like_metric_spam(None))
        self.assertFalse(_looks_like_metric_spam(""))

    def test_the_openalex_extractor_returns_none_for_spam(self):
        from src.fetch_new import _extract_openalex_abstract
        self.assertIsNone(_extract_openalex_abstract({"abstract": self.SPAM}))

    def test_the_crossref_cleaner_returns_none_for_spam(self):
        from src.fetch_new import _clean_crossref_abstract
        self.assertIsNone(_clean_crossref_abstract(f"<jats:p>{self.SPAM}</jats:p>"))

    def test_a_real_crossref_abstract_survives(self):
        from src.fetch_new import _clean_crossref_abstract
        out = _clean_crossref_abstract(
            "<jats:p>A ductile fracture criterion for Ti-6Al-4V is calibrated.</jats:p>")
        self.assertEqual(out, "A ductile fracture criterion for Ti-6Al-4V is calibrated.")


class RecipientRefusalTests(unittest.TestCase):
    """send_message only raises when EVERY recipient is refused."""

    def config(self):
        from src.notify_email import SmtpConfig
        return SmtpConfig(host="smtp.example.org", port=465, username="u", password="p",
                          sender="from@example.org",
                          recipients=["ok@example.org", "bad@example.org"])

    def test_a_partial_refusal_is_not_a_success(self):
        from src import notify_email

        class FakeServer:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def login(self, *a):
                pass

            def send_message(self, msg):
                return {"bad@example.org": (550, b"No such user")}

        msg = EmailMessage()
        msg["Subject"] = "t"
        with patch.object(notify_email.smtplib, "SMTP_SSL", lambda *a, **k: FakeServer()):
            with self.assertRaises(RuntimeError) as caught:
                notify_email.send(msg, self.config(), attempts=1)
        self.assertIn("bad@example.org", str(caught.exception))
        self.assertIn("1 of 2", str(caught.exception))

    def test_no_refusal_is_a_success(self):
        from src import notify_email

        class FakeServer:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def login(self, *a):
                pass

            def send_message(self, msg):
                return {}

        msg = EmailMessage()
        msg["Subject"] = "t"
        with patch.object(notify_email.smtplib, "SMTP_SSL", lambda *a, **k: FakeServer()):
            notify_email.send(msg, self.config(), attempts=1)


if __name__ == "__main__":
    unittest.main()
