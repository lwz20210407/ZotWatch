import logging
from datetime import datetime, timezone
import unittest
from unittest.mock import Mock, patch

from src.fetch_new import CandidateFetcher
from src.http_utils import request_with_retry
from src.settings import load_settings
from src.source_paging import crossref_publication_date, iter_works


class SourcePagingTests(unittest.TestCase):
    def response(self, body: dict) -> Mock:
        response = Mock()
        response.json.return_value = body
        return response

    def test_crossref_cursor_and_immutability(self) -> None:
        params = {"rows": 2}
        responses = [self.response({"message": {"items": [{"DOI": "a"}, {"DOI": "b"}], "next-cursor": "next", "total-results": 3}}),
                     self.response({"message": {"items": [{"DOI": "c"}], "next-cursor": "end", "total-results": 3}})]
        with patch("src.source_paging.request_with_retry", side_effect=responses) as request:
            items = list(iter_works(Mock(), "url", params, provider="crossref", max_pages=5,
                                    interval_seconds=0, logger=logging.getLogger(__name__), context="test"))
        self.assertEqual([item["DOI"] for item in items], ["a", "b", "c"])
        self.assertEqual(request.call_args_list[1].kwargs["params"]["cursor"], "next")
        self.assertEqual(params, {"rows": 2})

    def test_openalex_cap_warns(self) -> None:
        response = self.response({"results": [{"id": "a"}], "meta": {"count": 10, "next_cursor": "next"}})
        logger = Mock()
        with patch("src.source_paging.request_with_retry", return_value=response):
            result = list(iter_works(Mock(), "url", {"per-page": 1}, provider="openalex", max_pages=1,
                                     interval_seconds=0, logger=logger, context="test"))
        self.assertEqual(len(result), 1)
        self.assertIn("cap", logger.warning.call_args.args[0])

    def test_missing_cursor_warns(self) -> None:
        response = self.response({"message": {"items": [{"DOI": "a"}], "total-results": 9}})
        logger = Mock()
        with patch("src.source_paging.request_with_retry", return_value=response):
            list(iter_works(Mock(), "url", {"rows": 1}, provider="crossref", max_pages=2,
                            interval_seconds=0, logger=logger, context="test"))
        self.assertIn("cursor", logger.warning.call_args.args[0])

    def test_publication_date_not_record_creation(self) -> None:
        item = {"created": {"date-time": "2026-09-17T00:00:00Z"},
                "published-online": {"date-parts": [[2026, 8, 20]]},
                "published-print": {"date-parts": [[2027, 1]]}}
        self.assertEqual(crossref_publication_date(item), datetime(2026, 8, 20, tzinfo=timezone.utc))
        self.assertIsNone(crossref_publication_date({"created": item["created"]}))

    def test_venue_uses_issn_and_actual_name(self) -> None:
        fetcher = object.__new__(CandidateFetcher)
        fetcher.settings = load_settings(".")
        fetcher.settings.sources.tracked_venues = ["Materials Science and Engineering: A"]
        fetcher.top_venues = []
        fetcher.session = Mock()
        item = {"title": ["Fracture of TC4"], "DOI": "test", "container-title": ["Materials Science and Engineering A"],
                "published": {"date-parts": [[2026, 9, 1]]}}
        with patch("src.fetch_new.iter_works", return_value=iter([item])) as pages:
            results = fetcher._fetch_crossref_top_venues(datetime(2026, 8, 18))
        self.assertIn("issn:0921-5093", pages.call_args.args[2]["filter"])
        self.assertEqual(results[0].venue, "Materials Science and Engineering A")
        self.assertEqual(results[0].published.day, 1)

    def test_retry_after_is_respected(self) -> None:
        limited = Mock(status_code=429, headers={"Retry-After": "8"})
        success = Mock(status_code=200)
        session = Mock()
        session.request.side_effect = [limited, success]
        with patch("src.http_utils.time.sleep") as sleep:
            result = request_with_retry(session, "GET", "url", logger=Mock(), context="test")
        self.assertIs(result, success)
        sleep.assert_called_once_with(8.0)

    def test_crossref_query_relevance_and_date(self) -> None:
        fetcher = object.__new__(CandidateFetcher)
        fetcher.settings = load_settings(".")
        fetcher.settings.sources.queries = ["Ti-6Al-4V fracture"]
        fetcher.session = Mock()
        item = {"title": ["TC4 fracture"], "DOI": "test", "published": {"date-parts": [[2026, 9, 5]]},
                "created": {"date-time": "2026-09-17T00:00:00Z"}}
        with patch("src.fetch_new.iter_works", return_value=iter([item])) as pages:
            results = fetcher._fetch_crossref(datetime(2026, 8, 18))
        self.assertEqual(pages.call_args.args[2]["sort"], "relevance")
        self.assertEqual(results[0].published.day, 5)


if __name__ == "__main__":
    unittest.main()
