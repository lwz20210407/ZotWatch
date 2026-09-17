from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from pydantic import ValidationError

from src.author_watch import author_news, candidate_from_openalex, fetch_author_works, mark_watched_authors, merge_report_works
from src.fetch_new import CandidateFetcher
from src.models import CandidateWork, RankedWork
from src.report_html import render_html
from src.settings import AuthorWatchConfig, TrackedAuthor, load_settings


class AuthorWatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AuthorWatchConfig(enabled=True, authors=[TrackedAuthor(name="Researcher", openalex_ids=["A123"], institution_ids=["I456"])])
        self.settings = load_settings(Path(__file__).resolve().parents[1])
        self.settings.author_watch = self.config

    def work(self, aid="A123", institution="I456", title="TC4 ductile fracture") -> CandidateWork:
        return CandidateWork(source="test", identifier="w", title=title, authors=["Researcher"],
                             extra={"openalex_authorships": [{"author_id": aid, "institution_ids": [institution]}]})

    def ranked(self, work: CandidateWork) -> RankedWork:
        return RankedWork(**work.model_dump(),score=0.4,similarity=0.5,recency_score=1,metric_score=0,
                          author_bonus=0,venue_bonus=0,label="ignore")

    def test_identifiers_not_name(self) -> None:
        self.assertTrue(mark_watched_authors(self.work(),self.config).extra.get("watched_authors"))
        self.assertFalse(mark_watched_authors(self.work(aid="A999"),self.config).extra.get("watched_authors"))
        self.assertFalse(mark_watched_authors(self.work(institution="I999"),self.config).extra.get("watched_authors"))
        with self.assertRaises(ValidationError):
            TrackedAuthor(name="Invalid",openalex_ids=["Researcher"])

    def test_disable_clears_cache_without_mutation(self) -> None:
        original = mark_watched_authors(self.work(),self.config)
        self.config.enabled = False
        cleared = mark_watched_authors(original,self.config)
        self.assertNotIn("watched_authors",cleared.extra)
        self.assertIn("watched_authors",original.extra)

    def test_topic_and_retraction_still_apply(self) -> None:
        f=object.__new__(CandidateFetcher);f.settings=self.settings
        self.assertTrue(f._filter_by_topic([self.work()]))
        self.assertFalse(f._filter_by_topic([self.work(title="Clinical trial of biomedical implant")]))
        self.assertFalse(f._filter_by_topic([self.work(title="RETRACTED: TC4 ductile fracture")]))
        self.assertTrue(f._filter_by_topic([self.work(title="TC4 ductile fracture of retractable metal coupler")]))
        self.assertTrue(f._filter_by_topic([self.work(title="Polycrystalline plasticity with surface energy effects")]))
        self.assertFalse(f._filter_by_topic([self.work(title="Process scheduling and commodity price prediction")]))

    def test_fresh_cache_does_not_skip_author_discovery(self) -> None:
        f=object.__new__(CandidateFetcher);f.settings=self.settings;f.session=Mock()
        f._load_cache=Mock(return_value=(datetime.now(timezone.utc),[]))
        with patch("src.fetch_new.fetch_author_works",return_value=[self.work()]) as fetch:
            result=f.fetch_all()
        fetch.assert_called_once()
        self.assertEqual(len(result),1)

    def test_author_filter_and_defensive_id_check(self) -> None:
        raw={"id":"https://openalex.org/W1","display_name":"TC4 ductile fracture","publication_date":"2026-09-01",
             "authorships":[{"author":{"id":"https://openalex.org/A123","display_name":"Researcher"},
                             "institutions":[{"id":"https://openalex.org/I456"}]}]}
        with patch("src.author_watch.iter_works",return_value=iter([raw])) as pages:
            result=fetch_author_works(Mock(),self.settings,datetime(2026,8,18,tzinfo=timezone.utc))
        self.assertEqual(len(result),1)
        self.assertIn("authorships.author.id:A123",pages.call_args.args[2]["filter"])
        self.assertIsNone(candidate_from_openalex({**raw,"is_retracted":True}))
        self.assertIsNone(candidate_from_openalex({**raw,"type":"dataset"}))
        raw["authorships"][0]["author"]["id"]="https://openalex.org/A999"
        with patch("src.author_watch.iter_works",return_value=iter([raw])):
            self.assertEqual(fetch_author_works(Mock(),self.settings,datetime.now(timezone.utc)),[])

    def test_news_independent_of_recommendation_threshold(self) -> None:
        watched=self.ranked(mark_watched_authors(self.work(),self.config))
        self.assertEqual(author_news([watched],self.config),[watched])
        regular=watched.model_copy(update={"doi":"10.1234/test"})
        duplicate=watched.model_copy(update={"doi":"https://doi.org/10.1234/TEST","identifier":"other"})
        self.assertEqual(len(merge_report_works([regular],[duplicate])),1)
        with patch.object(Path,"mkdir"),patch.object(Path,"write_text") as write:
            render_html([],Path("unused.html"),watched_works=[watched])
        self.assertIn("Researcher",write.call_args.args[0])
        self.assertIn("重点作者新作",write.call_args.args[0])

    def test_verified_author_bonus(self) -> None:
        with patch.dict(sys.modules,{"src.vectorizer":SimpleNamespace(TextVectorizer=object)}):
            from src.score_rank import WorkRanker
        ranker=object.__new__(WorkRanker);ranker.settings=self.settings;ranker.journal_metrics={}
        ranker.vectorizer=SimpleNamespace(encode=lambda texts:np.zeros((len(texts),2)))
        ranker.index=SimpleNamespace(search=lambda vectors,top_k:(np.ones((len(vectors),1)),None))
        known=mark_watched_authors(self.work(),self.config)
        unknown=self.work(aid="A999").model_copy(update={"identifier":"unknown"})
        out={w.identifier:w for w in ranker.rank([known,unknown])}
        self.assertAlmostEqual(out['w'].score-out['unknown'].score,self.config.score_bonus)

    def test_existing_library_doi_url_deduplication(self) -> None:
        from src.dedupe import DedupeEngine
        storage=Mock()
        storage.iter_items.return_value=[SimpleNamespace(doi="10.1234/test",url=None,title="Old metadata title")]
        candidate=self.work().model_copy(update={"doi":"https://doi.org/10.1234/TEST"})
        self.assertEqual(DedupeEngine(storage).filter([candidate]),[])


if __name__=="__main__": unittest.main()
