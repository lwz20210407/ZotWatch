import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import requests

from src.http_utils import DeferredRequest, request_with_retry
from src.network_budget import BudgetSession
from src.settings import NetworkConfig, load_settings
from src.models import CandidateWork, RankedWork, ZoteroItem
from src.research_features import FeedbackEntry, FeedbackModel, load_feedback, save_feedback, collaboration_groups
from src.problem_ranking import build_problem_profiles, diverse_select, local_evidence_graph
from src.fetch_new import CandidateFetcher
from src.evaluate import evaluate


def response(payload, status=200, headers=None):
    r = requests.Response(); r.status_code = status
    r._content = json.dumps(payload).encode(); r.headers.update(headers or {})
    return r


def paper(i, **kwargs):
    data = dict(source='test', identifier=str(i), doi=f'10.1234/p{i}', title=f'Steel inverse identification method {i}',
                abstract='Plasticity parameter calibration', score=0.8-i*0.01, similarity=0.8,
                recency_score=0, metric_score=0, author_bonus=0, venue_bonus=0, label='consider')
    data.update(kwargs)
    return RankedWork(**data)


def facet_id(settings, term="LS-OPT"):
    """The facet covering `term`; survives renames of the research taxonomy."""
    for facet in settings.research.facets:
        if any(term.lower() == t.lower() for t in facet.terms):
            return facet.id
    raise AssertionError(f"no facet covers {term!r}")


class ReliableTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(Path(__file__).resolve().parents[1])

    def test_long_retry_never_sleeps_or_retries_early(self):
        session = Mock(); session.request.return_value = response({}, 429, {'Retry-After': '68413'})
        with patch('src.http_utils.time.sleep') as sleep:
            with self.assertRaises(DeferredRequest):
                request_with_retry(session, 'GET', 'https://api.openalex.org/works', logger=Mock(), context='test')
            sleep.assert_not_called(); self.assertEqual(session.request.call_count, 1)
        session.defer_host.assert_called_once()

    def test_cooldown_persists_and_other_provider_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BudgetSession(tmp, NetworkConfig())
            session.defer_host('https://api.openalex.org/works', 68413)
            restored = BudgetSession(tmp, NetworkConfig())
            with patch('requests.Session.request', return_value=response({'ok': True})) as req:
                with self.assertRaises(DeferredRequest): restored.get('https://api.openalex.org/works')
                self.assertEqual(restored.get('https://api.crossref.org/works').status_code, 200)
                self.assertEqual(req.call_count, 1)

    def test_success_cache_survives_new_session_no_secret_persisted(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'OPENALEX_API_KEY': 'PRIVATEKEY'}):
            with patch('requests.Session.request', return_value=response({'results': []})) as req:
                session = BudgetSession(tmp, NetworkConfig())
                session.get('https://api.openalex.org/works', params={'search': 'plasticity'})
                self.assertEqual(req.call_args.kwargs['params']['api_key'], 'PRIVATEKEY')
                restored = BudgetSession(tmp, NetworkConfig())
                restored.get('https://api.openalex.org/works', params={'search': 'plasticity'})
                self.assertEqual(req.call_count, 1)
            self.assertNotIn('PRIVATEKEY', ''.join(p.read_text() for p in Path(tmp).glob('*.json')))

    def test_budget_rejects_without_network_and_rotation_persists(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'OPENALEX_API_KEY': ''}):
            session = BudgetSession(tmp, NetworkConfig(openalex_anonymous_budget=0))
            with patch('requests.Session.request') as req:
                with self.assertRaises(DeferredRequest): session.get('https://api.openalex.org/works', params={'search.semantic': 'metal'})
                req.assert_not_called()
            self.assertEqual(session.rotate('q', ['a','b','c'], 2), ['a','b'])
            self.assertEqual(BudgetSession(tmp, NetworkConfig()).rotate('q', ['a','b','c'], 2), ['c','a'])

    def test_scoped_feedback_does_not_penalize_other_problem_and_reset(self):
        config = self.settings.research
        entries = [FeedbackEntry(doi='10.1234/p1', rating='irrelevant', scope=facet_id(self.settings), facets=[facet_id(self.settings)])]
        model = FeedbackModel(entries, config)
        same = paper(1, extra={'primary_problem':facet_id(self.settings)})
        other = paper(1, extra={'primary_problem':facet_id(self.settings,'ductile fracture')})
        a = model.apply([same], self.settings.scoring.thresholds)[0]
        b = model.apply([other], self.settings.scoring.thresholds)[0]
        self.assertLess(a.score, same.score); self.assertEqual(b.score, other.score)
        neutral = FeedbackModel([FeedbackEntry(doi=same.doi, rating='later', facets=[facet_id(self.settings)])], config)
        self.assertEqual(neutral.preferences, {})
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'config').mkdir()
            save_feedback(tmp, entries[0].model_dump(), config)
            save_feedback(tmp, {'doi':same.doi,'rating':'reset','scope': facet_id(self.settings)}, config)
            restored = FeedbackModel(load_feedback(tmp, config), config)
            self.assertEqual(restored.preferences, {})

    def test_budget_break_resumes_unprocessed_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = BudgetSession(tmp, NetworkConfig())
            sequence = session.iter_rotation('venues', ['a', 'b', 'c'], 3)
            self.assertEqual(next(sequence), 'a')
            self.assertEqual(next(sequence), 'b')  # a processed; b yielded but budget check stops it.
            sequence.close()
            restored = BudgetSession(tmp, NetworkConfig())
            self.assertEqual(list(restored.iter_rotation('venues', ['a', 'b', 'c'], 2)), ['b', 'c'])

    def test_profiles_are_separate_and_respect_explicit_collections(self):
        config = self.settings.research.model_copy(deep=True)
        config.facets = config.facets[:2]
        config.facets[0].collection_keys = ['plasticity']
        config.facets[1].collection_keys = ['thermal']
        items = [ZoteroItem(key='a', version=1, title='A', collections=['plasticity']),
                 ZoteroItem(key='b', version=1, title='B', collections=['thermal'])]
        first, second = config.facets[0].id, config.facets[1].id
        profiles = build_problem_profiles(items, np.eye(2), config)
        np.testing.assert_allclose(profiles[first]['centroid'], [1.0,0.0])
        self.assertEqual(profiles[second]['count'], 1)

    def test_semantic_route_skips_keyword_groups_but_still_needs_a_mechanics_anchor(self):
        """Cross-circle discovery survives; characterisation no longer rides in on it.

        A semantic facet hit skips the keyword group sets on purpose -- that is how
        wording the keyword lists cannot anticipate gets found. But the hit alone used
        to bypass the topic gate outright, and because facet queries are sentences,
        OpenAlex returned the sentence's neighbours: a facet phrased around
        microstructure delivered EBSD/TEM papers straight into the digest. The hit now
        also has to name a mechanics-of-materials object.
        """
        f = object.__new__(CandidateFetcher); f.settings = self.settings
        semantic = {'semantic_facets': [facet_id(self.settings)]}
        # Wording no keyword group anticipates, but plainly about failure: still kept.
        unusual = CandidateWork(source='openalex', identifier='W1',
                                title='Unusual new wording for a damage model', extra=semantic)
        self.assertTrue(f._filter_by_topic([unusual]))
        # Same privileged route, nothing mechanical about it: no longer waved through.
        characterisation = CandidateWork(
            source='openalex', identifier='W2',
            title='Microstructural evolution and texture of a gradient alloy',
            abstract='Grain refinement and precipitation behaviour observed by EBSD and TEM.',
            extra=semantic)
        self.assertFalse(f._filter_by_topic([characterisation]))
        self.assertFalse(f._filter_by_topic([unusual.model_copy(
            update={'title': 'Retraction: Unusual new wording for a damage model'})]))
        raw = {'id':'https://openalex.org/W1','display_name':'New wording','publication_date':'2026-09-17'}
        f.session = Mock(); f.settings.research.facets = f.settings.research.facets[:1]
        with patch('src.fetch_new.request_with_retry',return_value=response({'results':[raw]})) as request:
            out = f._fetch_semantic(datetime.now(timezone.utc))
        self.assertIn('search.semantic', request.call_args.kwargs['params'])
        self.assertNotIn('is_oa', request.call_args.kwargs['params']['filter'])
        self.assertTrue(out[0].extra['semantic_facets'])

    def test_diversity_reserves_relevant_exploration_and_limits_dominant_group(self):
        works = [paper(i,extra={'primary_problem':facet_id(self.settings)}) for i in range(8)]
        works += [paper(9,extra={'primary_problem':'fracture','semantic_facets':['fracture']})]
        works += [paper(10,extra={'primary_problem':'impact'})]
        vec = SimpleNamespace(encode=lambda texts: np.eye(len(texts)))
        selected = diverse_select(works, 4, self.settings.research, vec)
        self.assertEqual(len(selected),4)
        self.assertIn('9',{w.identifier for w in selected})
        self.assertGreater(len({w.extra['primary_problem'] for w in selected}),1)

    def test_collaboration_pairs_require_distinct_papers_and_typed_edges(self):
        known = self.settings.author_watch.authors[0].openalex_ids[0]
        row={'title':'A','url':'https://doi.org/10.1234/a','authors':[{'author_id':known,'name':'Known'},{'author_id':'A99999','name':'New'}]}
        self.assertFalse(collaboration_groups({'research_observations':{'a':row}},self.settings))
        groups=collaboration_groups({'research_observations':{'a':row,'b':{**row,'title':'B'}}},self.settings)
        self.assertEqual(groups[0]['count'],2); self.assertTrue(groups[0]['new_collaborator'])
        graph=local_evidence_graph(paper(0,extra={'cites_seeds':[{'title':'Seed','url':'https://example.org'}], 'nearest_library_work':{'title':'Near'}}))
        self.assertEqual({e['direction'] for e in graph},{'out','similar'})

    def test_evaluation_unjudged_is_not_negative(self):
        rankings=[{'doi':'10.1234/a','facets':['inverse']},{'doi':'10.1234/b'}]
        result=evaluate(rankings,[{'doi':'10.1234/a','relevant':True}],2)
        self.assertIsNone(result['precision_at_k']); self.assertEqual(result['label_coverage'],0.5)
        self.assertEqual(result['judged_precision'],1)


if __name__ == '__main__': unittest.main()
