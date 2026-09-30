import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from src.entity_tracking import confirm, load_registry, set_enabled, mark_entities, fetch_entities, tracking_report
from src.aminer_source import AMinerSource, normalize_paper, query_plan
from src.author_watch import candidate_from_openalex
from src.fetch_new import CandidateFetcher
from src.models import CandidateWork
from src.research_features import FeedbackEntry, FeedbackModel
from src.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


class EntityTrackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name); self.settings = load_settings(ROOT)

    def add(self, kind='person', ident='p1', mapping=()):
        review = self.base / (ident + '.json')
        review.write_text(json.dumps({'kind':kind, 'status':'needs_review', 'candidates':[
            {'id':ident, 'name':'Confirmed Entity'}]}), 'utf-8')
        return confirm(self.base, review, ident, reason='Checked original identity evidence', openalex_ids=mapping)

    def test_confirmation_persists_and_preserves_existing_entries(self):
        person = self.add(mapping=['A123']); org = self.add('organization', 'i1', ['I456'])
        self.assertEqual(len(load_registry(self.base)['entries']), 2)
        set_enabled(self.base, person['key'], False)
        entries = load_registry(self.base)['entries']
        self.assertFalse(entries[0]['enabled']); self.assertTrue(entries[1]['enabled'])
        self.assertEqual(entries[1], org)
        with self.assertRaises(ValueError): self.add(mapping=['A123'])

    def test_rejects_wrong_provider_type_and_conflicting_bridge(self):
        with self.assertRaises(ValueError): self.add('organization', 'bad', ['A123'])
        self.add(mapping=['A123'])
        with self.assertRaises(ValueError): self.add('person', 'p2', ['A123'])
        self.assertEqual(len(load_registry(self.base)['entries']), 1)

    def test_cannot_confirm_unreturned_or_ambiguous_identity(self):
        review = self.base / 'review.json'
        review.write_text(json.dumps({'kind':'person','status':'needs_review','candidates':[{'id':'x','name':'One'},{'id':'x','name':'Two'}]}),'utf-8')
        for ident in ('x', 'unknown'):
            with self.assertRaises(ValueError): confirm(self.base, review, ident, reason='test')

    def test_name_alone_never_marks_authorship_and_disable_clears_cached_match(self):
        entity = self.add()
        work = CandidateWork(source='aminer', identifier='paper', title='Metal fracture', authors=['Confirmed Entity'])
        self.assertEqual(mark_entities(work, [entity]).extra['watched_entities'], [])
        work.extra['aminer_author_ids'] = ['p1']
        tagged = mark_entities(work, [entity]); self.assertEqual(len(tagged.extra['watched_entities']), 1)
        entity['enabled'] = False
        self.assertEqual(mark_entities(tagged, [entity]).extra['watched_entities'], [])
        self.assertEqual(len(tagged.extra['watched_entities']), 1)

    def test_native_aminer_author_org_venue_ids_are_preserved(self):
        paper = normalize_paper({'id':'paper', 'title':'Metal fracture', 'authors':[{'id':'p1','name':'Name','org_id':['i1']}],
                                 'venue_id':'v1'}, route='info')
        entities = [self.add(), self.add('organization','i1'), self.add('venue','v1')]
        self.assertEqual(len(mark_entities(paper, entities).extra['watched_entities']), 3)

    def test_venue_issn_matches_crossref_without_title_guessing(self):
        entity = self.add('venue','v1')
        entity['issns'] = ['0734-743X']
        work = CandidateWork(source='crossref', identifier='x', title='Metal fracture', extra={'issns':['0734-743X']})
        self.assertEqual(mark_entities(work,[entity]).extra['watched_entities'][0]['evidence'], 'issn')
        from src.entity_tracking import valid_issn
        self.assertTrue(valid_issn('0734-743X'))
        self.assertFalse(valid_issn('0734-7430'))

    def test_malformed_optional_identity_metadata_does_not_crash(self):
        entity = self.add()
        paper = CandidateWork(source='test', identifier='x', title='Metal fracture',
            extra={'aminer_author_ids':[{}], 'aminer_venue_id':{}, 'openalex_authorships':[{'author_id':{},'institution_ids':[{}]}]})
        self.assertEqual(mark_entities(paper, [entity]).extra['watched_entities'], [])

    def test_recommendations_use_confirmed_author_id_without_fabricating_facets(self):
        entity = self.add()
        rows = query_plan(self.settings, [entity])
        targeted = [r for r in rows if r[0].startswith('entity:')]
        self.assertEqual(targeted[0][2]['aminer_author_id'], 'p1')
        self.assertEqual(len(query_plan(self.settings, [{**entity,'enabled':False}])), len(query_plan(self.settings)))
        client = Mock(stopped=False)
        client.summary.return_value = {}
        client.query.return_value = [{'id':'a','title':'Financial commodity prices'}]
        source = AMinerSource(self.settings, self.base/'cache', client=client, entities=[entity])
        source.plan = targeted
        papers = source.fetch()
        self.assertEqual(papers[0].extra['aminer_facets'], [])
        self.assertNotIn('watched_entities', papers[0].extra)
        gate = object.__new__(CandidateFetcher); gate.settings = self.settings
        self.assertEqual(gate._filter_by_topic(papers), [])

    def test_bridge_discovery_is_bounded_rotates_and_verifies_returned_ids(self):
        entries = [self.add('organization','i1',['I1']), self.add('venue','v1',['S1']), self.add('person','p1',['A1'])]
        raw = {'id':'https://openalex.org/W1','display_name':'Steel fracture', 'publication_date':'2026-09-30',
               'authorships':[{'author':{'id':'https://openalex.org/A1'},'institutions':[{'id':'https://openalex.org/I1'}]}],
               'primary_location':{'source':{'id':'https://openalex.org/S1'}}}
        with patch('src.source_paging.iter_works', side_effect=lambda *a,**k: iter([raw])) as pages:
            first = fetch_entities(Mock(), self.settings, entries, datetime.now(timezone.utc), self.base/'cursor')
            self.assertEqual(pages.call_count, 2)
            for call in pages.call_args_list: self.assertEqual(call.kwargs['max_pages'], 1)
            self.assertTrue(first)
            pages.reset_mock()
            fetch_entities(Mock(), self.settings, entries, datetime.now(timezone.utc), self.base/'cursor')
            self.assertIn('authorships.author.id:A1', pages.call_args_list[0].args[2]['filter'])
        wrong = candidate_from_openalex({**raw, 'authorships':[], 'primary_location':{}})
        self.assertEqual(mark_entities(wrong, entries).extra['watched_entities'], [])

    def test_failure_is_isolated_and_does_not_repeat_forever_on_first_entity(self):
        entries = [self.add('organization','i1',['I1']), self.add('venue','v1',['S1']), self.add('person','p1',['A1'])]
        with patch('src.source_paging.iter_works', side_effect=OSError('offline')):
            self.assertEqual(fetch_entities(Mock(), self.settings, entries, datetime.now(timezone.utc), self.base/'cursor'), [])
        self.assertEqual(json.loads((self.base/'cursor/rotation.json').read_text())['next'], 2)

    def test_feedback_counts_are_explicit_deduplicated_and_not_delivery(self):
        entity = self.add()
        paper = CandidateWork(source='aminer', identifier='aminer:a', doi='10.1234/a', title='Metal fracture', extra={'aminer_author_ids':['p1']})
        tagged = mark_entities(paper, [entity])
        feedback = FeedbackModel([FeedbackEntry(doi=paper.doi,rating='transferable')], self.settings.research)
        alias = tagged.model_copy(update={'doi':'https://doi.org/10.1234/A'})
        report = tracking_report([entity], [tagged,alias], [tagged], feedback)[0]
        self.assertEqual(report['candidate_matches'], 1)
        self.assertEqual(report['feedback'], {'transferable':1})
        self.assertEqual(report['selected_for_report'], 1)

    def test_invalid_registry_does_not_break_fetcher_creation(self):
        path = self.base/'config/entity-tracking.json'; path.parent.mkdir(); path.write_text('[]','utf-8')
        self.assertEqual(CandidateFetcher(self.settings,self.base).tracked_entities, [])


if __name__ == '__main__': unittest.main()
