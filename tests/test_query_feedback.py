import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.models import CandidateWork, RankedWork
from src.query_feedback import schedule, annotate_counts, main
from src.research_features import FeedbackEntry, FeedbackModel
from src.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


class QueryFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.rows = [(name, 'recommend', {'topics':[name]}) for name in ['a','b','c','d','e']]
        self.settings = load_settings(ROOT)

    def test_no_feedback_keeps_rotation(self):
        ordered, explain = schedule(self.rows, 2, 3, {})
        self.assertEqual([r[0] for r in ordered], ['c','d','e','a','b'])
        self.assertFalse(explain['feedback_active'])

    def test_positive_priority_never_removes_rotating_exploration(self):
        seen = set()
        for start in range(len(self.rows)):
            ordered, explain = schedule(self.rows, start, 3, {'e':0.9,'d':0.5,'a':-0.9})
            self.assertEqual(ordered[0][0], self.rows[start][0]); seen.add(ordered[0][0])
            self.assertTrue(explain['feedback_active'])
        self.assertEqual(len(seen), 5)

    def test_material_error_is_paper_feedback_not_direction_disinterest(self):
        model = FeedbackModel([FeedbackEntry(doi='10.1234/a', rating='irrelevant', reason='wrong_material', facets=['lpbf_tc4'])], self.settings.research)
        self.assertEqual(model.preferences, {})
        w = RankedWork(source='test', identifier='a', doi='10.1234/a', title='LPBF Ti6Al4V plasticity',
            score=0.7, similarity=0.7, label='consider', recency_score=0, metric_score=0, author_bonus=0, venue_bonus=0, extra={'primary_problem':'lpbf_tc4'})
        out = model.apply([w], self.settings.scoring.thresholds)[0]
        self.assertLess(out.score, w.score)
        self.assertEqual(out.extra['feedback_reason'], 'wrong_material')

    def test_counts_preserve_multiple_routes_but_not_duplicate_works(self):
        summary = {'queries':[{'query_id':'q1'},{'query_id':'q2'}]}
        w = CandidateWork(source='aminer', identifier='a', doi='10.1234/a', title='A', extra={'provenance':[{'query_id':'q1'},{'query_id':'q2'}]})
        result = annotate_counts(summary, [w,w], 'selected_for_report')
        self.assertEqual([r['selected_for_report'] for r in result['queries']], [1,1])
        self.assertNotIn('selected_for_report', summary['queries'][0])

    def test_preview_is_offline(self):
        with tempfile.TemporaryDirectory() as tmp, patch('requests.Session.request', side_effect=AssertionError('offline')):
            main(['--base-dir',str(ROOT),'--output-dir',tmp])
            result = json.loads((Path(tmp)/'query-plan.json').read_text('utf-8'))
            self.assertEqual(result['network_calls'],0)


if __name__ == '__main__': unittest.main()
