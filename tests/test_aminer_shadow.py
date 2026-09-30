import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.aminer_shadow import export_shadow
from src.models import CandidateWork
from src.research_dossier import read_snapshot


class ShadowEvidenceTests(unittest.TestCase):
    def test_export_keeps_replayable_cohorts_and_does_not_invent_labels(self):
        baseline = CandidateWork(source='crossref', identifier='x', title='Shared paper', doi='10.1234/shared')
        shared = baseline.model_copy(update={'source':'aminer', 'extra':{'aminer_id':'shared'}})
        novel = CandidateWork(source='aminer', identifier='aminer:new', title='=External title',
            abstract='Original abstract', extra={'aminer_id':'new', 'abstract_is_partial':True, 'private_note':'must not export'})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = export_shadow({'baseline':[baseline], 'aminer':[shared, novel], 'combined':[baseline, novel]},
                {'calls':2}, root, window_days=30, baseline_cached_at='2026-09-30T00:00:00Z')
            self.assertEqual(result['aminer_distinct_candidates'], 1)
            self.assertEqual(result['human_labels'], 0)
            self.assertEqual(len(read_snapshot(root/'aminer-shadow.json', 'combined')), 2)
            raw = (root/'aminer-shadow.json').read_text('utf-8')
            self.assertNotIn('must not export', raw)
            self.assertEqual(json.loads(raw)['stage'], 'topic_filtered_before_library_history_and_ranking')
            with (root/'aminer-review.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['work_id'], 'aminer:new')
            self.assertEqual(rows[0]['relevant'], '')
            self.assertTrue(rows[0]['title'].startswith("'="))


if __name__ == '__main__': unittest.main()
