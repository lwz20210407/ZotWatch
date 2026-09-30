import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.models import CandidateWork, RankedWork
from src.research_workflow import execute, parse_args, export_comparison
from src.research_dossier import build_dossier
from src.aminer_acceptance import profile_fingerprint
from src.settings import load_settings
from src.storage import ProfileStorage
from src.watch_history import WatchHistory

ROOT = Path(__file__).resolve().parents[1]


class ResearchWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.base = self.root / 'repo'; self.output = self.root / 'run'
        self.settings = load_settings(ROOT)
        storage = ProfileStorage(self.base / 'data/profile.sqlite'); storage.initialize(); storage.close()
        self.paper = CandidateWork(source='aminer', identifier='aminer:a', doi='10.1234/a',
            title='Steel ductile fracture damage model', abstract='Steel GTN damage evolution under dynamic loading.',
            extra={'aminer_id': 'a'})
        self.snapshot = self.root / 'snapshot.json'
        self.snapshot.write_text(json.dumps([self.paper.model_dump(mode='json')]), 'utf-8')
        self.settings_patch = patch('src.research_workflow.load_settings', return_value=self.settings)
        self.settings_patch.start(); self.addCleanup(self.settings_patch.stop)

    def args(self, *extra):
        return parse_args(['--base-dir', str(self.base), '--topic', 'Steel fracture evidence',
                           '--snapshot', str(self.snapshot), '--output-dir', str(self.output), *extra])

    def ranked(self):
        return RankedWork(**self.paper.model_dump(), score=.8, similarity=.8, label='consider',
                          recency_score=0, metric_score=0, author_bonus=0, venue_bonus=0).model_dump(mode='json')

    def test_offline_end_to_end_and_idempotent_resume_no_state_writes(self):
        before = profile_fingerprint(self.base)
        with patch('requests.Session.request', side_effect=AssertionError('Offline run')):
            result = execute(self.args('--archive'))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['selected_count'], 1)
        self.assertTrue((self.output / 'comparison.csv').exists())
        self.assertTrue((self.output / 'dossier/citation-traces.html').exists())
        self.assertTrue((self.output / 'private-archive').is_dir())
        manifest_before = (self.output / 'run.json').read_bytes()
        self.assertEqual(execute(self.args('--archive', '--resume'))['run_id'], result['run_id'])
        self.assertEqual(manifest_before, (self.output / 'run.json').read_bytes())
        self.assertEqual(before, profile_fingerprint(self.base))
        self.assertFalse((self.base / 'reports').exists())

    def test_changed_inputs_refused_without_overwriting_existing_run(self):
        execute(self.args())
        before = (self.output / 'run.json').read_bytes()
        self.snapshot.write_text('[]', 'utf-8')
        with self.assertRaises(ValueError): execute(self.args('--resume'))
        self.assertEqual(before, (self.output / 'run.json').read_bytes())

    def test_modified_output_is_not_replaced(self):
        execute(self.args())
        path = self.output / 'comparison.csv'; path.write_text('human changes', 'utf-8')
        with self.assertRaises(ValueError): execute(self.args('--resume'))
        self.assertEqual(path.read_text('utf-8'), 'human changes')

    def test_requires_empty_directory_or_compatible_resume(self):
        self.output.mkdir(); (self.output / 'note.txt').write_text('preserve', 'utf-8')
        with self.assertRaises(ValueError): execute(self.args())
        with self.assertRaises(ValueError): execute(self.args('--resume'))

    def test_partial_ranking_resumes_without_repeating_discovery(self):
        source = Mock(); source.fetch.return_value = [self.paper]; source.stats = {'calls': 2, 'warnings': []}
        partial = {'status': 'partial', 'ranked': [self.ranked()]}
        complete = {'status': 'complete', 'ranked': [self.ranked()]}
        options = ('--discover', '--facet', 'ductile_fracture', '--rank')
        with patch('src.research_workflow.AMinerSource', return_value=source), patch('src.research_workflow.run_ranking', side_effect=[partial, complete]):
            first = execute(self.args(*options))
            self.assertEqual(first['status'], 'partial')
            self.assertFalse((self.output / 'dossier').exists())
            second = execute(self.args(*options, '--resume', '--max-new-texts', '50'))
        self.assertEqual(source.fetch.call_count, 1)
        self.assertEqual(second['status'], 'complete')

    def test_auth_failure_is_not_reported_as_empty_success(self):
        source = Mock(); source.fetch.return_value = []; source.stats = {'calls': 1, 'warnings': ['credential_or_permission_failure']}
        with patch('src.research_workflow.AMinerSource', return_value=source):
            result = execute(self.args('--discover', '--facet', 'ductile_fracture'))
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['stage'], 'discovery')
        self.assertFalse(result.get('discovery_done', False))

    def test_discovery_retry_uses_same_run_after_temporary_failure(self):
        source = Mock(); source.fetch.return_value = []; source.stats = {'calls': 1, 'warnings': ['network_failure']}
        with patch('src.research_workflow.AMinerSource', return_value=source):
            first = execute(self.args('--discover', '--facet', 'ductile_fracture'))
            source.fetch.return_value = [self.paper]; source.stats = {'calls': 1, 'warnings': []}
            second = execute(self.args('--discover', '--facet', 'ductile_fracture', '--resume'))
        self.assertEqual(first['run_id'], second['run_id'])
        self.assertEqual(second['status'], 'complete')

    def test_empty_unranked_metadata_stays_held_for_minimal_aminer_snapshot(self):
        self.snapshot.write_text(json.dumps({'aminer_candidates': [{'title': self.paper.title,
            'url': 'https://www.aminer.cn/pub/a'}]}), 'utf-8')
        result = execute(self.args())
        self.assertEqual(result['status'], 'empty')
        selection = json.loads((self.output / 'selection.json').read_text('utf-8'))
        self.assertEqual(selection['review']['held_count'], 1)

    def test_archive_failure_resumes_completed_dossier(self):
        with patch('src.research_workflow.import_dossier', side_effect=OSError('simulated')):
            with self.assertRaises(OSError): execute(self.args('--archive'))
        path = self.output / 'dossier/evidence-ledger.json'; before = path.read_bytes()
        result = execute(self.args('--archive', '--resume'))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(path.read_bytes(), before)

    def test_sent_history_filters_candidate_and_remains_unchanged(self):
        history = WatchHistory(self.base / 'data/watch-state'); history.stage([self.paper]); history.commit()
        before = profile_fingerprint(self.base)
        result = execute(self.args())
        self.assertEqual(result['status'], 'empty')
        self.assertEqual(result['selected_count'], 0)
        self.assertEqual(before, profile_fingerprint(self.base))

    def test_unknown_facets_rejected_before_network_or_output(self):
        with self.assertRaises(ValueError): execute(self.args('--discover', '--facet', 'unknown'))
        self.assertFalse(self.output.exists())

    def test_evidence_changes_prevent_resume(self):
        doc = self.root / 'a.md'; doc.write_text('Steel GTN calibration.', 'utf-8')
        mapping = self.root / 'map.json'
        mapping.write_text(json.dumps({'documents': [{'work_id': 'aminer:a', 'path': 'a.md'}]}), 'utf-8')
        execute(self.args('--evidence-map', str(mapping)))
        doc.write_text('Changed scientific evidence.', 'utf-8')
        with self.assertRaises(ValueError): execute(self.args('--evidence-map', str(mapping), '--resume'))

    def test_csv_titles_cannot_execute_spreadsheet_formulas(self):
        malicious = self.paper.model_copy(update={'title': '  =HYPERLINK("https://example.org")'})
        ledger = build_dossier('Example', [malicious], self.settings.research)
        export_comparison(ledger, self.root / 'comparison.csv')
        self.assertIn("'  =HYPERLINK", (self.root / 'comparison.csv').read_text('utf-8-sig'))

    def test_discovery_keeps_production_switch_off_and_enforces_budget(self):
        source = Mock(); source.fetch.return_value = [self.paper]; source.stats = {'calls': 2, 'warnings': []}
        with patch('src.research_workflow.AMinerSource', return_value=source) as factory:
            execute(self.args('--discover', '--facet', 'ductile_fracture', '--max-requests', '6'))
        config = factory.call_args.args[0]
        self.assertTrue(config.sources.aminer.enabled)
        self.assertEqual(config.sources.aminer.max_requests, 6)
        self.assertTrue(config.sources.aminer.free_only)
        self.assertFalse(self.settings.sources.aminer.enabled)


if __name__ == '__main__': unittest.main()
