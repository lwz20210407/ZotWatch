import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.models import CandidateWork, RankedWork
from src.research_archive import (import_dossier, load_archive, publish_summary,
    withdraw_summary, attach_topics, publication_lock, TOMBSTONE)
from src.research_dossier import build_dossier, export_dossier
from src.report_html import render_html
from src.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.private, self.public = [self.root / x for x in ('source', 'private', 'public/research')]
        self.work = CandidateWork(source='aminer', identifier='aminer:abc', doi='10.1234/test',
            title='Steel GTN calibration', abstract='Steel GTN calibration requires private_marker.',
            url='https://doi.org/10.1234/test', extra={'aminer_id': 'abc'})
        self.ledger = build_dossier('Fracture <comparison>', [self.work], load_settings(ROOT).research)
        self.ledger['private_note'] = 'private_note_marker'
        self.ledger['documents'] = {'doi:10.1234/test': {'path': 'private_filename.pdf'}}
        export_dossier(self.ledger, self.source)

    def save(self, ledger):
        (self.source / 'evidence-ledger.json').write_text(json.dumps(ledger), 'utf-8')

    def publish(self):
        entry = import_dossier(self.source, self.private)
        publish_summary(self.private, self.public, entry['id'])
        return entry

    def test_import_is_private_idempotent_and_ignores_timestamp(self):
        first = import_dossier(self.source, self.private)
        self.ledger['created_at'] = 'later'
        self.save(self.ledger)
        second = import_dossier(self.source, self.private)
        self.assertEqual(first['id'], second['id'])
        self.assertTrue(second['reused'])
        self.assertFalse(self.public.exists())
        self.ledger['topic'] = 'Another topic'
        self.save(self.ledger)
        self.assertNotEqual(first['id'], import_dossier(self.source, self.private)['id'])

    def test_import_rerenders_html_and_does_not_certify_source(self):
        (self.source / 'dossier.html').write_text('<script>malicious()</script>', 'utf-8')
        entry = import_dossier(self.source, self.private)
        page = (Path(entry['path']) / 'dossier.html').read_text('utf-8')
        self.assertNotIn('<script>', page)
        self.assertIn('&lt;comparison&gt;', page)
        self.assertIn('未重新读取原文核验', page)

    def test_modified_private_archive_refused_without_overwrite(self):
        entry = import_dossier(self.source, self.private)
        ledger_path = Path(entry['path']) / 'evidence-ledger.json'
        altered = copy.deepcopy(self.ledger)
        altered['topic'] = 'External edit'
        ledger_path.write_text(json.dumps(altered), 'utf-8')
        before = ledger_path.read_bytes()
        with self.assertRaises(ValueError): import_dossier(self.source, self.private)
        self.assertEqual(before, ledger_path.read_bytes())

    def test_manifest_identity_must_match_content_and_directory(self):
        entry = import_dossier(self.source, self.private)
        path = Path(entry['path']) / 'manifest.json'
        data = json.loads(path.read_text('utf-8')); data['id'] = 'r_' + '0' * 16
        path.write_text(json.dumps(data), 'utf-8')
        with self.assertRaises(ValueError): load_archive(self.private, entry['id'])

    def test_public_summary_allowlist_excludes_private_material(self):
        with patch('requests.Session.request', side_effect=AssertionError('No network')):
            entry = self.publish()
        exported = ''.join(p.read_text('utf-8') for p in self.public.iterdir())
        for forbidden in ('private_marker', 'private_note_marker', 'private_filename.pdf', 'next_checks', 'evidence_ids'):
            self.assertNotIn(forbidden, exported)
        self.assertIn('Steel GTN calibration', exported)
        self.assertTrue((Path(entry['path']) / 'evidence-ledger.json').is_file())

    def test_backlinks_match_either_identity_and_render_in_report(self):
        entry = self.publish()
        aminer_only = self.work.model_copy(update={'doi': None})
        linked = attach_topics([aminer_only], self.public)[0]
        self.assertEqual(linked.extra['research_topics'][0]['url'], f"research/{entry['id']}.html")
        ranked = RankedWork(**linked.model_dump(), score=.7, similarity=.7, label='recommend',
                            recency_score=0, metric_score=0, author_bonus=0, venue_bonus=0)
        report = self.public.parent / 'report.html'
        render_html([ranked], report)
        self.assertIn(f"research/{entry['id']}.html", report.read_text('utf-8'))
        doi_only = self.work.model_copy(update={'identifier': 'elsewhere', 'extra': {}})
        self.assertEqual(len(attach_topics([doi_only], self.public)[0].extra['research_topics']), 1)
        unrelated = self.work.model_copy(update={'doi': '10.1234/unrelated', 'extra': {}, 'identifier': 'different'})
        self.assertEqual(attach_topics([unrelated], self.public)[0].extra['research_topics'], [])

    def test_withdraw_preserves_private_files_removes_links_and_can_republish(self):
        entry = self.publish()
        ledger = Path(entry['path']) / 'evidence-ledger.json'; before = ledger.read_bytes()
        linked = attach_topics([self.work], self.public)
        withdraw_summary(self.public, entry['id'])
        self.assertEqual(attach_topics(linked, self.public)[0].extra['research_topics'], [])
        self.assertEqual((self.public / (entry['id'] + '.html')).read_text('utf-8'), TOMBSTONE)
        self.assertEqual(ledger.read_bytes(), before)
        publish_summary(self.private, self.public, entry['id'])
        self.assertEqual(len(attach_topics([self.work], self.public)[0].extra['research_topics']), 1)

    def test_invalid_registry_is_nonfatal_and_clears_stale_links(self):
        self.publish()
        linked = attach_topics([self.work], self.public)
        (self.public / 'index.json').write_text('[]', 'utf-8')
        self.assertEqual(attach_topics(linked, self.public)[0].extra['research_topics'], [])

    def test_missing_or_modified_public_page_does_not_produce_link(self):
        entry = self.publish()
        page = self.public / (entry['id'] + '.html')
        page.unlink()  # This test created it.
        self.assertEqual(attach_topics([self.work], self.public)[0].extra['research_topics'], [])
        page.write_text('unmanaged edits', 'utf-8')
        self.assertEqual(attach_topics([self.work], self.public)[0].extra['research_topics'], [])
        for action in (lambda: publish_summary(self.private, self.public, entry['id']),
                       lambda: withdraw_summary(self.public, entry['id'])):
            with self.assertRaises(ValueError): action()
        self.assertEqual(page.read_text('utf-8'), 'unmanaged edits')

    def test_invalid_citation_fails_before_creating_archive(self):
        self.ledger['citations'] = [{'from': 'doi:10.1234/test', 'to': 'doi:10.1234/test', 'relation': 'cites'}]
        self.save(self.ledger)
        with self.assertRaises(ValueError): import_dossier(self.source, self.private)
        self.assertFalse(self.private.exists())

    def test_cross_paper_evidence_fails_before_creating_archive(self):
        second = copy.deepcopy(self.ledger['papers'][0])
        second['work_id'] = 'doi:10.1234/other'; second['external_ids'] = {}
        self.ledger['papers'].append(second); self.save(self.ledger)
        with self.assertRaises(ValueError): import_dossier(self.source, self.private)
        self.assertFalse(self.private.exists())

    def test_id_traversal_and_alias_conflict_rejected(self):
        with self.assertRaises(ValueError): load_archive(self.private, '../outside')
        self.ledger['papers'][0]['external_ids']['doi'] = '10.1234/wrong'
        self.save(self.ledger)
        with self.assertRaises(ValueError): import_dossier(self.source, self.private)

    def test_private_symlink_escape_rejected(self):
        entry = import_dossier(self.source, self.private)
        path = Path(entry['path']) / 'evidence-ledger.json'
        outside = self.root / 'outside.json'; outside.write_bytes(path.read_bytes())
        link = self.root / 'symlink-probe'
        try: link.symlink_to(outside)
        except (OSError, NotImplementedError): self.skipTest('Host does not allow symlinks')
        path.unlink(); path.symlink_to(outside)
        with self.assertRaises(ValueError): load_archive(self.private, entry['id'])

    def test_public_symlink_escape_rejected(self):
        entry = import_dossier(self.source, self.private)
        self.public.mkdir(parents=True)
        outside = self.root / 'outside.html'; outside.write_text('preserve', 'utf-8')
        try: (self.public / (entry['id'] + '.html')).symlink_to(outside)
        except (OSError, NotImplementedError): self.skipTest('Host does not allow symlinks')
        with self.assertRaises(ValueError): publish_summary(self.private, self.public, entry['id'])
        self.assertEqual(outside.read_text('utf-8'), 'preserve')

    def test_publication_lock_rejects_competing_writer(self):
        with publication_lock(self.public):
            with self.assertRaises(FileExistsError):
                with publication_lock(self.public): pass
            self.assertTrue((self.public / '.publish.lock').exists())
        self.assertFalse((self.public / '.publish.lock').exists())


if __name__ == '__main__': unittest.main()
