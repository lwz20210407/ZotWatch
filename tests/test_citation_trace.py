import json
import tempfile
import unittest
from pathlib import Path

from src.citation_trace import marker_numbers, propose_citations, reference_matches
from src.models import CandidateWork
from src.research_dossier import build_dossier, export_dossier, verify_citations
from src.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


def work(name, title=None, doi=None):
    return CandidateWork(source='snapshot', identifier=name, title=title or 'Grounded fracture investigation ' + name,
                         doi=doi or '10.1234/' + name)


class CitationTraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.a, self.b = work('a'), work('b')
        self.papers = {'doi:10.1234/a': self.a, 'doi:10.1234/b': self.b}

    def propose(self, text, page=False):
        blocks = [{'locator': 'p.1', 'text': text}] if page else [
            {'locator': f'L{i}', 'text': line} for i, line in enumerate(text.splitlines(), 1)]
        docs = {'doi:10.1234/a': {'blocks': blocks}}
        proposed, unresolved = propose_citations(docs, self.papers)
        verified, failed = verify_citations(proposed, docs, self.papers)
        return verified, unresolved + failed

    def test_numeric_marker_groups_and_bounded_ranges(self):
        self.assertEqual(marker_numbers('[1, 3–5; 7]'), [1, 3, 4, 5, 7])
        for invalid in ('[1-1000]', '[5-2]', '[0]', '[1-2-3]', '[x]', '1'):
            self.assertEqual(marker_numbers(invalid), [])

    def test_exact_doi_does_not_accept_prefix_or_conflicting_title(self):
        self.assertFalse(reference_matches('[1] DOI:10.1234/b123', self.b))
        self.assertFalse(reference_matches('[1] ' + self.b.title + ' DOI:10.1234/c', self.b))
        self.assertTrue(reference_matches('[1] DOI:10.1234/B.', self.b))
        self.assertTrue(reference_matches('[1] ' + self.b.title, self.b))
        self.assertFalse(reference_matches('[1] DOI:10.1234/b DOI:10.1234/c', self.b))

    def test_detects_grounded_context_and_preserves_original_words(self):
        edges, gaps = self.propose('We did not adopt the method [1].\nReferences\n[1] DOI:10.1234/b')
        self.assertEqual(gaps, [])
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['quote'], 'We did not adopt the method [1].')
        self.assertEqual(edges[0]['relation'], 'cites')
        self.assertEqual(edges[0]['locator'], 'L1')

    def test_group_marker_links_only_matched_selected_targets(self):
        edges, gaps = self.propose('Prior work [1, 2–3].\nReferences\n[1] DOI:10.1234/b\n[2] DOI:10.1234/c\n[3] DOI:10.1234/d')
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['marker'], '[1, 2–3]')
        self.assertEqual(edges[0]['reference_marker'], '[1]')
        self.assertEqual(len(gaps), 2)

    def test_no_heading_does_not_guess_reference_section(self):
        edges, gaps = self.propose('A model [1].\n[1] DOI:10.1234/b')
        self.assertEqual(edges, [])
        self.assertEqual(gaps[0]['reason'], 'reference_heading_not_found')

    def test_reference_only_is_not_a_citation_context(self):
        self.assertEqual(self.propose('References\n[1] DOI:10.1234/b'), ([], []))

    def test_duplicate_reference_numbers_are_unresolved(self):
        edges, gaps = self.propose('A model [1].\nReferences\n[1] DOI:10.1234/b\n[1] DOI:10.1234/b')
        self.assertEqual(edges, [])
        self.assertEqual(gaps[0]['reason'], 'missing_or_duplicate_reference_number')

    def test_multiline_pdf_reference_keeps_page_anchor(self):
        edges, gaps = self.propose('Prior work [1].\nReferences\n[1] Author.\nDOI:10.1234/b', page=True)
        self.assertEqual(gaps, [])
        self.assertEqual(edges[0]['reference_locator'], 'p.1')

    def test_cross_locator_reference_requires_manual_anchor(self):
        edges, gaps = self.propose('Prior work [1].\nReferences\n[1] Author.\nDOI:10.1234/b')
        self.assertEqual(edges, [])
        self.assertEqual(gaps[0]['reason'], 'reference_requires_manual_anchor')

    def test_ambiguous_title_does_not_pick_a_target(self):
        self.papers['doi:10.1234/c'] = work('c', title=self.b.title)
        edges, gaps = self.propose('A model [1].\nReferences\n[1] ' + self.b.title)
        self.assertEqual(edges, [])
        self.assertEqual(gaps[0]['reason'], 'target_not_selected_or_ambiguous')

    def test_source_is_never_its_own_target(self):
        edges, _ = self.propose('A model [1].\nReferences\n[1] DOI:10.1234/a')
        self.assertEqual(edges, [])

    def test_limit_records_incomplete_scan(self):
        docs = {'doi:10.1234/a': {'blocks': [{'locator': 'p.1', 'text': 'First [1].\nSecond [1].\nReferences\n[1] DOI:10.1234/b'}]}}
        proposals, gaps = propose_citations(docs, self.papers, limit=1)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(gaps[-1]['reason'], 'automatic_citation_limit_reached')

    def test_dossier_opt_in_export_and_escaped_trace_browser(self):
        self.a.title = '<img src=x onerror=alert(1)>'
        path = self.root / 'a.md'
        path.write_text('We compare [1].\n## References\n[1] DOI:10.1234/b', 'utf-8')
        options = dict(evidence_map={'documents': [{'work_id': 'doi:10.1234/a', 'path': 'a.md'}]}, evidence_root=self.root)
        cfg = load_settings(ROOT).research
        self.assertEqual(build_dossier('Topic', [self.a, self.b], cfg, **options)['citations'], [])
        ledger = build_dossier('Topic', [self.a, self.b], cfg, auto_citations=True, **options)
        self.assertEqual(len(ledger['citations']), 1)
        export_dossier(ledger, self.root / 'out')
        page = (self.root / 'out/citation-traces.html').read_text('utf-8')
        self.assertNotIn('<img src=x', page)
        self.assertIn('&lt;img', page)
        self.assertIn('We compare [1].', page)
        trace = json.loads((self.root / 'out/citation-traces.json').read_text('utf-8'))
        self.assertEqual(len(trace['edges']), 1)
        export_dossier(ledger, self.root / 'imported', archive_notice='Imported')
        self.assertIn('未重新核验原文', (self.root / 'imported/citation-traces.html').read_text('utf-8'))


if __name__ == '__main__': unittest.main()
