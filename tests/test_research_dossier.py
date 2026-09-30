import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.models import CandidateWork
from src.settings import load_settings
from src.research_dossier import (build_dossier, export_dossier, read_document, read_snapshot,
                                  paper_id, select_papers, main, safe_url)

ROOT = Path(__file__).resolve().parents[1]


def paper(n="a", **kwargs):
    data = dict(source="aminer", identifier="aminer:" + n, title="Metal fracture and damage study " + n,
                doi="10.1234/" + n, url="https://doi.org/10.1234/" + n,
                abstract="Steel specimens were calibrated with DIC. We did not use GTN.",
                extra={"aminer_id": n})
    data.update(kwargs)
    return CandidateWork(**data)


class DossierTests(unittest.TestCase):
    def setUp(self):
        self.config = load_settings(ROOT).research
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_abstract_mentions_are_not_fulltext_or_method_adoption(self):
        ledger = build_dossier("Fracture", [paper()], self.config)
        self.assertEqual(ledger["papers"][0]["evidence_level"], "abstract")
        self.assertEqual(ledger["papers"][0]["fields"]["conditions"]["status"], "unknown")
        self.assertIn("GTN", ledger["papers"][0]["fields"]["model"]["mentions"])
        self.assertTrue(all(e["interpretation"] == "keyword_mention_only" for e in ledger["evidence"]))
        self.assertLessEqual(sum(len(e["quote"].split()) for e in ledger["evidence"]), 24)

    def test_generated_summary_is_not_evidence(self):
        ledger = build_dossier("Topic", [paper(abstract=None, extra={"aminer_id": "a", "generated_summary": "Steel GTN DIC 300 K"})], self.config)
        self.assertEqual(ledger["evidence"], [])
        self.assertTrue(all(f["status"] == "unknown" for f in ledger["papers"][0]["fields"].values()))

    def test_title_only_method_mention_is_labeled_title_evidence(self):
        ledger = build_dossier('Topic', [paper(title='Inverse identification using non-linear VFM', abstract=None)], self.config)
        self.assertEqual(ledger['papers'][0]['evidence_level'], 'title_only')
        self.assertTrue(ledger['evidence'])
        self.assertTrue(all(e['level'] == 'title_only' and e['locator'] == 'title' for e in ledger['evidence']))

    def test_markdown_line_anchor_and_file_hash(self):
        path = self.root / "paper.md"
        path.write_text("# Specimen\nTi-6Al-4V specimens were tested at high strain rate.\n", "utf-8")
        mapping = {"documents": [{"work_id": "doi:10.1234/a", "path": "paper.md"}]}
        ledger = build_dossier("Topic", [paper()], self.config, evidence_map=mapping, evidence_root=self.root)
        self.assertEqual(ledger["papers"][0]["evidence_level"], "local_text")
        self.assertTrue(any(e["locator"] == "L2" for e in ledger["evidence"]))
        self.assertEqual(len(ledger["documents"]["doi:10.1234/a"]["sha256"]), 64)
        self.assertNotIn("blocks", ledger["documents"]["doi:10.1234/a"])

    def test_file_escape_and_unsupported_types_rejected(self):
        (self.root / "file.py").write_text("raise RuntimeError()", "utf-8")
        for name in ["../outside.md", "file.py"]:
            with self.assertRaises(ValueError): read_document(name, self.root)

    def test_blank_pdf_is_not_reported_as_successful_fulltext(self):
        from pypdf import PdfWriter
        writer = PdfWriter(); writer.add_blank_page(width=100, height=100)
        with (self.root / "blank.pdf").open("wb") as f: writer.write(f)
        d = read_document("blank.pdf", self.root)
        self.assertEqual(d["status"], "no_extractable_text")

    def test_pdf_text_has_real_page_anchor(self):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer = PdfWriter(); page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject(); stream.set_data(b'BT /F1 12 Tf 20 260 Td (Steel GTN calibration.) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
        with (self.root / 'text.pdf').open('wb') as f: writer.write(f)
        ledger = build_dossier('PDF evidence', [paper()], self.config,
            evidence_map={'documents': [{'work_id': 'aminer:a', 'path': 'text.pdf'}]}, evidence_root=self.root)
        self.assertEqual(ledger['papers'][0]['evidence_level'], 'local_text')
        self.assertTrue(any(e['locator'] == 'p.1' for e in ledger['evidence']))

    def test_invalid_map_shape_rejected(self):
        with self.assertRaises(ValueError): build_dossier('Topic', [paper()], self.config, evidence_map=['not an object'])

    def test_local_document_does_not_require_an_abstract(self):
        (self.root / 'a.txt').write_text('Steel GTN model calibration.', 'utf-8')
        ledger = build_dossier('Topic', [paper(abstract=None)], self.config,
            evidence_map={'documents': [{'work_id': 'aminer:a', 'path': 'a.txt'}]}, evidence_root=self.root)
        self.assertEqual(ledger['papers'][0]['evidence_level'], 'local_text')

    def test_grounded_citation_requires_context_reference_and_target_identity(self):
        a, b = paper("a"), paper("b")
        self.root.joinpath("a.txt").write_text("We compare with the damage model [1].\n[1] Metal fracture and damage study b. DOI:10.1234/b\n", "utf-8")
        edge = {"from": paper_id(a), "to": paper_id(b), "marker": "[1]", "locator": "L1",
                "quote": "We compare with the damage model [1].", "reference_locator": "L2",
                "reference_quote": "[1] Metal fracture and damage study b. DOI:10.1234/b"}
        mapping = {"documents": [{"work_id": paper_id(a), "path": "a.txt"}], "citations": [edge]}
        ledger = build_dossier("Topic", [a, b], self.config, evidence_map=mapping, evidence_root=self.root)
        self.assertEqual(len(ledger["citations"]), 1)
        self.assertEqual(ledger["citations"][0]["relation"], "cites")
        for change in [{"quote": "Invented support [1]."}, {"relation": "supports"}, {"to": "doi:10.1234/missing"}]:
            mapping["citations"] = [{**edge, **change}]
            ledger = build_dossier("Topic", [a, b], self.config, evidence_map=mapping, evidence_root=self.root)
            self.assertEqual(ledger["citations"], [])
            self.assertEqual(len(ledger["unresolved_citations"]), 1)

    def test_html_escapes_source_and_blocks_script_links(self):
        w = paper(title='<script>alert(1)</script>', url='javascript:alert(1)')
        ledger = build_dossier('<img src=x onerror=alert(1)>', [w], self.config)
        export_dossier(ledger, self.root / "out")
        page = (self.root / "out/dossier.html").read_text("utf-8")
        self.assertNotIn('<script>', page)
        self.assertNotIn('href="javascript:', page)
        self.assertIn('&lt;script&gt;', page)
        self.assertEqual(safe_url('https://secret:password@example.com'), '')

    def test_minimal_weekly_snapshot_keeps_aminer_identity(self):
        path = self.root / "snapshot.json"
        path.write_text(json.dumps({"aminer_candidates": [{"title": "A paper", "url": "https://www.aminer.cn/pub/abc"}]}), "utf-8")
        rows = read_snapshot(path)
        self.assertEqual(paper_id(rows[0]), "aminer:abc")
        self.assertEqual(select_papers(rows, ["aminer:abc"], 10), rows)
        with self.assertRaises(ValueError): select_papers(rows, ["aminer:missing"], 10)

    def test_offline_cli_does_not_call_network_or_modify_inputs(self):
        path = self.root / "snapshot.json"
        original = json.dumps([paper().model_dump(mode="json")])
        path.write_text(original, "utf-8")
        with patch('requests.Session.request', side_effect=AssertionError('No network allowed')):
            main(['--base-dir', str(ROOT), '--topic', 'Fracture evidence', '--snapshot', str(path), '--output-dir', str(self.root/'out')])
        self.assertEqual(path.read_text('utf-8'), original)
        self.assertTrue((self.root/'out/agent-handoff.json').exists())
        with self.assertRaises(SystemExit):
            main(['--base-dir', str(ROOT), '--topic', 'Fracture', '--snapshot', str(path), '--output-dir', str(self.root/'out')])

    def test_explicit_discovery_is_bounded_and_still_uses_topic_gate(self):
        with patch('src.research_dossier.AMinerSource') as source:
            source.return_value.fetch.return_value = [paper(), paper('bad', title='Commodity price prediction', abstract='Financial market forecasting.')]
            source.return_value.stats = {'calls': 0}
            main(['--base-dir', str(ROOT), '--topic', 'Steel ductile fracture', '--discover', '--facet', 'damage_evolution', '--output-dir', str(self.root/'discovery')])
            cfg = source.call_args.args[0]
            self.assertEqual(cfg.sources.aminer.max_requests, 12)
            self.assertEqual(cfg.sources.aminer.max_recommendation_queries, 2)
            self.assertIn('Steel ductile fracture', cfg.research.facets[0].semantic_query)
            ledger = json.loads((self.root/'discovery/evidence-ledger.json').read_text('utf-8'))
            self.assertEqual(len(ledger['papers']), 1)


if __name__ == '__main__': unittest.main()
