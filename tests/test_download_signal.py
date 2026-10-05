"""Full-text downloads of delivered papers as implicit feedback.

Measured on 2026-10-05: 374 PDFs in the owner's download folders, 28 of them papers
the digest had delivered. These tests pin what turns a download into a signal and,
just as much, what must not: explicit feedback wins, nothing is persisted, and the
folder paths never leave the machine.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.download_signal import doi_from_name, implicit_entries, matches, scan, title_from_name
from src.research_features import FeedbackEntry, FeedbackModel
from src.settings import load_settings

REPO = Path(__file__).resolve().parents[1]
STATE = {"catalog": {"10.1016/j.ijplas.2007.09.004":
                     {"title": "A new model of metal plasticity and fracture with pressure and Lode dependence"},
                     "10.1/other": {"title": "Something nobody downloaded"}},
         "sent": {"10.1016/j.ijplas.2007.09.004": "2026-09-24", "10.1/other": "2026-09-24"}}


class NameParsingTests(unittest.TestCase):
    def test_english_title_from_the_owner_renamer(self):
        self.assertEqual(title_from_name("【考虑压力和Lode相关的金属塑性与断裂新模型A new model of metal plasticity "
                                         "and fracture with pressure and Lode dependence】"),
                         "A new model of metal plasticity and fracture with pressure and Lode dependence")
        self.assertEqual(title_from_name("博士论文_2024_【中文Effects of LPBF Process Variables on Flaws】"),
                         "Effects of LPBF Process Variables on Flaws")
        self.assertEqual(title_from_name("Smith2023_CRISPR"), "")

    def test_doi_from_a_scansci_file_name(self):
        self.assertEqual(doi_from_name("10.1016_j.ijplas.2026.104835"), "10.1016/j.ijplas.2026.104835")
        self.assertEqual(doi_from_name("Smith2023_CRISPR"), "")


class ScanTests(unittest.TestCase):
    def test_scan_records_title_folder_and_date_but_no_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp, "簇C_应力状态")
            folder.mkdir()
            (folder / "【中文A new model of metal plasticity and fracture with pressure and Lode dependence】.pdf").write_bytes(b"%PDF")
            (folder / "notes.txt").write_text("x")
            signal = scan([tmp])
        self.assertEqual(len(signal["items"]), 1)
        item = signal["items"][0]
        self.assertEqual(item["folder"], "簇C_应力状态")
        self.assertTrue(item["seen"])
        self.assertNotIn(tmp, json.dumps(signal), "no local path may leave the machine")


class MatchingTests(unittest.TestCase):
    def test_delivered_and_downloaded_papers_are_found_by_title_or_doi(self):
        signal = {"items": [
            {"title": "A New Model of Metal Plasticity and Fracture with Pressure and Lode Dependence", "doi": "", "folder": "C"},
            {"title": "", "doi": "10.1/other", "folder": "D"},
            {"title": "A paper the digest never sent", "doi": "", "folder": "E"}]}
        found = sorted(doi for doi, *_ in matches(signal, STATE))
        self.assertEqual(found, ["10.1/other", "10.1016/j.ijplas.2007.09.004"])

    def test_entries_are_positive_and_carry_facets(self):
        settings = load_settings(REPO)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "data").mkdir()
            Path(tmp, "data", "download-signal.json").write_text(json.dumps({"items": [
                {"title": "A new model of metal plasticity and fracture with pressure and Lode dependence",
                 "doi": "", "folder": "C"}]}), encoding="utf-8")
            result = implicit_entries(Path(tmp), STATE, settings.research)
        self.assertEqual(result["matched"], 1)
        entry = result["entries"][0]
        self.assertEqual(entry.rating, "transferable")
        self.assertIn("stress_state", entry.facets, "facets come from the title alone")

    def test_no_signal_file_means_no_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(implicit_entries(Path(tmp), STATE, load_settings(REPO).research)["entries"], [])

    def test_explicit_feedback_wins_over_a_download(self):
        """cli puts download entries first, so an explicit rating on the same DOI replaces it."""
        config = load_settings(REPO).research
        doi = "10.1016/j.ijplas.2007.09.004"
        implicit = FeedbackEntry(doi=doi, rating="transferable", facets=["ductile_fracture"])
        explicit = FeedbackEntry(doi=doi, rating="irrelevant", facets=["ductile_fracture"])
        model = FeedbackModel([implicit, explicit], config)
        self.assertEqual(model.entry_for(doi).rating, "irrelevant")

    def test_an_invalid_doi_costs_one_paper_not_the_batch(self):
        """10.1/other passes matching but not FeedbackEntry validation."""
        settings = load_settings(REPO)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "data").mkdir()
            Path(tmp, "data", "download-signal.json").write_text(json.dumps({"items": [
                {"title": "", "doi": "10.1/other", "folder": "D"},
                {"title": "A new model of metal plasticity and fracture with pressure and Lode dependence",
                 "doi": "", "folder": "C"}]}), encoding="utf-8")
            result = implicit_entries(Path(tmp), STATE, settings.research)
        self.assertEqual([e.doi for e in result["entries"]], ["10.1016/j.ijplas.2007.09.004"])


class RenderTests(unittest.TestCase):
    def test_the_page_reports_the_count(self):
        from src.report_html import render_html
        out = Path(tempfile.mkdtemp()) / "r.html"
        render_html([], out, diagnostics={"downloads": {"matched": 28, "downloads": 374}})
        self.assertIn("你已下载全文 <b style=\"color:var(--ink)\">28</b> 篇", out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
