"""Lineage and method comparison: the two web-page sections added on 2026-10-05.

Both run after the digest is decided and must never change it, so besides what they
compute, these tests pin how they fail: a capped or broken lookup is reported as
incomplete, a paper without open full text falls back to its abstract and says so,
and a slow paper is marked unfinished instead of holding up the run.
"""

import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.lineage import Lookup, build_lineage, resolve_week
from src.method_compare import MethodComparer, UNREPORTED, method_excerpt, pdf_text, pdf_urls
from src.models import CandidateWork
from src.settings import TranslationConfig


class FakeResponse:
    headers = {}

    def __init__(self, data, status=200, content=b""):
        self.status_code, self._data, self.content = status, data, content

    def json(self):
        return self._data

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpenAlex:
    """Answers doi: and openalex: list filters from two small tables."""

    def __init__(self, week, refs):
        self.week, self.refs, self.calls = week, refs, 0

    def request(self, method, url, params=None, **kwargs):
        self.calls += 1
        values = params["filter"].split(":", 1)[1].split("|")
        if params["filter"].startswith("doi:"):
            wanted = {v.replace("https://doi.org/", "") for v in values}
            return FakeResponse({"results": [r for d, r in self.week.items() if d in wanted]})
        return FakeResponse({"results": [r for i, r in self.refs.items() if i in values]})


def paper(title, doi=None, refs=None):
    extra = {"referenced_works": [f"https://openalex.org/{r}" for r in refs]} if refs else {}
    return CandidateWork(source="t", identifier=title, title=title, doi=doi, extra=extra)


def ref(ident, title, year, doi=None, cites=10):
    return {"id": f"https://openalex.org/{ident}", "display_name": title, "publication_year": year,
            "doi": f"https://doi.org/{doi}" if doi else None, "cited_by_count": cites}


LIBRARY = [SimpleNamespace(title="Bai & Wierzbicki 2010 MMC", doi="10.1016/lib.mmc", year=2010),
           SimpleNamespace(title="No DOI item", doi=None, year=2001)]


class LineageTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeOpenAlex(
            week={"10.1/b": {"id": "https://openalex.org/W200", "doi": "https://doi.org/10.1/b",
                             "referenced_works": ["https://openalex.org/W1", "https://openalex.org/W3"]}},
            refs={"W1": ref("W1", "Johnson & Cook 1985", 1985, doi="10.1016/jc"),
                  "W2": ref("W2", "MMC", 2010, doi="10.1016/LIB.MMC"),
                  "W3": ref("W3", "Bao & Wierzbicki 2004", 2004, doi="10.1016/bw")})
        # Paper 1 carries its list from the fetch; paper 2 needs the lookup for it.
        self.works = [paper("New A", doi="10.1/a", refs=["W1", "W2"]), paper("New B", doi="10.1/b")]

    def run_lineage(self, **kw):
        lookup = Lookup(self.api, interval=0, **kw)
        return build_lineage(self.works, LIBRARY, resolve_week(self.works, lookup), lookup)

    def test_new_papers_are_linked_to_the_library_by_doi(self):
        result = self.run_lineage()
        self.assertEqual([p["rank"] for p in result["papers"]], [1])
        self.assertEqual(result["papers"][0]["library"][0]["title"], "Bai & Wierzbicki 2010 MMC")

    def test_shared_references_become_the_lineage_oldest_first(self):
        result = self.run_lineage()
        shared = result["ancestors"]
        self.assertEqual([a["title"] for a in shared], ["Johnson & Cook 1985"])
        self.assertEqual(shared[0]["cited_by"], [1, 2])
        self.assertFalse(shared[0]["in_library"])
        self.assertEqual(result["coverage"]["with_references"], 2)

    def test_a_capped_lookup_is_reported_not_hidden(self):
        result = self.run_lineage(max_requests=1)
        self.assertIn("request_cap", result["coverage"]["incomplete"])
        self.assertEqual(result["ancestors"], [])

    def test_a_failing_session_degrades_to_an_empty_but_honest_result(self):
        broken = mock.Mock()
        broken.request.side_effect = RuntimeError("boom")
        lookup = Lookup(broken, interval=0)
        result = build_lineage(self.works, LIBRARY, resolve_week(self.works, lookup), lookup)
        self.assertEqual(result["papers"], [])
        self.assertIn("RuntimeError", result["coverage"]["incomplete"])


METHODS_TEXT = ("Title\nAbstract text " + "x" * 900 + "\n1. Introduction\nBackground.\n"
                "2. Materials and methods\nLPBF Ti-6Al-4V specimens were tested by SHPB at 3000/s.\n"
                + "body " * 2500 + "\n5. Conclusions\nFracture strain fell by 30%.\nReferences\n[1] x")


class LineageBudgetTests(unittest.TestCase):
    def test_the_first_failure_stops_further_lookups(self):
        """A slow OpenAlex must not cost ~62 s per remaining request."""
        broken = mock.Mock()
        broken.request.side_effect = RuntimeError("timeout")
        lookup = Lookup(broken, interval=0)
        lookup.by_id([f"W{i}" for i in range(500)], "id")
        self.assertEqual(broken.request.call_count, 1)

    def test_the_wall_clock_cap_stops_lookups(self):
        api = FakeOpenAlex(week={}, refs={})
        lookup = Lookup(api, interval=0, max_seconds=0)
        lookup.by_id(["W1", "W2"], "id")
        self.assertEqual(api.calls, 0)
        self.assertIn("time_cap", lookup.incomplete)


class GroundingTests(unittest.TestCase):
    SOURCE = ("SHPB tests were carried out at 25-600 C with strain rates from 1000 to 4600 s-1. "
              "The conventional Johnson-Cook model was calibrated with the same data set, and the "
              "finite element analysis of the specimen used an explicit solver.")

    def test_a_real_quote_passes_despite_respacing_and_hyphens(self):
        from src.method_compare import grounded
        self.assertTrue(grounded("the conventional Johnson Cook model was calibrated with the same data-set",
                                 self.SOURCE))

    def test_a_plausible_but_absent_quote_fails(self):
        """Review case: every word but DIC occurs somewhere in an FE paper."""
        from src.method_compare import grounded
        source = self.SOURCE + " Model parameters were measured; the analysis used measurements."
        self.assertFalse(grounded("the model parameters were calibrated using DIC measurements "
                                  "and the finite element analysis", source))

    def test_a_value_naming_something_absent_is_rejected(self):
        from src.method_compare import named_terms_present
        self.assertTrue(named_terms_present("Johnson-Cook 模型", self.SOURCE))
        self.assertFalse(named_terms_present("DIC + 有限元反演", self.SOURCE))


class SafeUrlTests(unittest.TestCase):
    def test_only_https_to_named_hosts(self):
        from src.method_compare import safe_url
        self.assertTrue(safe_url("https://www.mdpi.com/x.pdf"))
        for url in ("http://www.mdpi.com/x.pdf", "https://169.254.169.254/latest", "https://localhost/x",
                    "https://[::1]/x", "file:///etc/passwd", "ftp://x.org/a.pdf"):
            self.assertFalse(safe_url(url), url)

    def test_a_redirect_to_an_unsafe_host_is_not_followed(self):
        redirect = FakeResponse(None, status=302)
        redirect.headers = {"Location": "http://169.254.169.254/latest/meta-data"}
        redirect.close = lambda: None
        session = mock.Mock()
        session.get.return_value = redirect
        self.assertIsNone(pdf_text(session, "https://oa.example/a.pdf"))
        self.assertEqual(session.get.call_count, 1)


class MethodTextTests(unittest.TestCase):
    def test_excerpt_starts_at_the_methods_and_keeps_the_conclusions(self):
        excerpt = method_excerpt(METHODS_TEXT)
        self.assertTrue(excerpt.startswith("2. Materials and methods"))
        self.assertIn("Fracture strain fell by 30%", excerpt)
        self.assertNotIn("[1] x", excerpt)

    def test_open_access_pdf_links_come_first_then_arxiv(self):
        work = CandidateWork(source="t", identifier="x", title="x", url="https://arxiv.org/abs/2401.00001")
        record = {"best_oa_location": {"pdf_url": "https://oa.example/a.pdf"},
                  "locations": [{"pdf_url": "https://oa.example/a.pdf"}, {"pdf_url": None}]}
        self.assertEqual(pdf_urls(work, record),
                         ["https://oa.example/a.pdf", "https://arxiv.org/pdf/2401.00001"])

    def test_a_landing_page_is_not_mistaken_for_a_pdf(self):
        session = mock.Mock()
        session.get.return_value = FakeResponse(None, content=b"<html>Just a moment...</html>")
        self.assertIsNone(pdf_text(session, "https://publisher.example/x.pdf"))


class ComparerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, {"TEST_LLM_KEY": "k"})
        env.start()
        self.addCleanup(env.stop)
        config = TranslationConfig(api_key_env="TEST_LLM_KEY", timeout_seconds=5)
        self.comparer = MethodComparer(config, Path(self.tmp.name), pdf_session=mock.Mock())
        self.comparer.model_reserve = self.comparer.min_call_seconds = 0
        self.asked = []

        self.reply = {"material": {"value": UNREPORTED, "quote": ""},
                      "model": {"value": "Johnson-Cook + MMC",
                                "quote": "calibrated the Johnson-Cook and MMC models"}}

        def ask(work, source, text, timeout=None):
            self.asked.append(work.title)
            if work.title == "slow":
                time.sleep(1.5)
            return self.reply
        self.comparer._ask = ask

    ABSTRACT = "LPBF Ti-6Al-4V was tested at 3000/s. We calibrated the Johnson-Cook and MMC models. " * 4

    def work(self, title, abstract=ABSTRACT):
        return CandidateWork(source="t", identifier=title, title=title, abstract=abstract, doi=f"10.1/{title}")

    def test_a_value_whose_quote_is_not_in_the_text_is_dropped(self):
        """Prompt v1 made Qwen3-8B copy its own examples into a paper lacking them."""
        self.reply = {**self.reply,
                      "calibration": {"value": "DIC + 有限元反演", "quote": "digital image correlation and inverse FE"},
                      "validation": {"value": "与弹道极限速度对比", "quote": ""}}
        row = self.comparer.compare([self.work("e")], {})["rows"][0]
        self.assertEqual(row["fields"]["calibration"], UNREPORTED)
        self.assertEqual(row["fields"]["validation"], UNREPORTED)
        self.assertEqual(row["fields"]["model"], "Johnson-Cook + MMC")
        self.assertEqual(row["dropped"], 2)

    def test_variants_of_unreported_are_normalised(self):
        quote = "LPBF Ti-6Al-4V was tested"
        self.reply = {"strain_rate": {"value": "未报告应变率范围", "quote": quote},
                      "material": {"value": "氢脆钢，未报告制备或热处理状态", "quote": quote}}
        fields = self.comparer.compare([self.work("f")], {})["rows"][0]["fields"]
        self.assertEqual(fields["strain_rate"], UNREPORTED)
        self.assertEqual(fields["material"], "氢脆钢")

    def test_without_full_text_the_abstract_is_used_and_labelled(self):
        result = self.comparer.compare([self.work("a")], {})
        row = result["rows"][0]
        self.assertEqual(row["source"], "摘要")
        self.assertEqual(row["fields"]["model"], "Johnson-Cook + MMC")
        self.assertEqual(row["fields"]["validation"], UNREPORTED, "missing fields must be explicit")

    def test_a_paper_with_nothing_to_read_is_marked_not_guessed(self):
        rows = self.comparer.compare([self.work("b", "too short"), self.work("g", None)], {})["rows"]
        self.assertIsNone(rows[0]["fields"])
        self.assertIn("摘要过短", rows[0]["note"])
        self.assertIn("未提供摘要", rows[1]["note"], "a missing abstract is not a short one")
        self.assertEqual(self.asked, [])

    def test_extractions_are_cached_across_runs(self):
        works = [self.work("c")]
        self.comparer.compare(works, {})
        self.comparer.compare(works, {})
        self.assertEqual(self.asked, ["c"])

    def test_a_slow_paper_is_marked_unfinished_instead_of_holding_the_run(self):
        works = [self.work("fast"), self.work("slow")]
        started = time.monotonic()
        rows = self.comparer.compare(works, {}, deadline_seconds=0.5)["rows"]
        self.assertLess(time.monotonic() - started, 1.4)
        self.assertEqual(rows[0]["fields"]["model"], "Johnson-Cook + MMC")
        self.assertIn("未完成", rows[1]["note"])
        time.sleep(1.2)  # let the abandoned worker finish before the temp dir goes

    def test_a_stuck_worker_does_not_hold_the_process_open(self):
        """Executor workers are joined at exit; review measured 6.4 s for a 1 s deadline."""
        import subprocess
        import sys
        code = (
            "import os, time; os.environ['K']='k'\n"
            "from src.method_compare import MethodComparer\n"
            "from src.models import CandidateWork\n"
            "from src.settings import TranslationConfig\n"
            "c = MethodComparer(TranslationConfig(api_key_env='K'), 'unused-cache')\n"
            "c.model_reserve = c.min_call_seconds = 0\n"
            "c._ask = lambda *a, **k: time.sleep(5)\n"
            "w = CandidateWork(source='t', identifier='x', title='x', abstract='y' * 300)\n"
            "print(c.compare([w], {}, deadline_seconds=0.3)['rows'][0]['note'])\n")
        started = time.monotonic()
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=30,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                             cwd=Path(__file__).resolve().parents[1])
        self.assertLess(time.monotonic() - started, 3.5, out.stderr)
        self.assertIn("未完成", out.stdout)

    def test_disabled_without_a_key(self):
        with mock.patch.dict(os.environ, {"TEST_LLM_KEY": ""}):
            self.assertIsNone(self.comparer.compare([self.work("d")], {}))


class OptionalGuardTests(unittest.TestCase):
    def test_a_failing_auxiliary_feature_returns_its_default(self):
        """The guard itself used to raise NameError on this path, killing the digest."""
        from src.cli import _optional
        with self.assertLogs("src.cli", level="WARNING"):
            self.assertEqual(_optional("broken", lambda: 1 / 0, "fallback"), "fallback")


class RenderTests(unittest.TestCase):
    def render(self, **kw):
        from src.report_html import render_html
        out = Path(tempfile.mkdtemp()) / "report.html"
        render_html([], out, **kw)
        return out.read_text(encoding="utf-8")

    def test_both_sections_render_with_links_back_to_the_cards(self):
        html = self.render(
            method_table={"rows": [
                {"rank": 1, "title": "New A", "source": "全文",
                 "fields": {"material": "LPBF Ti-6Al-4V", "tests": "SHPB", "strain_rate": "3000 s^-1",
                            "temperature": UNREPORTED, "stress_state": UNREPORTED, "model": UNREPORTED,
                            "calibration": UNREPORTED, "simulation": UNREPORTED, "validation": UNREPORTED,
                            "finding": "εf 降低 30%"}},
                {"rank": 2, "title": "New B", "source": "无", "fields": None, "note": "无开放全文，摘要过短"}],
                "model": "Qwen/Qwen3-8B", "full_text": 1, "abstract_only": 0},
            lineage={"papers": [{"rank": 2, "title": "New B", "references": 3, "library_total": 1,
                                 "library": [{"title": "MMC 2010", "year": 2010, "doi": "10.1/m"}]}],
                     "ancestors": [{"title": "Johnson & Cook 1985", "year": 1985, "doi": "10.1/jc",
                                    "url": "https://doi.org/10.1/jc", "cited_by": [1, 2],
                                    "in_library": False, "citations": 9000}],
                     "coverage": {"papers": 2, "with_references": 2, "linked_to_library": 1,
                                  "truncated": False, "incomplete": []}})
        self.assertIn('id="methods"', html)
        self.assertIn('id="lineage"', html)
        self.assertIn('href="#p2"', html)
        self.assertIn("3000 s^-1", html)
        self.assertIn("<i>应变率</i>", html)
        self.assertNotIn("<i>温度</i>", html, "unreported fields are left out of the cell")
        self.assertIn("无开放全文，摘要过短", html)
        self.assertIn("文库没有 · 可补读", html)
        self.assertIn("MMC 2010", html)
        self.assertIn('href="#methods"', html, "the sidebar points to the sections")

    def test_nothing_renders_without_results(self):
        html = self.render()
        self.assertNotIn('id="methods"', html)
        self.assertNotIn('id="lineage"', html)

    def test_the_email_never_carries_them(self):
        from src.digest_email import render_digest
        self.assertNotIn("方法对比", render_digest([]))
        self.assertNotIn("研究脉络", render_digest([]))


if __name__ == "__main__":
    unittest.main()
