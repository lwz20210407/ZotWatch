"""Fill in missing abstracts from AMiner for papers already selected for the page.

Elsevier abstracts are often absent from OpenAlex and Crossref alike. On the
2026-10-05 dry run the two top-ranked papers had none: their cards were bare, the
Chinese TLDR had nothing to summarise and the method comparison could not read them.

A free title search finds the AMiner record; the paid detail endpoint (0.01 yuan a
call) returns the full abstract. A record is accepted only when its DOI equals the
paper's, or -- when one side has no DOI -- when the titles are near-identical. A
missed fill costs one bare card; a wrong one puts another paper's abstract under this
title, so the matching errs towards missing.

Paid calls go through ResearchClient, whose ledger reserves each call before it is
made and refuses once the cap is reached; the ledger lives in a per-week directory,
so the cap is per ISO week. Off by default (config/research.yaml) until the hit rate
has been measured with `python -m src.abstract_fill <report.html>`.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import re
from pathlib import Path

from rapidfuzz import fuzz

from .lineage import norm_doi

logger = logging.getLogger(__name__)

MIN_CHARS = 200
TITLE_MATCH = 95  # rapidfuzz ratio on normalised titles, used only without a DOI on one side
# Failures after which further calls are pointless this run. Network and service
# failures included: from the US GitHub runner AMiner often fails outright, and ten
# papers x (two 15 s search attempts + a 25 s detail call) would add minutes.
STOP = {"credential_missing", "credential_or_permission_failure", "credential_permission_or_balance",
        "account_attention_required", "budget_exhausted", "rate_limited", "rate_limit_cooldown",
        "source_stopped", "time_cap", "request_cap", "paid_calls_not_enabled",
        "network_failure", "service_failure"}


def needs_abstract(work, min_chars=MIN_CHARS):
    return len((work.abstract or "").strip()) < min_chars


def _title(value):
    return re.sub(r"[^a-z0-9]+", " ", html.unescape(str(value or "")).lower()).strip()


def match(work, rows):
    """The AMiner row that is this paper, and how that was established."""
    doi = norm_doi(work.doi)
    for row in rows:
        if doi and norm_doi(row.get("doi")) == doi:
            return row, "doi"
    title = _title(work.title)
    for row in rows:
        # Two different DOIs are two different papers, however alike the titles.
        if doi and norm_doi(row.get("doi")):
            continue
        if title and fuzz.ratio(title, _title(row.get("title"))) >= TITLE_MATCH:
            return row, "title"
    return None, ""


def _abstract(result):
    data = result.get("data") if isinstance(result, dict) else None
    if isinstance(data, list):
        data = data[0] if data else {}
    text = data.get("abstract") if isinstance(data, dict) else ""
    return re.sub(r"\s+", " ", html.unescape(str(text or ""))).strip()


def fill_abstracts(works, search, detail, *, limit=10, min_chars=MIN_CHARS):
    """Mutates works in place; returns a report fit for the diagnostics block.

    `search(title) -> rows` and `detail(aminer_id) -> result` are injected so the
    weekly run, the measurement command and the tests share one decision path.
    """
    report = {"missing": 0, "attempted": 0, "matched": 0, "filled": 0, "items": [], "failures": {}}
    stopped = ""
    for work in works:
        if not needs_abstract(work, min_chars):
            continue
        report["missing"] += 1
        if stopped or report["attempted"] >= limit:
            continue
        report["attempted"] += 1
        item = {"title": work.title[:120], "doi": norm_doi(work.doi), "result": ""}
        try:
            row, how = match(work, search(work.title))
            if not row:
                item["result"] = "no_match"
            else:
                report["matched"] += 1
                abstract = _abstract(detail(str(row.get("id") or "")))
                if len(abstract) >= min_chars:
                    work.abstract = abstract
                    work.extra["abstract_filled_from"] = "AMiner"
                    report["filled"] += 1
                    item["result"] = f"filled_by_{how}"
                else:
                    item["result"] = "matched_without_abstract"
        except Exception as exc:  # controlled categories from both clients; anything else by type
            category = str(exc) if str(exc) and " " not in str(exc) else type(exc).__name__
            item["result"] = category
            report["failures"][category] = report["failures"].get(category, 0) + 1
            if category in STOP:
                stopped = category
        report["items"].append(item)
    report["stopped"] = stopped
    return report


def weekly_fill(works, settings, base_dir, week):
    """The weekly path: free search through the shadow client's budget, paid detail
    through a ResearchClient whose ledger directory is this ISO week's."""
    from .aminer_advanced import ResearchClient
    from .aminer_client import AMinerClient

    cfg = settings.research
    searcher = AMinerClient(settings.sources.aminer, Path(base_dir) / "data" / "cache" / "aminer")
    searcher.check_cooldown()
    payer = ResearchClient(Path(base_dir) / "data" / "cache" / "aminer-abstracts" / week,
                           budget_yuan=f"{cfg.aminer_abstract_budget_yuan:.2f}", allow_paid=True)
    report = fill_abstracts(
        works,
        search=lambda title: searcher.query("search", {"title": title, "page": 1, "size": 5}),
        detail=lambda ident: payer.query("paper_detail", {"id": ident}),
        limit=cfg.aminer_abstract_limit)
    try:
        report["spent_yuan_this_week"] = payer.summary().get("estimated_reserved_yuan")
    except Exception:  # the ledger summary is informational only
        pass
    return report


def _works_without_abstract(report_html):
    """Cards on a published page whose abstract block is empty: (title, doi)."""
    from .models import CandidateWork

    page = Path(report_html).read_text(encoding="utf-8")
    works = []
    for card in re.findall(r'<article id="p\d+".*?</article>', page, re.S):
        if re.search(r'<div class="abs">', card):
            continue
        title = re.search(r'<a class="title"[^>]*>(.*?)(?:<span class="yr">|</a>)', card, re.S)
        doi = re.search(r'href="https://doi\.org/([^"]+)"', card)
        if title:
            works.append(CandidateWork(source="report", identifier=title.group(1), doi=doi.group(1) if doi else None,
                                       title=re.sub(r"\s+", " ", html.unescape(title.group(1))).strip()))
    return works


def main(argv=None):
    """Measure the hit rate on a published page's bare cards. Spends at most --budget."""
    from dotenv import load_dotenv

    from .settings import load_settings

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--budget", default="0.20", help="yuan, 0.01-5.00; ledger in data/cache/aminer-abstracts/measure")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    load_dotenv(args.base_dir / ".env")
    from .aminer_advanced import ResearchClient
    from .aminer_client import AMinerClient

    settings = load_settings(args.base_dir)
    works = _works_without_abstract(args.report)
    searcher = AMinerClient(settings.sources.aminer, args.base_dir / "data" / "cache" / "aminer")
    payer = ResearchClient(args.base_dir / "data" / "cache" / "aminer-abstracts" / "measure",
                           budget_yuan=args.budget, allow_paid=True)
    report = fill_abstracts(works, search=lambda t: searcher.query("search", {"title": t, "page": 1, "size": 5}),
                            detail=lambda i: payer.query("paper_detail", {"id": i}), limit=args.limit)
    report["spent_yuan_total"] = payer.summary().get("estimated_reserved_yuan")
    report["filled_preview"] = [{"title": w.title[:80], "abstract": w.abstract[:160]}
                                for w in works if w.extra.get("abstract_filled_from")]
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
