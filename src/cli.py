from __future__ import annotations

import argparse
import logging
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from .build_profile import ProfileBuilder
from .author_watch import author_news, merge_report_works, work_key
from .citation_watch import CitationDiscovery, merge_candidates
from .watch_history import WatchHistory
from .dedupe import DedupeEngine
from .fetch_new import CandidateFetcher
from .ingest_zotero_api import ZoteroIngestor
from .logging_utils import setup_logging
from .metadata_enrich import enrich_ranked_works
from .models import RankedWork
from .push_to_zotero import ZoteroPusher
from .rss_writer import write_rss
from .score_rank import WorkRanker
from .settings import Settings, load_settings
from .storage import ProfileStorage
from .report_html import render_html
from .research_features import FeedbackModel, RetrievalWarnings, coverage_report, load_feedback, propose_tracking, save_feedback
from .research_evidence import attach_evidence
from .version_watch import VersionMonitor

load_dotenv()  # Load default .env if present
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_SQLITE = BASE_DIR / "data" / "profile.sqlite"
RSS_PATH = BASE_DIR / "reports" / "feed.xml"


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="ZotWatcher CLI")
    parser.add_argument("command", choices=["profile", "watch", "commit-history", "feedback"], help="Command to run")
    parser.add_argument("--base-dir", default=str(BASE_DIR), help="Repository base directory")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--full", action="store_true", help="Full rebuild (profile command)")
    parser.add_argument("--weekly", action="store_true", help="Alias for --full in profile command")
    parser.add_argument("--rss", action="store_true", help="Generate RSS feed (watch command)")
    parser.add_argument("--report", action="store_true", help="Generate HTML report (watch command)")
    parser.add_argument("--top", type=int, default=50, help="Number of top results to keep")
    parser.add_argument("--push", action="store_true", help="Push top items back to Zotero")
    parser.add_argument("--defer-history", action="store_true", help="Commit delivery history separately after publication/email")
    parser.add_argument("--doi", help="Paper DOI for explicit feedback")
    parser.add_argument("--rating", choices=["direct", "transferable", "mechanism", "irrelevant", "read"])
    parser.add_argument("--facets", nargs="*", default=[], help="Explicit method facets to learn from")

    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose)
    base_dir = Path(args.base_dir)
    if args.command == "commit-history":
        WatchHistory(base_dir / "data" / "watch-state").commit()
        return
    load_dotenv(base_dir / ".env")
    settings = load_settings(base_dir)
    if args.command == "feedback":
        if not args.doi or not args.rating:
            parser.error("feedback requires --doi and --rating")
        save_feedback(base_dir, {"doi": args.doi, "rating": args.rating, "facets": args.facets}, settings.research)
        return
    storage = ProfileStorage(base_dir / "data" / "profile.sqlite")

    if args.command == "profile":
        run_profile(base_dir, settings, storage, full=args.full or args.weekly)
    elif args.command == "watch":
        run_watch(base_dir, settings, storage, rss=args.rss, report=args.report, top=args.top, push=args.push,
                  defer_history=args.defer_history)


def run_profile(base_dir: Path, settings: Settings, storage: ProfileStorage, *, full: bool) -> None:
    ingest = ZoteroIngestor(storage, settings)
    stats = ingest.run(full=full)
    logging.getLogger(__name__).info(
        "Ingest stats: fetched=%s updated=%s removed=%s", stats.fetched, stats.updated, stats.removed
    )
    builder = ProfileBuilder(base_dir, storage, settings)
    artifacts = builder.run()
    logging.getLogger(__name__).info(
        "Profile artifacts generated: sqlite=%s faiss=%s json=%s",
        artifacts.sqlite_path,
        artifacts.faiss_path,
        artifacts.profile_json_path,
    )


def run_watch(base_dir, settings, storage, **kwargs):
    recorder = RetrievalWarnings()
    root = logging.getLogger()
    root.addHandler(recorder)
    try:
        return _run_watch(base_dir, settings, storage, warning_recorder=recorder, **kwargs)
    finally:
        root.removeHandler(recorder)


def _run_watch(
    base_dir: Path,
    settings: Settings,
    storage: ProfileStorage,
    *,
    rss: bool,
    report: bool,
    top: int,
    push: bool,
    defer_history: bool = False,
    warning_recorder=None,
) -> None:
    history = WatchHistory(base_dir / "data" / "watch-state")
    config = settings.research
    feedback = FeedbackModel(load_feedback(base_dir, config, history.state.get("feedback_entries", [])), config)
    history.state["feedback_entries"] = [e.model_dump() for e in feedback.entries.values()]
    ingest = ZoteroIngestor(storage, settings)
    ingest.run(full=False)
    # Always match the index and its ordered evidence mapping to the current library.
    builder = ProfileBuilder(base_dir, storage, settings)
    builder.run()

    fetcher = CandidateFetcher(settings, base_dir)
    candidates = fetcher.fetch_all()

    dedupe = DedupeEngine(storage)
    filtered = dedupe.filter(candidates)
    ranker = WorkRanker(base_dir, settings, vectorizer=builder.vectorizer)
    preliminary = ranker.rank(filtered)
    if config.enabled:
        preliminary = feedback.apply(preliminary, settings.scoring.thresholds)
    dynamic = [w for w in _filter_recent(preliminary, days=settings.sources.window_days)
               if w.label != "ignore" and w.similarity >= settings.citation_watch.min_similarity]
    discovery = CitationDiscovery(settings, fetcher.session)
    discovered = discovery.fetch(dynamic[:settings.citation_watch.dynamic_seed_count])
    discovered = fetcher._filter_by_topic(discovered)
    merged = discovery.annotate(merge_candidates([*discovered, *filtered]))
    deduped = dedupe.filter(merged)
    ranked = ranker.rank(deduped)
    if config.enabled:
        ranked = feedback.apply(ranked, settings.scoring.thresholds)
    proposals = propose_tracking(ranked, config, settings, history.state, feedback) if config.enabled else []
    ranked = history.filter(ranked)
    if config.enabled:
        ranked = [w for w in ranked if not w.extra.get("feedback_read")]
    recent = _filter_recent(ranked, days=settings.sources.window_days)
    recent_keys = {work_key(w) for w in recent}
    now = datetime.now(timezone.utc)
    classics = [w for w in ranked if work_key(w) not in recent_keys and w.extra.get("referenced_by")
                and w.published and w.published <= now and w.label != "ignore"
                and w.similarity >= settings.citation_watch.min_similarity]
    classics = classics[:settings.citation_watch.max_classic_items] if settings.citation_watch.enabled else []
    for work in classics:
        work.extra["report_channel"] = "经典文献补漏"
    ranked = recent
    watched = author_news(ranked, settings.author_watch)
    ranked = [work for work in ranked if work.label != "ignore"]
    ranked = _limit_preprints(ranked, max_ratio=0.3)

    if top and len(ranked) > top:
        ranked = ranked[:top]

    combined = enrich_ranked_works(merge_report_works(merge_report_works(ranked, watched), classics), settings)
    if config.enabled:
        combined = [attach_evidence(work, config) for work in combined]
    enriched_by_key = {work_key(work): work for work in combined}
    ranked = [enriched_by_key[work_key(work)] for work in ranked]
    watched = [enriched_by_key[work_key(work)] for work in watched]
    classics = [enriched_by_key[work_key(work)] for work in classics]
    monitor = VersionMonitor(settings, fetcher.session)
    alerts = monitor.check(combined, history.state) if config.enabled else []
    retrieval_warnings = list(getattr(warning_recorder, "messages", [])) + discovery.warnings
    raw = getattr(fetcher, "coverage_raw", None)
    coverage = coverage_report(config, {"raw": list(raw.values()) if isinstance(raw, dict) else candidates,
                               "topic": merged, "dedup": deduped, "delivered": combined},
                               retrieval_warnings, history.state) if config.enabled else []
    diagnostics = {"coverage": coverage, "proposals": proposals,
                   "retrieval_warnings": retrieval_warnings, "version_warnings": monitor.warnings,
                   "feedback_count": len(feedback.entries), "evidence_mode": "title_abstract_only"}

    if not combined:
        logging.getLogger(__name__).info("No ranked results available")

    _log_top_results(ranked)

    if rss:
        write_rss([*alerts, *combined], base_dir / "reports" / "feed.xml")
    if report:
        report_date = datetime.now(ZoneInfo("Asia/Shanghai"))
        report_name = f"report-{report_date:%Y%m%d}.html"
        render_html(ranked, base_dir / "reports" / report_name, watched_works=watched,
                    classic_works=classics, coverage_warnings=retrieval_warnings + monitor.warnings,
                    diagnostics=diagnostics, update_works=alerts)
        (base_dir / "reports" / "research-diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    if push:
        ZoteroPusher(settings).push(ranked)
    if rss or report:
        history.stage(combined)
        if not defer_history:
            history.commit()


def _log_top_results(ranked: list[RankedWork]) -> None:
    logger = logging.getLogger(__name__)
    for idx, work in enumerate(ranked[:10], start=1):
        logger.info("%02d | %.3f | %s | %s", idx, work.score, work.label, work.title)


def _filter_recent(ranked: list[RankedWork], *, days: int) -> list[RankedWork]:
    if days <= 0:
        return ranked
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    kept = [work for work in ranked if work.published and cutoff <= work.published <= now]
    removed = len(ranked) - len(kept)
    if removed > 0:
        logging.getLogger(__name__).info("Dropped %d items older than %d days", removed, days)
    return kept


def _limit_preprints(ranked: list[RankedWork], *, max_ratio: float) -> list[RankedWork]:
    if not ranked or max_ratio <= 0:
        return ranked
    preprint_sources = {"arxiv", "biorxiv", "medrxiv"}
    filtered: list[RankedWork] = []
    preprint_count = 0
    for work in ranked:
        source = work.source.lower()
        proposed_total = len(filtered) + 1
        if source in preprint_sources:
            proposed_preprints = preprint_count + 1
            if (proposed_preprints / proposed_total) > max_ratio:
                continue
            preprint_count = proposed_preprints
        filtered.append(work)
    removed = len(ranked) - len(filtered)
    if removed > 0:
        logging.getLogger(__name__).info("Preprint cap removed %d items to respect %.0f%% limit", removed, max_ratio * 100)
    return filtered


if __name__ == "__main__":
    main()
