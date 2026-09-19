from __future__ import annotations

import argparse
import logging
import json
import tarfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from .build_profile import ProfileBuilder, rederive_profile, verify_profile
from .author_watch import author_news, merge_report_works, work_key
from .citation_watch import CitationDiscovery, merge_candidates
from .watch_history import WatchHistory
from .dedupe import DedupeEngine
from .fetch_new import CandidateFetcher
from .ingest_zotero_api import ZoteroIngestor
from .zotero_local import ingest_local
from .logging_utils import setup_logging
from .metadata_enrich import enrich_ranked_works
from .models import RankedWork
from .notify_email import notify
from .push_to_zotero import ZoteroPusher
from .rss_writer import write_rss, site_url
from .score_rank import WorkRanker
from .settings import Settings, load_settings
from .storage import ProfileStorage
from .report_html import render_html
from .digest_email import render_digest, render_text
from .enrich_zh import enrich_chinese
from .utils import beijing_now
from .research_features import FeedbackModel, RetrievalWarnings, coverage_report, load_feedback, propose_tracking, save_feedback, collaboration_groups
from .problem_ranking import diverse_select
from .network_budget import BudgetSession
from .research_evidence import attach_evidence
from .version_watch import VersionMonitor

load_dotenv()  # Load default .env if present
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_SQLITE = BASE_DIR / "data" / "profile.sqlite"
RSS_PATH = BASE_DIR / "reports" / "feed.xml"


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="ZotWatcher CLI")
    parser.add_argument(
        "command",
        choices=["profile", "watch", "commit-history", "feedback", "notify",
                 "rederive-profile", "verify-profile"],
        help="Command to run",
    )
    parser.add_argument("--base-dir", default=str(BASE_DIR), help="Repository base directory")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--full", action="store_true", help="Full rebuild (profile command)")
    parser.add_argument("--reembed", action="store_true",
                        help="Drop every cached vector first, forcing a full re-encode "
                             "(profile command; repairs pre-2026-09-19 cache drift)")
    parser.add_argument("--weekly", action="store_true", help="Alias for --full in profile command")
    parser.add_argument("--local", action="store_true",
                        help="Build the profile from the local zotero.sqlite instead of the Web API")
    parser.add_argument("--bundle", action="store_true",
                        help="Also pack the profile artifacts into data/profile-bundle.tar.gz")
    parser.add_argument("--prebuilt-profile", action="store_true",
                        help="watch: use the existing profile as-is; skip Zotero sync and re-embedding")
    parser.add_argument("--rss", action="store_true", help="Generate RSS feed (watch command)")
    parser.add_argument("--report", action="store_true", help="Generate HTML report (watch command)")
    parser.add_argument("--top", type=int, default=50, help="Number of top results to keep")
    parser.add_argument("--push", action="store_true", help="Push top items back to Zotero")
    parser.add_argument("--defer-history", action="store_true", help="Commit delivery history separately after publication/email")
    parser.add_argument("--dry-run", action="store_true", help="Build the email without sending it (notify command)")
    parser.add_argument("--doi", help="Paper DOI for explicit feedback")
    parser.add_argument("--rating", choices=["direct", "transferable", "mechanism", "irrelevant", "read", "later", "reading", "reset"])
    parser.add_argument("--scope", default="", help="Apply relevance feedback to one research problem")
    parser.add_argument("--facets", nargs="*", default=[], help="Explicit method facets to learn from")

    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose)
    base_dir = Path(args.base_dir)
    if args.command == "commit-history":
        WatchHistory(base_dir / "data" / "watch-state").commit()
        return
    load_dotenv(base_dir / ".env")
    if args.command == "notify":
        notify(base_dir, dry_run=args.dry_run)
        return
    settings = load_settings(base_dir)
    if args.command == "feedback":
        if not args.doi or not args.rating:
            parser.error("feedback requires --doi and --rating")
        save_feedback(base_dir, {"doi": args.doi, "rating": args.rating, "facets": args.facets, "scope": args.scope}, settings.research)
        return
    if args.command == "verify-profile":
        problems = verify_profile(base_dir, settings)
        for problem in problems:
            logging.getLogger(__name__).error("profile: %s", problem)
        raise SystemExit(1 if problems else 0)
    if args.command == "rederive-profile":
        rederive_profile(base_dir, settings)
        return
    storage = ProfileStorage(base_dir / "data" / "profile.sqlite")

    if args.command == "profile":
        run_profile(base_dir, settings, storage, full=args.full or args.weekly,
                    local=args.local, bundle=args.bundle, reembed=args.reembed)
    elif args.command == "watch":
        run_watch(base_dir, settings, storage, rss=args.rss, report=args.report, top=args.top, push=args.push,
                  defer_history=args.defer_history, prebuilt_profile=args.prebuilt_profile)


def run_profile(base_dir: Path, settings: Settings, storage: ProfileStorage, *, full: bool,
                local: bool = False, bundle: bool = False, reembed: bool = False) -> None:
    logger = logging.getLogger(__name__)
    if reembed:
        # Repairs drift that predates the content-hash cache fix: a row edited before
        # that commit already stores the new hash, so it is indistinguishable from a
        # clean one and can only be repaired by re-encoding everything. Costs a full
        # embedding bill, so it is never automatic.
        storage.initialize()
        conn = storage.connect()
        affected = conn.execute(
            "UPDATE items SET embedding=NULL, embedding_signature=NULL, embedding_version=NULL"
            " WHERE embedding IS NOT NULL"
        ).rowcount
        conn.commit()
        logger.warning("Dropped %d cached vectors; every item will be re-encoded", affected)
    if local:
        data_dir = settings.zotero.local.resolved_dir()
        if not data_dir:
            raise SystemExit(
                "profile --local needs zotero.local.data_dir in config/zotero.yaml "
                "(the directory containing zotero.sqlite)."
            )
        stats = ingest_local(storage, data_dir)
        logger.info(
            "Local ingest: %s documents (%s synced, %s local-only)",
            stats.documents, stats.synced, stats.local_only,
        )
    else:
        ingest = ZoteroIngestor(storage, settings)
        stats = ingest.run(full=full)
        logger.info(
            "Ingest stats: fetched=%s updated=%s removed=%s", stats.fetched, stats.updated, stats.removed
        )
    builder = ProfileBuilder(base_dir, storage, settings)
    artifacts = builder.run()
    logger.info(
        "Profile artifacts generated: sqlite=%s faiss=%s json=%s",
        artifacts.sqlite_path,
        artifacts.faiss_path,
        artifacts.profile_json_path,
    )
    if bundle:
        path = write_profile_bundle(base_dir)
        logger.info("Profile bundle: %s (%.1f MB)", path, path.stat().st_size / 1e6)


def write_profile_bundle(base_dir: Path) -> Path:
    """Pack the profile artifacts so CI can consume a locally built profile.

    Shipped as a release asset rather than committed: the FAISS index is tens of
    megabytes and is replaced on every refresh, which would bloat git history.
    """
    data_dir = Path(base_dir) / "data"
    path = data_dir / "profile-bundle.tar.gz"
    members = [
        data_dir / "profile.sqlite",
        data_dir / "faiss.index",
        data_dir / "profile.json",
    ]
    missing = [member.name for member in members if not member.exists()]
    if missing:
        raise SystemExit(f"Cannot bundle profile; missing {', '.join(missing)}")
    with tarfile.open(path, "w:gz") as archive:
        for member in members:
            archive.add(member, arcname=f"data/{member.name}")
    return path


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
    prebuilt_profile: bool = False,
    warning_recorder=None,
) -> None:
    history = WatchHistory(base_dir / "data" / "watch-state")
    config = settings.research
    feedback = FeedbackModel(load_feedback(base_dir, config, history.state.get("feedback_entries", [])), config)
    history.state["feedback_entries"] = [e.model_dump() for e in feedback.entries.values()]
    builder = ProfileBuilder(base_dir, storage, settings)
    builder.feedback_entries = list(feedback.entries.values())
    if prebuilt_profile:
        # The profile was built elsewhere from the full local library. Re-running
        # the Web API sync here would add nothing (it only sees synced items) and
        # rebuilding would re-embed a library this run cannot see in full.
        logging.getLogger(__name__).info(
            "Using the prebuilt profile; skipping Zotero sync and profile rebuild"
        )
        builder.vectorizer.load()
    else:
        ingest = ZoteroIngestor(storage, settings)
        ingest.run(full=False)
        # Always match the index and its ordered evidence mapping to the current library.
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
    _log_score_distribution(ranked, settings.scoring.thresholds)
    proposals = _optional("Author/venue proposals", lambda: propose_tracking(
        ranked, config, settings, history.state, feedback), []) if config.enabled else []
    # Collaboration analysis is off by default now (config/research.yaml). It was never
    # asked for, its value was never demonstrated, and it ran every week regardless.
    groups = _optional("Collaboration analysis", lambda: collaboration_groups(
        history.state, settings), []) if config.enabled and config.collaboration_analysis else []
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
    classic_keys = {work_key(w) for w in classics}
    exploration = [w for w in ranked if w.extra.get("semantic_facets") and work_key(w) not in recent_keys | classic_keys
                   and w.label != "ignore" and w.published and w.published <= now][:config.semantic_backfill_items] if config.enabled else []
    for work in exploration:
        work.extra["report_channel"] = "跨圈方法发现（非近期新作）"
    ranked = recent
    # Kept in full at the owner's explicit instruction ("作者动向我没说要删，甚至让你
    # 好好做"), including the gap-filling and newly-emerging author candidates. Only
    # its ability to break the digest is removed.
    watched = _optional("Author trends", lambda: author_news(ranked, settings.author_watch), [])
    ranked = [work for work in ranked if work.label != "ignore"]
    ranked = _limit_preprints(ranked, max_ratio=0.3)

    baseline = sorted(ranked, key=lambda w: w.extra.get("legacy_score", w.score), reverse=True)[:top or len(ranked)]
    if config.enabled:
        ranked = diverse_select(ranked, top, config, builder.vectorizer)
    elif top:
        ranked = ranked[:top]

    combined = enrich_ranked_works(merge_report_works(merge_report_works(merge_report_works(ranked, watched), classics), exploration), settings,
                                   session=fetcher.session)
    if config.enabled:
        combined = [attach_evidence(work, config) for work in combined]
    enriched_by_key = {work_key(work): work for work in combined}
    ranked = [enriched_by_key[work_key(work)] for work in ranked]
    watched = [enriched_by_key[work_key(work)] for work in watched]
    classics = [enriched_by_key[work_key(work)] for work in classics]
    exploration = [enriched_by_key[work_key(work)] for work in exploration]
    # Chinese title + one-sentence TLDR for everything that will be shown.
    enrich_chinese(combined, settings.translation, base_dir / "data" / "zh-cache.json")

    monitor = VersionMonitor(settings, fetcher.session)
    alerts = _optional("Version monitoring", lambda: monitor.check(
        combined, history.state), []) if config.enabled else []
    retrieval_warnings = list(getattr(warning_recorder, "messages", [])) + discovery.warnings
    raw = getattr(fetcher, "coverage_raw", None)
    coverage = _optional("Coverage diagnostics", lambda: coverage_report(
        config, {"raw": list(raw.values()) if isinstance(raw, dict) else candidates,
                 "topic": merged, "dedup": deduped, "delivered": combined},
        retrieval_warnings, history.state), []) if config.enabled else []
    diagnostics = {"coverage": coverage, "proposals": proposals,
                   "collaboration_groups": groups,
                   "network": fetcher.session.summary() if isinstance(fetcher.session, BudgetSession) else {},
                   "retrieval_warnings": retrieval_warnings, "version_warnings": monitor.warnings,
                   "feedback_count": len(feedback.entries), "evidence_mode": "title_abstract_only"}

    if not combined:
        logging.getLogger(__name__).info("No ranked results available")

    _log_top_results(ranked)

    if rss:
        write_rss([*alerts, *combined], base_dir / "reports" / "feed.xml")
    if report:
        report_date = beijing_now()
        report_name = f"report-{report_date:%Y%m%d}.html"
        # Issue number = how many digests have been published, this one included.
        issue_no = len(list((base_dir / "reports").glob("report-*.html"))) + (
            0 if (base_dir / "reports" / report_name).exists() else 1)
        profile_names = {key: value.get("name", key) for key, value
                         in (ranker.profile.get("problem_profiles") or {}).items()}
        render_html(ranked, base_dir / "reports" / report_name, watched_works=watched,
                    classic_works=classics, coverage_warnings=retrieval_warnings + monitor.warnings,
                    diagnostics=diagnostics, update_works=alerts, exploration_works=exploration,
                    problem_names=profile_names,
                    library_size=f"{ranker.profile.get('item_count', 0)} 篇",
                    window_days=settings.sources.window_days,
                    issue_no=issue_no,
                    feedback_repository=settings.research.feedback_repository,
                    # How many papers back each research direction in the library:
                    # this is why the ranking leans the way it does.
                    library_directions=sorted(
                        ((v.get("name", k), int(v.get("count", 0)))
                         for k, v in (ranker.profile.get("problem_profiles") or {}).items()),
                        key=lambda kv: -kv[1]))
        # The email carries every paper, but one or three lines each; directions,
        # bylines and scores stay on the page, which is why it needs no names map.
        digest_args = dict(issue_no=issue_no)
        for suffix, text in (
            ("html", render_digest(ranked, report_url=site_url(),
                                   feed_url=site_url() + "feed.xml",
                                   extras={"重点作者新作": watched, "经典文献补漏": classics,
                                           "跨圈方法发现": exploration, "版本与更正提醒": alerts},
                                   **digest_args)),
            ("txt", render_text(ranked, report_url=site_url(), **digest_args)),
        ):
            (base_dir / "reports" / f"digest-{report_date:%Y%m%d}.{suffix}").write_text(
                text, encoding="utf-8")
        (base_dir / "reports" / "research-diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        for name, works in (("baseline", baseline), ("candidate", ranked)):
            snapshot = {"ranking": [{"doi": w.doi, "title": w.title, "score": w.score,
                         "facets": w.extra.get("research_facets", [])} for w in works],
                        "description": "同一候选集上的旧语义权重对照，不是完整历史版本回放" if name == "baseline" else "分问题画像与多样性排序",
                        "generated_at": datetime.now(timezone.utc).isoformat()}
            (base_dir / "reports" / f"ranking-{name}.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
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


def _optional(label: str, produce, default):
    """Run an auxiliary feature; its failure must never stop the digest going out.

    The product is one weekly email plus a web page. Author trends, collaboration
    clusters, version alerts and coverage diagnostics are all worth having, and none
    of them is worth a silent Thursday. Made a standing constraint by the owner on
    2026-09-19: "辅助功能失败，不影响主流程" -- a failed translation ships English, a
    failed source reports incomplete coverage, and no analysis may hold up the mail.

    The main path -- fetch, gate, dedupe, rank, render, send, record -- is
    deliberately NOT wrapped: if any of that breaks, the digest is wrong and the run
    should fail loudly rather than deliver something misleading.
    """
    try:
        return produce()
    except Exception as exc:  # noqa: BLE001 - auxiliary, must not propagate
        logger.warning("%s failed (%s: %s); the digest continues without it",
                       label, exc.__class__.__name__, exc)
        return default


def _log_score_distribution(works: list[RankedWork], thresholds) -> None:
    """Log score percentiles so the label thresholds can be set from real data.

    Scores are now bounded in [0, 1]; if almost nothing reaches must_read, or
    everything does, these percentiles say by how much to move the thresholds.
    """
    if not works:
        return
    scores = sorted(work.score for work in works)
    def pct(p: float) -> float:
        return scores[min(len(scores) - 1, int(p * (len(scores) - 1)))]
    logger = logging.getLogger(__name__)
    logger.info(
        "Score distribution over %d ranked works: p50=%.3f p75=%.3f p90=%.3f p99=%.3f max=%.3f "
        "(must_read=%.2f consider=%.2f)",
        len(scores), pct(0.50), pct(0.75), pct(0.90), pct(0.99), scores[-1],
        thresholds.must_read, thresholds.consider,
    )
    labels = Counter(work.label for work in works)
    logger.info("Label counts: %s", dict(labels))


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
