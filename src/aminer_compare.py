"""Side-effect-free candidate comparison against a frozen baseline snapshot.

No profile rebuild, email, publication, Zotero write or delivery-history update.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from .aminer_source import AMinerSource, align_identities
from .author_watch import work_key
from .citation_watch import merge_candidates
from .fetch_new import CandidateFetcher
from .models import CandidateWork
from .research_features import facet_ids
from .settings import load_settings


def write_rankings(base_dir, settings, cohorts, output_dir):
    """Runs in an isolated worker so a slow embedding service has a wall-clock limit."""
    from dotenv import load_dotenv
    from .score_rank import WorkRanker
    import numpy as np
    status = {"status": "complete"}
    try:
        load_dotenv(base_dir / ".env", override=False)
        ranker = WorkRanker(base_dir, settings)
        underlying = ranker.vectorizer
        class MemoizedVectorizer:
            def __init__(self):
                self.text_separator = underlying.text_separator
                self.memo = {}
            def encode(self, texts):
                texts = list(texts)
                missing = list(dict.fromkeys(t for t in texts if t not in self.memo))
                if missing:
                    self.memo.update(zip(missing, underlying.encode(missing)))
                return np.asarray([self.memo[t] for t in texts])
        ranker.vectorizer = MemoizedVectorizer()
        for name in ("baseline", "combined"):
            ranked = ranker.rank(cohorts[name])
            (output_dir / f"ranking-{name}.json").write_text(json.dumps([w.model_dump(mode="json") for w in ranked[:20]], ensure_ascii=False, indent=2), "utf-8")
    except Exception as exc:
        status = {"status": "failed", "error_type": type(exc).__name__}
    (output_dir / "ranking-status.json").write_text(json.dumps(status), "utf-8")


def bounded_rank(base_dir, settings, cohorts, output_dir, timeout):
    import multiprocessing
    status_path = output_dir / "ranking-status.json"
    status_path.write_text(json.dumps({"status": "running"}), "utf-8")
    worker = multiprocessing.get_context("spawn").Process(target=write_rankings, args=(base_dir, settings, cohorts, output_dir))
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        worker.terminate()
        worker.join(5)
        status = {"status": "timeout", "timeout_seconds": timeout}
    else:
        status = json.loads(status_path.read_text("utf-8"))
        if status.get("status") == "running":
            status = {"status": "worker_failed", "exitcode": worker.exitcode}
    status_path.write_text(json.dumps(status), "utf-8")
    return status


def compare(settings, baseline, aminer):
    aminer = align_identities(aminer, baseline)
    gate = object.__new__(CandidateFetcher)
    gate.settings = settings
    old = gate._filter_by_topic(merge_candidates(baseline))
    new = gate._filter_by_topic(merge_candidates(aminer))
    combined = gate._filter_by_topic(merge_candidates([*baseline, *aminer]))
    keys = {work_key(w) for w in old}
    novel = [w for w in new if work_key(w) not in keys]
    stages = {name: [set(facet_ids(w, settings.research)) for w in items]
              for name, items in {"baseline": old, "aminer": new, "combined": combined}.items()}
    summary = {"baseline_topic_candidates": len(old), "aminer_raw_candidates": len(aminer),
        "aminer_topic_candidates": len(new), "aminer_distinct_keys": len(novel),
        "combined_topic_candidates": len(combined),
        "note": "Frozen cached baseline; counts are identity-key differences, not proven recall/precision gains. No library/history filtering or ranking unless explicitly requested.",
        "facets": {f.id: {name: sum(f.id in ids for ids in groups) for name, groups in stages.items()}
                   for f in settings.research.facets}}
    return summary, {"baseline": old, "aminer": new, "combined": combined, "novel": novel}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--facets", nargs="*")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--rank", action="store_true", help="Also compare rankings; may call the configured embedding API")
    parser.add_argument("--rank-timeout", type=int, default=180, help="Total optional ranking worker deadline in seconds")
    parser.add_argument("--resolve-dates", action="store_true", help="Bounded exact DOI lookups using existing OpenAlex credentials")
    args = parser.parse_args(argv)
    if not 1 <= args.rank_timeout <= 600:
        parser.error("--rank-timeout must be 1..600")
    settings = load_settings(args.base_dir)
    cfg = settings.sources.aminer.model_dump()
    cfg.update(enabled=True, mode="shadow")
    if args.max_requests is not None:
        cfg["max_requests"] = args.max_requests
    settings.sources.aminer = type(settings.sources.aminer)(**cfg)
    if args.facets:
        known = {f.id for f in settings.research.facets}
        if set(args.facets) - known:
            parser.error("Unknown research facet")
        settings.research.facets = [f for f in settings.research.facets if f.id in args.facets]
    baseline_path = args.baseline or args.base_dir / "data/cache/candidate_cache.json"
    payload = json.loads(baseline_path.read_text("utf-8"))
    baseline = [CandidateWork(**r) for r in payload.get("candidates", [])]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = AMinerSource(settings, args.output_dir / "api-cache")
    aminer = source.fetch()
    if args.resolve_dates:
        from .network_budget import BudgetSession
        aminer = source.resolve_dates(align_identities(aminer, baseline), BudgetSession(args.output_dir / "network-state", settings.network))
    summary, cohorts = compare(settings, baseline, aminer)
    summary.update(aminer=source.stats, baseline_fetched_at=payload.get("fetched_at"),
                   baseline_path=str(baseline_path.resolve()))
    snapshots = {k: [w.model_dump(mode="json") for w in v] for k, v in cohorts.items()}
    (args.output_dir / "candidates.json").write_text(json.dumps(snapshots, ensure_ascii=False, indent=2), "utf-8")
    with (args.output_dir / "review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "title", "year", "doi", "facets", "url", "relevant", "reason"])
        for w in cohorts["novel"]:
            # Prevent spreadsheet formula execution when opening external titles.
            def safe(v):
                s = str(v or "")
                return "'" + s if s.startswith(("=", "+", "-", "@", "\t", "\r")) else s
            writer.writerow([safe(x) for x in [work_key(w), w.title, w.extra.get("publication_year"), w.doi,
                                             ", ".join(facet_ids(w, settings.research)), w.url, "", ""]])
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    items = "".join("<li><a href='" + html.escape(w.url or "", quote=True) + "'>" + html.escape(w.title) + "</a><p>" + html.escape(w.abstract or "无原始摘要") + "</p></li>" for w in cohorts["novel"])
    (args.output_dir / "review.html").write_text("<!doctype html><meta charset='utf-8'><title>AMiner 候选对照</title><style>body{max-width:1000px;margin:40px auto;font:16px/1.7 sans-serif}li{margin:24px 0}</style><h1>AMiner 补充候选人工复核</h1><p>身份键未出现在旧候选中；未核查已读历史，不代表已证明的新增相关论文。摘要可能为片段。</p><ol>" + items + "</ol>", "utf-8")
    if args.rank:
        summary["ranking_note"] = "Same frozen profile; raw ranker top20 only, not the final delivered digest (feedback/history/diversity not applied)."
        summary["ranking"] = bounded_rank(args.base_dir, settings, cohorts, args.output_dir, args.rank_timeout)
        (args.output_dir / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
