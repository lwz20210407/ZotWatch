"""Resumable, read-only validation of AMiner candidates against the deployed profile."""
import argparse
import hashlib
import json
import multiprocessing
import sqlite3
import time
from pathlib import Path

import requests

from .aminer_policy import screen_aminer_delivery, select_backfill
from .dedupe import DedupeEngine
from .fetch_new import CandidateFetcher
from .research_dossier import read_snapshot
from .research_features import FeedbackModel, load_feedback
from .score_rank import WorkRanker
from .settings import load_settings
from .storage import ProfileStorage
from .vectorizer import EmbeddingError
from .watch_history import WatchHistory
from .utils import atomic_json


def profile_fingerprint(base):
    names = ["data/profile.json", "data/faiss.index", "data/profile.sqlite", "data/watch-state/state.json", "config/feedback.yaml", "data/feedback-issues.json",
             "config/scoring.yaml", "config/research.yaml", "config/embedding.yaml", "config/sources.yaml"]
    return {n: hashlib.sha256((base / n).read_bytes()).hexdigest() for n in names if (base / n).is_file()}


def eligible_candidates(base, works, settings):
    """Read-only topic, frozen-library and sent-history filtering shared by research runs."""
    base = Path(base).resolve()
    gate = object.__new__(CandidateFetcher); gate.settings = settings
    topic = gate._filter_by_topic(works)
    storage = object.__new__(ProfileStorage); storage.path = base / "data/profile.sqlite"
    storage._conn = sqlite3.connect(storage.path.as_uri() + "?mode=ro&immutable=1", uri=True)
    storage._conn.row_factory = sqlite3.Row
    try:
        fresh = DedupeEngine(storage).filter(topic)
    finally:
        storage.close()
    history = WatchHistory(base / "data/watch-state")
    return history.filter(fresh), history, len(topic)


def evaluate_snapshot(base, snapshot, cohort, output, max_new, deadline, cache_only=False):
    from dotenv import load_dotenv
    load_dotenv(base / ".env", override=False)
    started = time.monotonic()
    before = profile_fingerprint(base)
    settings = load_settings(base)
    settings.embedding.timeout_seconds = min(settings.embedding.timeout_seconds, 20)
    works = read_snapshot(snapshot, cohort)
    fresh, history, topic_count = eligible_candidates(base, works, settings)
    ranker = WorkRanker(base, settings, cache_dir=output / "vector-cache")
    cache = ranker.vectorizer
    ready, failed = [], []
    new = 0
    for work in fresh:
        text = work.content_for_embedding(cache.text_separator)
        if cache.cached(text) is None:
            if cache_only or new >= max_new or time.monotonic() - started >= deadline - 5:
                failed.append({"id": work.identifier, "reason": "budget_or_uncached"})
                continue
            new += 1
            try:
                cache.encode([text])
            except (requests.RequestException, EmbeddingError) as exc:
                failed.append({"id": work.identifier, "reason": type(exc).__name__})
                atomic_json(output / "progress.json", {"ready": len(ready), "failed": failed, "new_text_attempts": new})
                continue
            if cache.cached(text) is None:
                failed.append({"id": work.identifier, "reason": "cache_write_failed"})
                continue
        ready.append(work)
        atomic_json(output / "progress.json", {"ready": len(ready), "failed": failed, "new_text_attempts": new})
    # No score is fabricated for missing vectors; no fallback encoder is selected.
    ranked = ranker.rank(ready)
    feedback = FeedbackModel(load_feedback(base, settings.research, history.state.get("feedback_entries", [])), settings.research)
    ranked = feedback.apply(ranked, settings.scoring.thresholds)
    above = [w for w in ranked if w.label != "ignore" and not w.extra.get("feedback_read")]
    allowed, review = screen_aminer_delivery(above, settings.research)
    from datetime import datetime, timezone
    from .cli import _filter_recent
    backfill = select_backfill(allowed, set(), settings, datetime.now(timezone.utc))
    unchanged = before == profile_fingerprint(base)
    result = {"status": "complete" if not failed and unchanged else "partial" if unchanged else "inputs_changed",
        "input_candidates": len(works), "topic_pass": topic_count, "library_history_filtered": len(fresh),
        "scored": len(ranked), "new_text_attempts": new, "failed": failed,
        "above_threshold": len(above), "applicable": len(allowed), "precise_recent": len(_filter_recent(allowed, days=settings.sources.window_days)),
        "review": review, "backfill": [w.model_dump(mode="json") for w in backfill],
        "ranked": [w.model_dump(mode="json") for w in ranked], "cache": cache.stats,
        "frozen_inputs": before, "inputs_unchanged": unchanged,
        "note": "AMiner cohort acceptance, not a full-source A/B test or a human relevance judgment"}
    atomic_json(output / "acceptance.json", result)
    return result


def worker(*args):
    try:
        result = evaluate_snapshot(*args)
        atomic_json(args[3] / "status.json", {"status": result["status"]})
    except Exception as exc:
        atomic_json(args[3] / "status.json", {"status": "failed", "error_type": type(exc).__name__})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--cohort", default="aminer")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-texts", type=int, default=60)
    parser.add_argument("--deadline", type=int, default=180)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.max_new_texts <= 200 or not 10 <= args.deadline <= 600:
        parser.error("max-new-texts must be 0..200 and deadline 10..600")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "status.json", {"status": "running", "result_valid": False})
    process = multiprocessing.get_context("spawn").Process(target=worker,
        args=(args.base_dir.resolve(), args.snapshot, args.cohort, args.output_dir, args.max_new_texts, args.deadline, args.cache_only))
    process.start(); process.join(args.deadline)
    if process.is_alive():
        process.terminate(); process.join(5)
        atomic_json(args.output_dir / "status.json", {"status": "timeout", "result_valid": False, "cache_preserved": True})
    status = json.loads((args.output_dir / "status.json").read_text("utf-8"))
    if status["status"] == "running":
        status = {"status": "worker_failed", "result_valid": False}
        atomic_json(args.output_dir / "status.json", status)
    print(json.dumps(status))
    raise SystemExit(0 if status["status"] == "complete" else 2)


if __name__ == "__main__": main()
