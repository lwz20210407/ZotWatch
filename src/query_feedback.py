"""Explicit-feedback scheduling with a guaranteed rotating exploration position."""
import argparse
import hashlib
import json
from pathlib import Path

from .author_watch import work_key


def query_id(endpoint, params):
    return hashlib.sha256(json.dumps([endpoint, params], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


def schedule(rows, start, quota, preferences):
    if not rows:
        return [], {"feedback_active": False, "exploration_facet": None}
    rotated = [rows[(start + i) % len(rows)] for i in range(len(rows))]
    active = quota > 0 and any(abs(float(preferences.get(row[0], 0))) > 0 for row in rows)
    if not active:
        return rotated, {"feedback_active": False, "exploration_facet": rotated[0][0] if quota else None}
    # First position rotates regardless of preference or novelty early-stop.
    rest = sorted(enumerate(rotated[1:]), key=lambda pair: (-max(-1, min(1, preferences.get(pair[1][0], 0))), pair[0]))
    ordered = [rotated[0]] + [row for _, row in rest]
    return ordered, {"feedback_active": True, "exploration_facet": rotated[0][0],
                     "preferred_facets": [row[0] for row in ordered[1:max(1, quota)]],
                     "weights": {row[0]: round(float(preferences.get(row[0], 0)), 4) for row in rows},
                     "note": "Explicit feedback reorders bounded recommendation slots; no facet is permanently disabled"}


def annotate_counts(summary, works, field):
    result = dict(summary)
    matches = {}
    for work in works:
        for route in work.extra.get("provenance", []):
            ident = route.get("query_id")
            if ident:
                matches.setdefault(ident, set()).add(work_key(work))
    result["queries"] = [{**row, field: len(matches.get(row["query_id"], set()))} for row in summary.get("queries", [])]
    return result


def main(argv=None):
    from .aminer_source import query_plan
    from .research_features import FeedbackModel, load_feedback
    from .settings import load_settings
    from .watch_history import WatchHistory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    settings = load_settings(args.base_dir)
    history = WatchHistory(args.base_dir / "data/watch-state")
    feedback = FeedbackModel(load_feedback(args.base_dir, settings.research, history.state.get("feedback_entries", [])), settings.research)
    rows = [r for r in query_plan(settings) if r[1] == "recommend"]
    cursor = 0
    path = args.base_dir / "data/cache/aminer/rotation.json"
    if path.exists():
        cursor = int(json.loads(path.read_text("utf-8")).get("recommendation_next", 0))
    ordered, explanation = schedule(rows, cursor, settings.sources.aminer.max_recommendation_queries, feedback.preferences)
    result = {"recommendations": [r[0] for r in ordered[:settings.sources.aminer.max_recommendation_queries]],
              "explanation": explanation, "feedback_reasons": feedback.reason_summary(), "network_calls": 0}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "query-plan.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
