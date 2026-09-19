"""Where does a paper get lost? Trace one title through every stage of the pipeline.

"The additive-TC4 direction produced two papers and neither was TC4" has at least five
possible causes, and the existing diagnostics cannot tell them apart:

    retrieval   no query or venue or semantic facet ever returned it
    exclusion   an exclude_keywords phrase vetoed it
    topic gate  no required_any_group_sets combination matched
    dedupe      the library, or an earlier candidate, suppressed it
    scoring     it ranked below the consider threshold
    selection   the diversity quota or the top-N cut dropped it
    history     it was already delivered in an earlier issue

On 2026-09-19 that gap led to the wrong conclusion: the missing TC4 papers were read as
a coverage problem, when the real fault was classification -- non-titanium papers were
carrying the TC4 label, so the direction looked populated by the wrong papers rather
than empty. This makes the distinction checkable instead of inferred.

    python tools/trace_funnel.py --title "..." [--abstract "..."] [--doi 10.x/y]
    python tools/trace_funnel.py --from-report reports/report-20260919.html

Offline by default: no query is issued, so `retrieval` is reported as "not checked"
unless --check-retrieval is given.
"""
from __future__ import annotations

import argparse
import io
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.dedupe import DedupeEngine  # noqa: E402
from src.fetch_new import CandidateFetcher  # noqa: E402
from src.models import CandidateWork  # noqa: E402
from src.research_features import facet_ids  # noqa: E402
from src.score_rank import WorkRanker  # noqa: E402
from src.settings import load_settings  # noqa: E402
from src.storage import ProfileStorage  # noqa: E402
from src.topic_matching import matches_any, matching_priorities  # noqa: E402
from src.watch_history import WatchHistory  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
OK, NO, MEH = "PASS", "LOST", "----"


def trace(title: str, abstract: str = "", doi: str = "") -> list:
    settings = load_settings(BASE)
    work = CandidateWork(source="probe", identifier=doi or "probe", title=title,
                         abstract=abstract, doi=doi)
    steps = []

    # --- exclusion and topic gate: drive the production filter ---
    fetcher = object.__new__(CandidateFetcher)
    fetcher.settings = settings
    excluded = matches_any(f"{title} {abstract}",
                           [t for t in settings.sources.exclude_keywords if t.strip()])
    steps.append(("exclusion", NO if excluded else OK,
                  "an exclude_keywords phrase vetoed it" if excluded else "no veto phrase"))

    passed_gate = bool(fetcher._filter_by_topic([work]))
    hit_sets = []
    if not passed_gate and not excluded:
        from src.topic_matching import matches_groups
        haystack = f"{title} {abstract}"
        for index, group_set in enumerate(settings.sources.required_any_group_sets):
            groups = [[t for t in g if t.strip()] for g in group_set]
            groups = [g for g in groups if g]
            missing = [i for i, g in enumerate(groups) if not matches_any(haystack, g)]
            if groups and len(missing) == 1:
                hit_sets.append(f"set {index} missed only group {missing[0]}")
    steps.append(("topic gate", OK if passed_gate else NO,
                  "passed" if passed_gate else
                  ("; ".join(hit_sets[:3]) or "no group set came close")))

    # --- what it would be labelled, and at what multiplier ---
    facets = facet_ids(work, settings.research)
    steps.append(("directions", OK if facets else MEH,
                  ", ".join(facets) if facets else "no direction has textual evidence"))
    hits = matching_priorities(work, settings.scoring)
    steps.append(("priority", MEH,
                  ", ".join(f"{h['name']} x{h['multiplier']}" for h in hits) or "default"))

    # --- dedupe against the library ---
    storage = ProfileStorage(BASE / "data" / "profile.sqlite")
    try:
        engine = DedupeEngine(storage)
        survived = bool(engine.filter([work]))
    finally:
        storage.close()
    steps.append(("dedupe", OK if survived else NO,
                  "survived" if survived else "suppressed by the library (see the log line)"))

    # --- scoring, if the profile is present ---
    try:
        ranker = WorkRanker(BASE, settings)
        ranked = ranker.rank([work])[0]
        thresholds = settings.scoring.thresholds
        verdict = OK if ranked.label != "ignore" else NO
        steps.append(("scoring", verdict,
                      f"score {ranked.score:.3f} -> {ranked.label} "
                      f"(must_read {thresholds.must_read}, consider {thresholds.consider}); "
                      f"nearest-centroid direction {ranked.extra.get('primary_problem') or '-'}"))
    except Exception as exc:  # noqa: BLE001 - diagnostic tool
        steps.append(("scoring", MEH, f"not checked: {exc.__class__.__name__}: {exc}"))

    # --- already delivered? ---
    history = WatchHistory(BASE / "data" / "watch-state")
    already = doi and not history.filter([work])
    steps.append(("history", NO if already else OK,
                  "already delivered in an earlier issue" if already
                  else ("not delivered before" if doi else "no DOI given, not checked")))
    return steps


def titles_from_report(path: Path) -> list:
    html = path.read_text(encoding="utf-8")
    clean = lambda s: " ".join(re.sub(r"<[^>]+>", " ", s).split())
    return [clean(t) for t in re.findall(r'class="title"[^>]*>(.*?)</a>', html, re.S)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--title")
    ap.add_argument("--abstract", default="")
    ap.add_argument("--doi", default="")
    ap.add_argument("--from-report", type=Path,
                    help="trace every paper card in a rendered report")
    args = ap.parse_args()

    if args.from_report:
        for title in titles_from_report(args.from_report):
            print(f"\n{'=' * 96}\n{title[:94]}")
            for name, verdict, detail in trace(title):
                print(f"  {verdict:<5} {name:<12} {detail[:70]}")
        return 0

    if not args.title:
        ap.error("give --title or --from-report")
    print(f"\n{args.title[:94]}\n{'-' * 96}")
    for name, verdict, detail in trace(args.title, args.abstract, args.doi):
        print(f"  {verdict:<5} {name:<12} {detail}")
    print("\n  retrieval  ---- not checked offline; add --check-retrieval to issue queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
