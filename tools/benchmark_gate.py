"""Stratified sampling and evaluation for a hand-labelled topic-gate benchmark.

Replaces two metrics that did not mean what they were called. "The library is on topic
by definition" contradicts the project's own record of ~247 ceramic-armour papers and
some characterisation-led work, so library-retention was never recall; and a rejection
rate over ten hand-picked negatives, tuned against repeatedly, was never precision.

    python tools/benchmark_gate.py --sample 120 --out tests/fixtures/benchmark.tsv
    # the owner fills in the `label` column, then:
    python tools/benchmark_gate.py --evaluate tests/fixtures/benchmark.tsv

Labels and what the gate should do with each are defined in tests/fixtures/benchmark.md.

Sampling is stratified by the gate's own verdict, deliberately: sampling only what the
gate keeps shows half the picture and hides every paper it wrongly killed. Each row also
carries whether an abstract exists, the year and the source, because the interesting
failures are systematic -- "abstract-less items do badly" is actionable, one aggregate
number is not.

Known limitation, stated rather than papered over: this samples the Zotero LIBRARY, not
the candidate stream. The library is real papers the owner has actually read and judged,
which is what makes labelling feasible, but its distribution is not the distribution of
what OpenAlex returns each week. Treat the result as a lower bound on gate quality.
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.fetch_new import CandidateFetcher  # noqa: E402
from src.models import CandidateWork  # noqa: E402
from src.settings import load_settings  # noqa: E402
from src.topic_matching import research_priority  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
LABELS = ("core", "transferable", "mechanism", "offtopic", "insufficient")
# What the gate is expected to do with each label. `insufficient` is excluded from the
# score entirely and reported separately -- forcing a verdict on an item nobody can
# judge is how a benchmark starts measuring its own noise.
SHOULD_KEEP = {"core": True, "transferable": True, "mechanism": True, "offtopic": False}


def gate(settings):
    fetcher = object.__new__(CandidateFetcher)
    fetcher.settings = settings

    def verdict(title: str, abstract: str) -> tuple:
        work = CandidateWork(source="probe", identifier="probe",
                             title=title or "", abstract=abstract or "")
        if not fetcher._filter_by_topic([work]):
            return False, "rejected", 0.0
        name, multiplier = research_priority(work, settings.scoring)
        return True, name, multiplier

    return verdict


def rows_from_library():
    conn = sqlite3.connect(BASE / "data" / "profile.sqlite")
    try:
        return conn.execute(
            "select key, title, abstract, year, doi from items where title != ''"
        ).fetchall()
    finally:
        conn.close()


def do_sample(count: int, out: Path, seed: int) -> int:
    settings = load_settings(BASE)
    verdict = gate(settings)
    strata = defaultdict(list)
    for key, title, abstract, year, doi in rows_from_library():
        kept, name, _ = verdict(title, abstract)
        strata[(kept, bool((abstract or "").strip()))].append(
            (key, title, abstract, year, doi, name))

    print("strata found (gate verdict, has abstract):")
    for stratum, items in sorted(strata.items(), key=lambda kv: str(kv[0])):
        print(f"  kept={str(stratum[0]):<5} abstract={str(stratum[1]):<5} {len(items):5d}")

    rng = random.Random(seed)
    per = max(1, count // max(len(strata), 1))
    picked = []
    for items in strata.values():
        rng.shuffle(items)
        picked.extend(items[:per])
    rng.shuffle(picked)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["label", "key", "gate_kept", "gate_priority", "year",
                         "has_abstract", "doi", "title"])
        for key, title, abstract, year, doi, name in picked:
            kept, _, _ = verdict(title, abstract)
            writer.writerow(["", key, int(kept), name, year or "",
                             int(bool((abstract or "").strip())), doi or "",
                             " ".join((title or "").split())])
    print(f"\nwrote {len(picked)} rows to {out}")
    print(f"fill the `label` column with one of: {', '.join(LABELS)}")
    print("see tests/fixtures/benchmark.md for what each label means")
    return 0


def do_evaluate(path: Path) -> int:
    settings = load_settings(BASE)
    verdict = gate(settings)
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [row for row in csv.DictReader(fh, delimiter="\t")]

    labelled = [row for row in rows if (row.get("label") or "").strip() in LABELS]
    if not labelled:
        print(f"{path} has no labelled rows yet; nothing to evaluate.")
        print(f"fill the `label` column with one of: {', '.join(LABELS)}")
        return 1
    print(f"{len(labelled)} of {len(rows)} rows labelled")

    scored = [row for row in labelled if row["label"] != "insufficient"]
    insufficient = len(labelled) - len(scored)

    def evaluate(subset, name):
        if not subset:
            return
        tp = fp = fn = tn = 0
        for row in subset:
            kept, _, _ = verdict(row["title"], row.get("abstract", ""))
            should = SHOULD_KEEP[row["label"]]
            if should and kept:
                tp += 1
            elif should and not kept:
                fn += 1
            elif not should and kept:
                fp += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        print(f"\n  {name}  (n={len(subset)})")
        print(f"    kept and should      {tp:4d}    missed and should    {fn:4d}")
        print(f"    kept and should not  {fp:4d}    rejected correctly   {tn:4d}")
        print(f"    precision {precision:.1%}   recall {recall:.1%}")

    evaluate(scored, "overall")
    evaluate([r for r in scored if r.get("has_abstract") == "1"], "with an abstract")
    evaluate([r for r in scored if r.get("has_abstract") != "1"], "without an abstract")
    for label in LABELS:
        subset = [r for r in scored if r["label"] == label]
        if subset:
            kept = sum(1 for r in subset
                       if verdict(r["title"], r.get("abstract", ""))[0])
            print(f"\n  label {label:<14} kept {kept}/{len(subset)}")
    if insufficient:
        print(f"\n  {insufficient} rows labelled `insufficient`, excluded from the score")
    print("\n  Note: sampled from the library, not the weekly candidate stream; read as")
    print("  a lower bound. See the limitation in this tool's docstring.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, metavar="N", help="draw N stratified rows")
    ap.add_argument("--out", type=Path, default=BASE / "tests" / "fixtures" / "benchmark.tsv")
    ap.add_argument("--seed", type=int, default=20260919, help="sampling seed, for reproducibility")
    ap.add_argument("--evaluate", type=Path, metavar="TSV", help="score a labelled TSV")
    args = ap.parse_args()

    if args.sample:
        return do_sample(args.sample, args.out, args.seed)
    if args.evaluate:
        return do_evaluate(args.evaluate)
    ap.error("give --sample N or --evaluate TSV")


if __name__ == "__main__":
    raise SystemExit(main())
