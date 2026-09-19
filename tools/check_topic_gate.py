"""Measure the topic gate by driving the production gate, not a copy of it.

The gate is a pile of keyword group sets, and it is easy to tighten it into rejecting
the real library or loosen it into admitting every materials paper. Neither failure is
visible until a digest arrives a week later, so this replays labelled sets through it
and prints the error rates.

    python tools/check_topic_gate.py                  # measure the working tree
    python tools/check_topic_gate.py --compare HEAD   # working tree vs a revision
    python tools/check_topic_gate.py --verdicts        # per-title detail
    python tools/check_topic_gate.py --json            # machine-readable

Two things this got wrong before, both found by external review on 2026-09-19:

1. It reimplemented the keyword half of CandidateFetcher._filter_by_topic, and the copy
   was not the gate -- no retraction filter, no work_type filter, no watched-author or
   citation route. The numbers it produced were quoted in commit messages and in
   docs/DESIGN.md as evidence a filtering change worked, while measuring a
   reimplementation. Demonstrated with one title: "RETRACTED: TC4 ductile fracture"
   scored TC4核心研究 x1.0 here and was rejected outright in production. A harness that
   duplicates the logic it tests lets both agree and both be wrong.

2. --compare swapped only the CONFIG. The gate lives in src/, so code changes were
   attributed to config, and the reported before/after partly did not come from the
   thing being measured. It now uses a real git worktree, pinning code and config
   together, and copies THIS harness in so both sides are measured the same way.

What the numbers do and do not mean -- also a review finding, and the reason the
metrics below are not called precision and recall:

  library_kept    the share of the owner's own library the gate keeps. NOT recall. The
                  library is not uniformly on topic: it holds ~247 ceramic-armour papers
                  and some characterisation-led work, both of which the gate SHOULD
                  reject. Read it as a regression signal -- a sudden drop means a change
                  cut deeper than intended.
  off_blocked     rejection rate on a small hand-labelled negative set. NOT precision,
                  and with a set this size, repeatedly tuning rules against it overfits.
                  It answers "did the specific thing I was fixing get fixed".
  guards_kept     a small positive set that must survive any tightening.

Honest benchmarking needs a real labelled sample; see docs/REMEDIATION-PLAN.md 3.3.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fetch_new import CandidateFetcher  # noqa: E402
from src.models import CandidateWork  # noqa: E402
from src.settings import load_settings  # noqa: E402
from src.topic_matching import research_priority  # noqa: E402

BASE = Path(__file__).resolve().parent.parent

# Off-topic examples. Real titles: the first block is what the owner rejected on
# 2026-09-19 ("太偏材料不偏力学"), the second is characterisation-led work drawn from
# his own library audit, the third is the ceramic gap the armour phrases missed.
OFF_TOPIC: List[Tuple[str, str]] = [
    ("Strain-rate-driven dynamic recrystallization and hetero-deformation response in a dual-phase HEA at elevated temperatures",
     "The dynamic recrystallization behaviour and hetero-deformation induced strengthening of a dual-phase high-entropy alloy are examined by EBSD and TEM across temperatures. Grain refinement improves the mechanical properties and tensile strength."),
    ("Direct current thermo-mechanical testing: principles, uncertainty hierarchy, and its role in advanced materials characterisation",
     "This review surveys direct current thermo-mechanical testing as a materials characterisation technique, covering principles, instrumentation and the uncertainty hierarchy of the measurement."),
    ("Superior resistance to cyclic creep in a gradient structured steel",
     "A gradient structured steel shows superior cyclic creep resistance. The microstructure evolution was characterised by EBSD and TEM, and the mechanical properties correlated with grain size gradients."),
    ("Atomic faulting induced exceptional cryogenic strain hardening in gradient cell-structured alloy",
     "Atomic faulting mechanisms observed by transmission electron microscopy explain exceptional cryogenic strain hardening in a gradient cell-structured alloy, improving the strength-ductility synergy."),
    ("Additive manufactured high entropy alloys: A review of the microstructure and properties",
     "This review summarises the microstructure and mechanical properties of additively manufactured high entropy alloys, including phase transformation, texture evolution and grain refinement."),
    ("Engineering fine grains, dislocations and precipitates for enhancing the strength of TiB2-modified CoCrFeNi",
     "Grain refinement, dislocation engineering and precipitation behaviour raise the tensile strength and elongation of a TiB2-modified CoCrFeNi alloy, characterised by SEM, EBSD and XRD."),
    ("EBSD characterization of microstructural evolution in laser powder bed fusion Ti-6Al-4V after heat treatment",
     "Electron backscatter diffraction reveals the microstructure evolution and crystallographic texture of LPBF Ti-6Al-4V across heat treatment conditions. Mechanical properties are reported."),
    ("A neural network-based constitutive model for fiber-reinforced ceramic matrix composites",
     "A neural network constitutive model is trained for fiber-reinforced ceramic matrix composites and predicts the nonlinear stress strain response under multiaxial loading."),
    ("On Damage Degree and Evolution of Mechanical and Electromagnetic Properties for SiCf/Si3N4 Composite",
     "Damage evolution of a SiCf/Si3N4 ceramic matrix composite is tracked through mechanical and electromagnetic property degradation under load."),
    ("Finite element modeling of porous polymer pipeline coating using X-ray micro computed tomography",
     "X-ray micro computed tomography informs a finite element model of a porous polymer pipeline coating; mechanical properties are homogenised."),
    # Only reachable through the production gate, which the old copy of the logic did
    # not model. Kept as a standing check that this harness still drives the real path.
    ("RETRACTED: TC4 ductile fracture under dynamic loading",
     "Stress triaxiality and Lode dependent fracture criterion for TC4, calibrated and validated."),
]

# On-topic examples that must survive any tightening. If the gate starts rejecting
# these, the change went too far.
ON_TOPIC: List[Tuple[str, str]] = [
    ("A stress-state dependent ductile fracture model for Ti-6Al-4V under dynamic loading",
     "A ductile fracture criterion coupling stress triaxiality and the Lode angle parameter is proposed for Ti-6Al-4V, calibrated from notched tension, shear and compression, and validated against ballistic perforation."),
    ("Build orientation effects on the dynamic tensile behaviour and fracture of LPBF Ti-6Al-4V",
     "Split Hopkinson tension bar tests on laser powder bed fusion Ti-6Al-4V across three build orientations show how lack-of-fusion defects control the fracture strain and the flow stress at high strain rate."),
    ("Inverse identification of post-necking hardening using integrated digital image correlation",
     "An integrated DIC framework identifies the post-necking hardening of DP800 steel without Bridgman correction, reducing the force prediction error from 9 percent to 2 percent."),
    ("Mesh-objective damage regularization for explicit finite element simulation of metal failure",
     "A nonlocal damage regularization scheme restores mesh objectivity for element sizes from 0.1 to 1.0 mm in explicit metal failure simulations with element deletion."),
    ("Fractographic evidence of void coalescence in ductile fracture of Ti-6Al-4V",
     "Fractography and SEM of broken notched specimens quantify void nucleation, growth and coalescence, and the measured fracture strain is compared with a Gurson model prediction."),
    ("Temperature and strain-rate coupling in the thermoviscoplastic response of Ti-6Al-4V",
     "Split-Hopkinson bar tests from 1e-3 to 5000 per second and 20 to 800 celsius show the Johnson-Cook single power law fails above 400 celsius; a bilinear rate term reproduces the flow stress."),
]


class Gate:
    """Drives the production gate. Does not reimplement it."""

    def __init__(self, root: Path) -> None:
        self.settings = load_settings(root)
        # Bypass __init__: a real CandidateFetcher opens sessions and reads state a pure
        # keyword probe has no use for. _filter_by_topic only needs .settings.
        self.fetcher = object.__new__(CandidateFetcher)
        self.fetcher.settings = self.settings

    def verdict(self, title: str, abstract: str, semantic: bool = False) -> str:
        """'rejected', or the priority name and multiplier the work would score under.

        'rejected' covers every reason production drops a candidate -- retraction, work
        type and the exclusion list included -- not only the keyword groups.
        """
        extra = {"semantic_facets": ["probe"]} if semantic else {}
        work = CandidateWork(source="probe", identifier="probe", title=title,
                             abstract=abstract, extra=extra)
        if not self.fetcher._filter_by_topic([work]):
            return "rejected"
        name, multiplier = research_priority(work, self.settings.scoring)
        return f"{name} x{multiplier}"


def measure(root: Path) -> dict:
    gate = Gate(root)
    conn = sqlite3.connect(root / "data" / "profile.sqlite")
    try:
        library = conn.execute("select title, abstract from items").fetchall()
    finally:
        conn.close()

    rejected = lambda verdict: verdict == "rejected"
    kept = sum(1 for t, a in library if not rejected(gate.verdict(t or "", a or "")))
    verdicts = [gate.verdict(t, a) for t, a in OFF_TOPIC]
    semantic = [gate.verdict(t, a, semantic=True) for t, a in OFF_TOPIC]
    return {
        "library_kept": kept,
        "library_total": len(library),
        "off_blocked": sum(1 for v in verdicts if rejected(v)),
        "off_blocked_semantic": sum(1 for v in semantic if rejected(v)),
        "off_total": len(OFF_TOPIC),
        "guards_kept": sum(1 for t, a in ON_TOPIC if not rejected(gate.verdict(t, a))),
        "guards_total": len(ON_TOPIC),
        "verdicts": verdicts,
    }


def checkout(ref: str) -> Path:
    """Materialise a git revision as a real worktree: its code AND its config."""
    tmp = Path(tempfile.mkdtemp(prefix="zotwatch-gate-"))
    worktree = tmp / "tree"
    out = subprocess.run(["git", "worktree", "add", "--detach", str(worktree), ref],
                         cwd=BASE, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"could not create a worktree for {ref!r}:\n{out.stderr}")
    # data/ is gitignored, so the library is absent from the worktree. Hard-link it
    # rather than copying 36 MB.
    (worktree / "data").mkdir(exist_ok=True)
    for name in ("profile.sqlite", "journal_metrics.csv"):
        source = BASE / "data" / name
        if not source.exists():
            continue
        try:
            os.link(source, worktree / "data" / name)
        except OSError:
            shutil.copy2(source, worktree / "data" / name)
    # Measure both sides with THIS harness, so the only difference is the code and
    # config under test rather than the measurement itself.
    shutil.copy2(Path(__file__).resolve(), worktree / "tools" / Path(__file__).name)
    return worktree


def release(worktree: Path) -> None:
    subprocess.run(["git", "worktree", "remove", "--force", str(worktree)],
                   cwd=BASE, capture_output=True)
    shutil.rmtree(worktree.parent, ignore_errors=True)


def measure_in(worktree: Path) -> dict:
    """Run the measurement inside a worktree, so its src/ is the code under test."""
    out = subprocess.run(
        [sys.executable, "-B", str(worktree / "tools" / Path(__file__).name), "--json"],
        cwd=worktree, capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise SystemExit(f"measurement failed inside {worktree}:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout)


def report(label: str, m: dict) -> None:
    kept, total = m["library_kept"], m["library_total"]
    print(f"\n=== {label} ===")
    print(f"  library_kept   {kept}/{total}  {kept / max(total, 1):.1%}"
          f"   (regression signal, NOT recall)")
    print(f"  off_blocked    {m['off_blocked']}/{m['off_total']}"
          f"   (via the semantic route: {m['off_blocked_semantic']}/{m['off_total']})")
    print(f"  guards_kept    {m['guards_kept']}/{m['guards_total']}"
          f"   (must stay at {m['guards_total']}/{m['guards_total']})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", metavar="REF", help="measure REF and the working tree")
    ap.add_argument("--verdicts", action="store_true", help="print the per-title verdict")
    ap.add_argument("--json", action="store_true", help="emit the working-tree result as JSON")
    args = ap.parse_args()

    here = measure(BASE)
    if args.json:
        print(json.dumps(here))
        return 0

    if args.compare:
        worktree = checkout(args.compare)
        try:
            before = measure_in(worktree)
        finally:
            release(worktree)
        report(f"{args.compare} (before)", before)
        report("working tree (after)", here)
        delta = (here["library_kept"] - before["library_kept"]) / max(before["library_total"], 1)
        print(f"\n  delta: off_blocked {before['off_blocked']} -> {here['off_blocked']}"
              f" (semantic {before['off_blocked_semantic']} -> {here['off_blocked_semantic']}),"
              f" library_kept {delta:+.1%}")
    else:
        report("working tree", here)

    if args.verdicts:
        print("\n  off-topic verdicts:")
        for (title, _), verdict in zip(OFF_TOPIC, here["verdicts"]):
            print(f"    {verdict:<26} {title[:74]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
