"""Measure the topic gate: does it keep the owner's reading and reject what it should?

The gate is a pile of keyword group sets, and it is very easy to tighten it into
rejecting the real library or loosen it into admitting every materials paper. Neither
failure is visible until a digest arrives, a week later. This replays two sets through
the *real* matching code and prints both error rates:

  recall     the 4399-paper Zotero library, which is by definition on topic --
             the gate should keep almost all of it
  precision  a labelled list of off-topic titles, which it should reject or demote

    python tools/check_topic_gate.py                  # measure the working tree
    python tools/check_topic_gate.py --ref HEAD       # measure a git revision
    python tools/check_topic_gate.py --compare HEAD   # print both, side by side

--compare is the useful one: it answers "did my change actually help, and what did
it cost in recall" instead of leaving both to a guess.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.settings import load_settings  # noqa: E402
from src.topic_matching import matches_any, matches_groups, research_priority  # noqa: E402
from src.models import CandidateWork  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
CONFIGS = ("sources.yaml", "scoring.yaml", "research.yaml", "embedding.yaml",
           "authors.yaml", "network.yaml", "email.yaml", "zotero.yaml")

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
]

# On-topic examples that must survive any tightening. If the gate starts rejecting
# these, the fix went too far.
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
    """The keyword half of CandidateFetcher._filter_by_topic, over one config tree."""

    def __init__(self, root: Path) -> None:
        self.settings = load_settings(root)
        src = self.settings.sources
        self.include = [t for t in src.include_keywords if t.strip()]
        self.anchor = [t for t in getattr(src, "mechanics_anchor_keywords", []) if t.strip()]
        self.exclude = [t for t in src.exclude_keywords if t.strip()]
        self.group_sets = [
            [[t for t in g if t.strip()] for g in gs if any(t.strip() for t in g)]
            for gs in src.required_any_group_sets
        ]
        self.group_sets = [gs for gs in self.group_sets if gs]
        self.require_topic_match = src.require_topic_match

    def verdict(self, title: str, abstract: str, semantic: bool = False) -> str:
        """Return 'excluded', 'rejected', or the priority name it would be scored under."""
        haystack = " ".join(p for p in (title, abstract) if p)
        if self.exclude and matches_any(haystack, self.exclude):
            return "excluded"
        # A semantic facet hit skips the group sets, but only if it clears the anchor.
        privileged = semantic and (not self.anchor or matches_any(haystack, self.anchor))
        if not privileged:
            if self.group_sets and not any(matches_groups(haystack, gs) for gs in self.group_sets):
                return "rejected"
            if self.require_topic_match and self.include and not matches_any(haystack, self.include):
                return "rejected"
        work = CandidateWork(source="probe", identifier="probe", title=title, abstract=abstract)
        name, multiplier = research_priority(work, self.settings.scoring)
        return f"{name} x{multiplier}"


def checkout(ref: str) -> Path:
    """Materialise a git revision's config + return a root that load_settings accepts."""
    tmp = Path(tempfile.mkdtemp(prefix="zotwatch-gate-"))
    (tmp / "config").mkdir()
    for name in CONFIGS:
        out = subprocess.run(["git", "show", f"{ref}:config/{name}"], cwd=BASE,
                             capture_output=True)
        if out.returncode == 0:
            (tmp / "config" / name).write_bytes(out.stdout)
    for name in ("data",):
        if (BASE / name).exists():
            (tmp / name).mkdir(exist_ok=True)
    return tmp


def measure(gate: Gate, library_rows) -> dict:
    kept = sum(1 for t, a in library_rows if not gate.verdict(t or "", a or "").startswith(("rejected", "excluded")))
    off_blocked = [t for t, a in OFF_TOPIC if gate.verdict(t, a).startswith(("rejected", "excluded"))]
    off_semantic = [t for t, a in OFF_TOPIC if gate.verdict(t, a, semantic=True).startswith(("rejected", "excluded"))]
    on_kept = [t for t, a in ON_TOPIC if not gate.verdict(t, a).startswith(("rejected", "excluded"))]
    demoted = [gate.verdict(t, a) for t, a in OFF_TOPIC]
    return {
        "library_kept": kept,
        "library_total": len(library_rows),
        "off_blocked": len(off_blocked),
        "off_blocked_semantic": len(off_semantic),
        "on_kept": len(on_kept),
        "on_total": len(ON_TOPIC),
        "verdicts": demoted,
    }


def report(label: str, m: dict) -> None:
    kept, total = m["library_kept"], m["library_total"]
    print(f"\n=== {label} ===")
    print(f"  recall     library kept          {kept}/{total}  {kept/total:.1%}")
    print(f"  precision  off-topic blocked     {m['off_blocked']}/{len(OFF_TOPIC)}"
          f"   (via semantic route: {m['off_blocked_semantic']}/{len(OFF_TOPIC)})")
    print(f"  guard      on-topic kept         {m['on_kept']}/{m['on_total']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", help="measure this git revision instead of the working tree")
    ap.add_argument("--compare", metavar="REF", help="measure REF and the working tree")
    ap.add_argument("--verdicts", action="store_true", help="print the per-title verdict")
    args = ap.parse_args()

    conn = sqlite3.connect(BASE / "data" / "profile.sqlite")
    library = conn.execute("select title, abstract from items").fetchall()

    targets = []
    if args.compare:
        targets.append((f"{args.compare} (before)", checkout(args.compare)))
        targets.append(("working tree (after)", BASE))
    elif args.ref:
        targets.append((args.ref, checkout(args.ref)))
    else:
        targets.append(("working tree", BASE))

    results = []
    for label, root in targets:
        gate = Gate(root)
        m = measure(gate, library)
        results.append((label, m))
        report(label, m)
        if args.verdicts:
            print("  off-topic verdicts:")
            for (t, _), v in zip(OFF_TOPIC, m["verdicts"]):
                print(f"    {v:<28} {t[:78]}")
        if root != BASE:
            shutil.rmtree(root, ignore_errors=True)

    if len(results) == 2:
        (_, before), (_, after) = results
        d_recall = (after["library_kept"] - before["library_kept"]) / before["library_total"]
        print(f"\n  delta: off-topic blocked {before['off_blocked']} -> {after['off_blocked']}"
              f" (semantic {before['off_blocked_semantic']} -> {after['off_blocked_semantic']}),"
              f" library recall {d_recall:+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
