"""Render the report and the email digest from fixture data.

Lets a design change be checked in seconds without running the whole weekly
pipeline (which costs a Zotero sync, several hundred API calls and ~13 minutes).
The fixtures deliberately cover the awkward cases: a long author list, a long
abstract that must clamp at a word boundary, a paper with a transfer card, papers
with and without a nearest-library anchor, and every research direction so the
colour assignment can be eyeballed.

    python tools/preview_report.py            # write reports/preview-*.html
    python tools/preview_report.py --open     # and open them in a browser

Chinese titles and TLDRs come from the live model when EMBEDDING_API_KEY is set
(cached in data/zh-cache.json); otherwise the cards simply render English-only.
"""
from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src.digest_email import render_digest  # noqa: E402
from src.enrich_zh import ChineseEnricher  # noqa: E402
from src.models import RankedWork  # noqa: E402
from src.report_html import render_html  # noqa: E402
from src.settings import load_settings  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
ISSUE = "https://github.com/lwz20210407/ZotWatch/issues/new?title=preview"
FEEDBACK = [{"name": n, "url": ISSUE} for n in
            ("直接有用", "方法可迁移", "机制参考", "不相关", "稍后看", "阅读中", "已读", "撤销反馈")]

NAMES = {"fracture": "延性断裂与损伤演化", "inverse": "参数反演与硬化外推",
         "microstructure": "增材组织—缺陷—失效", "regularization": "数值实现与损伤正则化",
         "thermorate": "温度—应变率耦合", "impact": "冲击与结构验证",
         "stress_state": "应力状态依赖塑性"}


def make(i, title, label, facet, authors, venue, days, abstract, cites, extra=None):
    return RankedWork(
        source="openalex", identifier=f"W{i}", doi=f"10.1016/j.ijimpeng.2026.1050{i:02d}",
        title=title, url="https://doi.org/10.1016/j.ijimpeng.2026.105000",
        authors=authors, venue=venue,
        published=datetime.now(timezone.utc) - timedelta(days=days),
        abstract=abstract, metrics={"cited_by": cites},
        score=0.79 - i * 0.03, similarity=0.62, recency_score=0.8, metric_score=0.2,
        author_bonus=0, venue_bonus=1, label=label,
        extra={"primary_problem": facet, "feedback_links": FEEDBACK, "volume": "192",
               "issue": "4", "pages": f"1050{i:02d}-1050{i + 18:02d}", **(extra or {})})


WORKS = [
    make(1, "A stress-state dependent ductile fracture model for Ti-6Al-4V under dynamic loading",
         "must_read", "fracture",
         ["Jian Zhang", "Wei Liu", "M. Rossi", "K. Tanaka", "L. Bertolini", "H. Okada"],
         "International Journal of Impact Engineering", 4,
         "A ductile fracture criterion coupling stress triaxiality and the Lode angle parameter is "
         "proposed for Ti-6Al-4V. Parameters are calibrated from notched tension, shear and "
         "compression tests, then validated against ballistic perforation experiments spanning 200 "
         "to 600 m/s. Predicted residual velocities agree with measurements to within 6 percent, "
         "and the failure modes observed in the recovered targets - petalling at low velocity and "
         "plugging above 420 m/s - are reproduced without any tuning of the damage evolution law.",
         37,
         {"nearest_library_work": {"title": "Ductile fracture of Ti-6Al-4V: experiments, modelling "
                                            "and validation against ballistic perforation"},
          "transfer_cards": [{"topic": "应力状态依赖塑性",
                              "use": "其 η–Lode 联合标定流程可直接搬到你的 TC4 缺口试样。",
                              "verify": "核对负三轴度区间、非比例加载路径与独立验证试样。"}]}),
    make(2, "Inverse identification of post-necking hardening using integrated digital image correlation",
         "must_read", "inverse", ["A. Dupont", "B. Moreau"], "Experimental Mechanics", 11,
         "An integrated DIC framework identifies post-necking hardening of DP800 steel without "
         "Bridgman correction, reducing force prediction error from 9 percent to 2 percent.", 12,
         {"nearest_library_work": {"title": "LS-OPT inverse calibration of Johnson-Cook "
                                            "parameters from notched tensile tests"},
          "date_precision": "month"}),
    make(3, "Adiabatic shear banding in additively manufactured titanium: build orientation and texture effects",
         "consider", "microstructure", ["S. Yoshikawa", "T. Itoh", "R. Mao"], "Acta Materialia", 19,
         "Shear localisation in LPBF Ti-6Al-4V is examined by EBSD and fractography across three "
         "build orientations. Bands nucleate preferentially where prior-beta grains align with the "
         "shear direction, lowering the critical strain by 18 percent relative to wrought material.",
         5, {"cites_seeds": [{"title": "Adiabatic shear localization in titanium alloys", "url": "#"}]}),
    make(4, "Mesh-objective damage regularization for explicit finite element simulation of metal failure",
         "consider", "regularization", ["P. Novak", "J. Svoboda"],
         "Computer Methods in Applied Mechanics and Engineering", 25,
         "A nonlocal damage regularization scheme restores mesh objectivity for element sizes from "
         "0.1 to 1.0 mm in explicit metal failure simulations, at roughly 12 percent extra runtime.",
         3, {"date_precision": "year"}),
    make(5, "Temperature and strain-rate coupling in the thermoviscoplastic response of Ti-6Al-4V",
         "consider", "thermorate", ["Y. Chen", "Q. Wang", "F. Liu"],
         "International Journal of Plasticity", 21,
         "Split-Hopkinson bar tests from 1e-3 to 5000 per second and 20 to 800 celsius show the "
         "Johnson-Cook single power law fails above 400 celsius; a bilinear rate term with a "
         "temperature-dependent exponent reproduces the flow stress within 4 percent.", 9),
    make(6, "Ballistic perforation of thin titanium plates: failure mode transition with target thickness",
         "consider", "impact", ["R. Kumar", "S. Gupta"], "Thin-Walled Structures", 27,
         "Gas-gun experiments on 1 to 6 mm Ti-6Al-4V plates map the transition from petalling to "
         "plugging, with the Wilkins t/d = 0.5 criterion reproducing the boundary for blunt "
         "projectiles but overpredicting it by 20 percent for ogival noses.", 14,
         {"watched_authors": [{"name": "R. Kumar"}]}),
    make(7, "Non-associated flow and anisotropic hardening of rolled titanium sheet",
         "consider", "stress_state", ["M. Ferrand"], "International Journal of Solids and Structures", 9,
         "A non-associated model with separate yield and potential functions captures the r-value "
         "anisotropy of rolled Ti-6Al-4V sheet that associated plasticity cannot reproduce.", 2),
]

# Real counts from the 4399-paper local profile, so the bars are honest.
LIBRARY_DIRECTIONS = [
    ("延性断裂与损伤演化", 1209), ("应力状态依赖塑性", 1161), ("温度—应变率耦合", 1092),
    ("冲击与结构验证", 950), ("增材组织—缺陷—失效", 382),
    ("数值实现与损伤正则化", 157), ("参数反演与硬化外推", 115),
]

DIAGNOSTICS = {
    "coverage": [
        {"facet": "延性断裂与损伤演化", "raw": 264, "topic": 61, "dedup": 24, "delivered": 8,
         "status": "正常", "zero_runs": 0},
        {"facet": "参数反演与硬化外推", "raw": 52, "topic": 9, "dedup": 4, "delivered": 1,
         "status": "覆盖不足", "zero_runs": 3},
        {"facet": "增材组织—缺陷—失效", "raw": 118, "topic": 22, "dedup": 9, "delivered": 2,
         "status": "正常", "zero_runs": 0}],
    "network": {"requests": 312, "cache_hits": 88, "local_openalex_remaining_usd": "0.061"},
    "proposals": [{"kind": "作者候选", "name": "Guozheng Kang", "id": "A5080733637", "count": 4}],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", action="store_true", help="open the results in a browser")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    load_dotenv(BASE / ".env")

    try:
        ChineseEnricher(load_settings(BASE).translation, BASE / "data" / "zh-cache.json").enrich(WORKS)
    except Exception as exc:  # preview must work offline
        print(f"  (跳过中文标题与 TLDR: {exc})")

    web = BASE / "reports" / "preview-web.html"
    mail = BASE / "reports" / "preview-mail.html"
    render_html(WORKS, web, diagnostics=DIAGNOSTICS, problem_names=NAMES,
                library_size="4399 篇", window_days=30,
                library_directions=LIBRARY_DIRECTIONS, issue_no=12,
                coverage_warnings=["OpenAlex 查询轮换未覆盖全部追踪期刊，本轮覆盖不足"])
    mail.write_text(render_digest(WORKS, report_url="https://lwz20210407.github.io/ZotWatch/",
                                  feed_url="https://lwz20210407.github.io/ZotWatch/feed.xml"),
                    encoding="utf-8")

    for path in (web, mail):
        print(f"  {path.relative_to(BASE)}  {path.stat().st_size / 1024:.1f} KB")
        if args.open:
            webbrowser.open(path.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
