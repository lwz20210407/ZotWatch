"""Evidence-based routing for AMiner discoveries; does not change relevance scores."""
from __future__ import annotations

import re
from collections import Counter

from .author_watch import work_key
from .research_features import facet_ids
from .topic_matching import matches_any


def aminer_only(work):
    # Metadata fetched by exact DOI preserves the original source='aminer'.
    # A candidate independently found by an existing source keeps its original lane.
    return work.source == "aminer" and bool(work.extra.get("aminer_id"))


def applicability(work, research):
    title = work.title
    if re.match(r"^\s*(?:preface\b|editorial\b|introduction to (?:the )?special issue\b|编者按|序言)", title, re.I):
        return {"status": "exclude", "reason": "前言或编辑内容，不作为方法研究论文推送"}
    nonmetal = ["calcareous sand", "rock acoustic emission", "concrete-like", "concrete subjected",
                "elastomers", "geotechnical", "ground reaction"]
    if matches_any(title, nonmetal):
        return {"status": "exclude", "reason": "题名明确指向砂土、岩石、混凝土或弹性体，未证实适用于金属研究链"}
    conditions = []
    if matches_any(title, ["hydrogen", "氢脆", "氢致"]):
        conditions.append("依赖氢环境，需核对失效机制及标定条件能否迁移")
    if matches_any(title, ["machining", "metal cutting", "for turning", "turning process", "cutting force", "切削", "车削"]):
        conditions.append("切削/加工工况，需核对热力边界、应变率及反演参数的适用范围")
    if matches_any(title, ["metal foam", "sandwich plates", "sandwich plate", "triply periodic minimal surfaces"]):
        conditions.append("胞状/夹层结构，需区分结构效应与实体材料本构和断裂机制")
    if work.extra.get("work_type") in {"software", "dataset"} or matches_any(title, ["material script"]):
        conditions.append("软件或数据资源，需要单独核对可用性，不能等同于已验证的研究方法")
    if conditions:
        return {"status": "conditional", "reason": "；".join(conditions)}
    if not work.abstract:
        return {"status": "insufficient_evidence", "reason": "缺原始摘要或摘要片段，保留待补证，不使用生成概述替代"}
    if not facet_ids(work, research):
        return {"status": "insufficient_evidence", "reason": "题名摘要未给出现有研究方向的可核查线索"}
    return {"status": "suitable", "reason": "题名摘要提供研究方向线索；参数和工况仍需全文核对"}


def screen_aminer_delivery(works, research):
    """Hold uncertain AMiner-only items for review, preserving other sources and scores."""
    allowed, held, counts = [], [], Counter()
    for work in works:
        if not aminer_only(work):
            allowed.append(work)
            continue
        assessment = applicability(work, research)
        counts[assessment["status"]] += 1
        tagged = work.model_copy(update={"extra": {**work.extra, "aminer_applicability": assessment}})
        if assessment["status"] == "suitable":
            allowed.append(tagged)
        else:
            held.append({"title": work.title, "doi": work.doi, "url": work.url,
                         "score": getattr(work, "score", None), "label": getattr(work, "label", None), **assessment})
    return allowed, {"counts": dict(counts), "held_count": len(held), "review": held[:50]}


def recent_delivery(works, config):
    return [w for w in works if not aminer_only(w) or config.delivery == "recent_and_backfill"]


def select_backfill(works, excluded, settings, now):
    selected = []
    for work in works:
        if not aminer_only(work) or work_key(work) in excluded:
            continue
        if work.label == "ignore" or work.similarity < settings.citation_watch.min_similarity:
            continue
        if work.published and work.published > now:
            continue
        if (work.extra.get("publication_year") or 0) > now.year:
            continue
        if applicability(work, settings.research)["status"] != "suitable":
            continue
        selected.append(work.model_copy(update={"extra": {**work.extra,
            "report_channel": "AMiner 方法补漏（不代表近期新发表）"}}))
    return selected[:settings.sources.aminer.backfill_items]
