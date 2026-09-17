"""Bounded Crossref/OpenAlex lifecycle checks, independent of ordinary paper dedupe."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from urllib.parse import quote

import requests

from .http_utils import request_with_retry
from .models import RankedWork
from .research_features import normalize_doi

logger = logging.getLogger(__name__)
RELATIONS = {"is-preprint-of": "预印本已有正式版本关联", "has-version": "发现其他版本关联", "is-version-of": "发现版本关联"}


def event_id(doi, kind, target):
    return hashlib.sha256(json.dumps([normalize_doi(doi), kind, normalize_doi(target)], ensure_ascii=False).encode()).hexdigest()


def metadata_events(doi, record, notices):
    events = []
    for relation, label in RELATIONS.items():
        for link in (record.get("relation") or {}).get(relation, []) or []:
            target = normalize_doi(link.get("id"))
            if link.get("id-type") == "doi" and target.startswith("10."):
                events.append({"kind": label, "target": target, "source": "Crossref版本关联"})
    for notice in notices:
        for update in notice.get("update-to") or []:
            if normalize_doi(update.get("DOI")) == normalize_doi(doi):
                kind = update.get("type", "update")
                names = {"retraction": "撤稿通知", "correction": "更正通知", "erratum": "勘误通知",
                         "withdrawal": "撤回通知", "expression-of-concern": "关注声明"}
                events.append({"kind": names.get(kind, "出版更新：" + kind), "target": normalize_doi(notice.get("DOI")),
                               "source": "Crossref update-to明确关联"})
    return events


class VersionMonitor:
    def __init__(self, settings, session):
        self.settings, self.session = settings, session
        self.warnings = []

    def get_json(self, url, params=None):
        try:
            return request_with_retry(self.session, "GET", url, params=params, timeout=20, attempts=2,
                                      logger=logger, context="Version metadata").json()
        except (requests.RequestException, ValueError):
            warning = "部分版本/更正检查失败，不能将未返回结果视为没有更新。"
            if warning not in self.warnings:
                self.warnings.append(warning)
            return None

    def check(self, works, state):
        config = self.settings.research
        if not config.enabled or not config.version_batch_size:
            return []
        catalog = state.setdefault("catalog", {})
        for seed in self.settings.citation_watch.seeds:
            if seed.doi:
                catalog.setdefault(normalize_doi(seed.doi), {"title": seed.title})
        for work in works:
            if work.doi:
                catalog[normalize_doi(work.doi)] = {"title": work.title}
        checked = state.setdefault("version_checked", {})
        attempted = state.setdefault("version_attempted", {})
        seen = state.setdefault("events_seen", {})
        batch = sorted(catalog, key=lambda doi: (attempted.get(doi, checked.get(doi, "")), doi))[:config.version_batch_size]
        now = datetime.now(timezone.utc)
        alerts = []
        # A single batch checks OpenAlex's retraction flag without fetching any paper body.
        oa = self.get_json("https://api.openalex.org/works", {
            "filter": "doi:" + "|".join("https://doi.org/" + doi for doi in batch), "per-page": 100,
            "select": "doi,is_retracted", "mailto": self.settings.sources.openalex.mailto}) if batch else None
        retracted = {normalize_doi(w.get("doi")) for w in (oa or {}).get("results", []) if w.get("is_retracted")}
        for doi in batch:
            attempted[doi] = now.isoformat()
            record = self.get_json("https://api.crossref.org/works/" + quote(doi, safe=""),
                                   {"mailto": self.settings.sources.crossref.mailto})
            notice_response = self.get_json("https://api.crossref.org/works", {
                "filter": "updates:" + doi, "rows": 20, "mailto": self.settings.sources.crossref.mailto})
            notice_message = (notice_response or {}).get("message", {})
            if notice_message.get("total-results", 0) > 20:
                self.warnings.append("某篇论文的出版更新超过20条，当前通知覆盖被截断。")
            message = (record or {}).get("message", {})
            # Defensively verify singleton identity instead of trusting DOI lookup routing.
            if message and normalize_doi(message.get("DOI")) != doi:
                self.warnings.append("版本元数据DOI不匹配，已跳过该记录。")
                message = {}
            events = metadata_events(doi, message, notice_message.get("items", []))
            if doi in retracted and not any(e["kind"] == "撤稿通知" for e in events):
                events.append({"kind": "数据库标记撤稿，需核对出版通知", "target": doi, "source": "OpenAlex is_retracted"})
            for event in events:
                ident = event_id(doi, event["kind"], event["target"])
                if ident in seen:
                    continue
                seen[ident] = now.isoformat()
                alerts.append(RankedWork(source="publication_update", identifier="zotwatch-update:" + ident,
                    title=event["kind"] + "｜" + catalog[doi]["title"], url="https://doi.org/" + (event["target"] or doi),
                    published=now, score=0, similarity=0, recency_score=0, metric_score=0, author_bonus=0,
                    venue_bonus=0, label="publication_update", extra={"report_channel": "版本与更正提醒",
                    "original_doi": doi, "update_source": event["source"], "update_kind": event["kind"],
                    "observed_at": now.isoformat(), "date_note": "日期为本系统发现日期，并非通知的发表日期"}))
            if record is not None and notice_response is not None and oa is not None:
                checked[doi] = now.isoformat()
        return alerts
