"""Transparent research facets, explicit feedback, diagnostics and opt-in proposals."""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Literal

from .author_watch import work_key
from .topic_matching import matches_any

logger = logging.getLogger(__name__)
RATINGS = {"direct": 1.0, "transferable": 0.8, "mechanism": 0.6, "irrelevant": -1.0, "read": 0.0}
RATING_NAMES = {"direct": "直接有用", "transferable": "方法可迁移", "mechanism": "机制参考", "irrelevant": "不相关", "read": "已读"}


def normalize_doi(value):
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", (value or "").strip(), flags=re.I).casefold()


class FeedbackEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doi: str
    rating: Literal["direct", "transferable", "mechanism", "irrelevant", "read"]
    facets: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("doi")
    @classmethod
    def doi_valid(cls, value):
        value = normalize_doi(value)
        if not re.fullmatch(r"10\.\d{4,9}/[^\s\"<>]+", value) or len(value) > 250:
            raise ValueError("Feedback requires a valid DOI")
        return value


def facet_ids(work, config):
    text = work.title + " " + (work.abstract or "")
    return [f.id for f in config.facets if matches_any(text, f.terms)]


def load_feedback(base_dir, config, saved=()):
    """Only repository owner's explicit JSON feedback is accepted, never issue instructions."""
    entries = {entry.doi: entry for entry in (FeedbackEntry.model_validate(row) for row in saved)}
    issues_path = Path(base_dir) / "data" / "feedback-issues.json"
    allowed = {f.id for f in config.facets}
    if issues_path.exists():
        pages = json.loads(issues_path.read_text(encoding="utf-8"))
        issues = [issue for page in pages for issue in page] if pages and isinstance(pages[0], list) else pages
        for issue in sorted(issues, key=lambda i: i.get("updated_at", "")):
            if issue.get("pull_request") or issue.get("user", {}).get("login", "").casefold() != config.feedback_owner.casefold():
                continue
            match = re.search(r"<!-- zotwatch-feedback-v1 -->\s*```json\s*(.*?)\s*```", issue.get("body") or "", re.S)
            if not match:
                continue
            try:
                entry = FeedbackEntry.model_validate_json(match.group(1))
                if not set(entry.facets) <= allowed:
                    raise ValueError("Unknown feedback facet")
                entries[entry.doi] = entry
            except ValueError:
                logger.warning("Ignored malformed feedback in issue #%s", issue.get("number"))
    path = Path(base_dir) / "config" / "feedback.yaml"
    if path.exists():
        for row in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("entries", []):
            entry = FeedbackEntry.model_validate(row)
            if not set(entry.facets) <= allowed:
                raise ValueError("Unknown facet in local feedback configuration")
            entries[entry.doi] = entry  # Explicit local configuration takes precedence.
    return list(entries.values())


def save_feedback(base_dir, entry, config):
    entry = FeedbackEntry.model_validate(entry)
    if not set(entry.facets) <= {f.id for f in config.facets}:
        raise ValueError("Unknown feedback facet")
    path = Path(base_dir) / "config" / "feedback.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {"entries": []}
    entries = [r for r in data.get("entries", []) if normalize_doi(r["doi"]) != entry.doi]
    path.write_text(yaml.safe_dump({"entries": [*entries, entry.model_dump()]}, allow_unicode=True, sort_keys=False), encoding="utf-8")


class FeedbackModel:
    def __init__(self, entries, config):
        self.entries = {e.doi: e for e in entries}
        self.config = config
        votes = defaultdict(list)
        for entry in entries:
            if entry.rating == "read":
                continue
            for facet in set(entry.facets):
                votes[facet].append(RATINGS[entry.rating])
        # Two neutral pseudo-observations: a single click cannot dominate the profile.
        self.preferences = {f: sum(v) / (len(v) + 2) for f, v in votes.items()}

    def apply(self, works, thresholds):
        result = []
        for work in works:
            entry = self.entries.get(normalize_doi(work.doi))
            facets = facet_ids(work, self.config)
            pref = sum(self.preferences.get(f, 0) for f in facets) / max(1, len(facets))
            if entry and entry.rating != "read":
                pref = (pref + RATINGS[entry.rating]) / 2
            delta = max(-1, min(1, pref)) * self.config.feedback_max_adjustment
            score = work.score + delta
            label = "must_read" if score >= thresholds.must_read else "consider" if score >= thresholds.consider else "ignore"
            extra = {**work.extra, "research_facets": facets, "feedback_adjustment": delta,
                     "feedback_read": bool(entry and entry.rating == "read")}
            result.append(work.model_copy(update={"score": score, "label": label, "extra": extra}))
        return sorted(result, key=lambda w: w.score, reverse=True)


def feedback_links(work, config):
    if not work.doi or not config.feedback_repository:
        return []
    links = []
    for rating, name in RATING_NAMES.items():
        payload = {"doi": normalize_doi(work.doi), "rating": rating, "facets": facet_ids(work, config)}
        body = "<!-- zotwatch-feedback-v1 -->\n```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```\n公开反馈，请勿填写私人笔记。"
        query = urlencode({"title": "[ZotWatch feedback] " + name + " " + work.title[:90], "body": body})
        links.append({"name": name, "url": f"https://github.com/{config.feedback_repository}/issues/new?{query}"})
    return links


class RetrievalWarnings(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        if record.name.startswith(("src.fetch_new", "src.source_paging", "src.ingest_zotero_api", "src.author_watch", "src.citation_watch")):
            # Avoid copying raw URLs, API parameters or credentials into a public report.
            text = record.getMessage().lower()
            if "retrying" in text:
                return  # A transient retry alone is not evidence of incomplete coverage.
            kind = "检索分页/候选数量达到上限" if any(k in text for k in ("cap", "上限", "pagination")) else "检索或同步出现故障/不完整响应"
            message = record.name + ": " + kind
            if message not in self.messages:
                self.messages.append(message)


def coverage_report(config, stages, warnings, state):
    counts = {stage: {f.id: 0 for f in config.facets} for stage in stages}
    for stage, works in stages.items():
        for work in {work_key(w): w for w in works}.values():
            for facet in facet_ids(work, config):
                counts[stage][facet] += 1
    previous = state.get("coverage", {})
    today = datetime.now(timezone.utc).date().isoformat()
    current, rows = {}, []
    for facet in config.facets:
        numbers = {stage: count[facet.id] for stage, count in counts.items()}
        old = previous.get(facet.id, {})
        zero = 0 if numbers.get("delivered") else old.get("zero_runs", 0) + int(old.get("date") != today)
        current[facet.id] = {"date": today, "zero_runs": zero}
        if warnings:
            status = "覆盖不完整，不能据此判断无新文献"
        elif not numbers.get("topic"):
            status = "有抓取命中但未通过主题筛选，建议复核" if numbers.get("raw") else "本轮候选集无命中（非全网无文献）"
        elif not numbers.get("dedup"):
            status = "命中文献已在库中或属于重复项"
        elif not numbers.get("delivered"):
            status = "有候选，受评分/日期/历史/反馈/名额限制"
        else:
            status = "本轮有推送"
        rows.append({"facet": facet.name, **numbers, "zero_runs": zero, "status": status})
    state["coverage"] = current
    return rows


def propose_tracking(works, config, settings, state, feedback):
    """Accumulate distinct papers per stable author ID; no automatic registry mutation."""
    observations = state.setdefault("research_observations", {})
    for work in works:
        if work.label == "ignore" or work.similarity < settings.citation_watch.min_similarity:
            continue
        entry = feedback.entries.get(normalize_doi(work.doi))
        if entry and entry.rating == "irrelevant":
            continue
        rows = work.extra.get("openalex_authorships", [])
        if len(rows) > 30:
            continue
        key = work_key(work)
        prior = observations.get(key, {})
        observations[key] = {"title": work.title, "doi": work.doi, "url": work.url,
                             "openalex_id": work.identifier if re.search(r"(?:^|/)W\d+$", work.identifier) else "",
                             "authors": rows or prior.get("authors", []), "score": work.score,
                             "facets": facet_ids(work, config), "approved": bool(entry and entry.rating in {"direct", "transferable", "mechanism"}),
                             "seen": datetime.now(timezone.utc).isoformat()}
    # Bound state growth; retain the latest 500 distinct papers, not repeated-run counts.
    observations = dict(sorted(observations.items(), key=lambda kv: kv[1].get("seen", ""), reverse=True)[:500])
    state["research_observations"] = observations
    watched = {aid for a in settings.author_watch.authors if a.enabled for aid in a.openalex_ids}
    authors = defaultdict(list)
    for key, row in observations.items():
        entry = feedback.entries.get(normalize_doi(row.get("doi")))
        if entry and entry.rating == "irrelevant":
            continue
        for author in row["authors"]:
            aid = author.get("author_id", "")
            if re.fullmatch(r"A\d+", aid) and aid not in watched:
                authors[aid].append({"key": key, **row, "identity": author})
    proposals = []
    for aid, papers in authors.items():
        papers = list({p["key"]: p for p in papers}.values())
        if len(papers) >= config.proposal_min_papers:
            proposals.append({"kind": "作者候选", "id": aid, "name": papers[0]["identity"].get("name") or aid,
                              "papers": [{"title": p["title"], "url": p["url"], "doi": p["doi"]} for p in papers[:3]],
                              "count": len(papers), "reason": "多篇不同的高相关论文；身份、机构和研究延续性仍待确认"})
    existing = {s.openalex_id for s in settings.citation_watch.seeds}
    for key, row in observations.items():
        entry = feedback.entries.get(normalize_doi(row.get("doi")))
        approved = entry and entry.rating in {"direct", "transferable", "mechanism"}
        ident = row["openalex_id"].rsplit("/", 1)[-1]
        if approved and ident and ident not in existing:
            proposals.append({"kind": "种子候选", "id": ident, "name": row["title"], "count": 1,
                              "papers": [{"title": row["title"], "url": row["url"], "doi": row["doi"]}],
                              "reason": "你已明确标记有用；确认后才加入固定引文追踪"})
    proposals.sort(key=lambda p: (p["kind"] == "种子候选", p["count"]), reverse=True)
    return proposals[:config.proposal_limit]
