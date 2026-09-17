"""Bounded one-hop citation discovery with auditable edges, never name inference."""
from __future__ import annotations

import logging
import math
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests

from .author_watch import candidate_from_openalex, work_key
from .http_utils import request_with_retry

logger = logging.getLogger(__name__)


def wid(value):
    value = (value or "").rsplit("/", 1)[-1]
    return value if re.fullmatch(r"W\d+", value) else ""


def reference_ids(item):
    return {wid(x) for x in item.get("referenced_works", []) if wid(x)}


def evidence(item):
    return {"id": wid(item.get("id")), "title": item.get("display_name") or "Untitled",
            "url": item.get("doi") or item.get("id")}


def merge_candidates(works):
    """Merge provenance BEFORE DOI deduplication can discard a discovery route."""
    merged = {}
    for work in works:
        key = work_key(work)
        if key not in merged:
            merged[key] = work.model_copy(deep=True)
            continue
        old = merged[key]
        extra = {**old.extra, **work.extra}
        for field in ("cites_seeds", "referenced_by", "watched_authors"):
            rows = old.extra.get(field, []) + work.extra.get(field, [])
            extra[field] = list({str(sorted(row.items())): row for row in rows}.values())
        extra["referenced_works"] = sorted(set(old.extra.get("referenced_works", [])) |
                                           set(work.extra.get("referenced_works", [])))
        merged[key] = old.model_copy(update={"extra": extra, "abstract": old.abstract or work.abstract})
    return list(merged.values())


class CitationDiscovery:
    def __init__(self, settings, session):
        self.settings, self.session = settings, session
        self.config = settings.citation_watch
        self.requests = 0
        self.warnings = []
        self.seed_records = {}

    def warn(self, message):
        if message not in self.warnings:
            self.warnings.append(message)
            logger.warning(message)

    def query(self, filter_value, *, sort=None, max_pages=1):
        cursor = "*"
        seen = set()
        for page in range(max_pages):
            if self.requests >= self.config.max_requests:
                self.warn("引文扩展已达到本轮请求上限，覆盖不完整。")
                break
            params = {"filter": filter_value, "per-page": 100, "cursor": cursor,
                      "mailto": self.settings.sources.openalex.mailto,
                      "select": "id,doi,display_name,type,is_retracted,publication_date,referenced_works,abstract_inverted_index,authorships,primary_location,cited_by_count"}
            if sort:
                params["sort"] = sort
            self.requests += 1
            time.sleep(self.settings.sources.request_interval_seconds)
            try:
                data = request_with_retry(self.session, "GET", "https://api.openalex.org/works",
                                          params=params, timeout=30, attempts=2,
                                          logger=logger, context="Citation discovery").json()
            except (requests.RequestException, ValueError):
                self.warn("部分引文 API 请求失败，本轮引文覆盖不完整；常规检索仍继续。")
                break
            rows = data.get("results", [])
            yield from rows
            total = data.get("meta", {}).get("count", 0)
            if len(rows) < 100 or (total and (page + 1) * 100 >= total):
                break
            nxt = data.get("meta", {}).get("next_cursor")
            if page + 1 == max_pages or not nxt or nxt in seen:
                self.warn("部分引文查询触及分页上限或游标缺失，覆盖不完整。")
                break
            seen.add(cursor)
            cursor = nxt

    def resolve(self, ids, field="openalex"):
        result = []
        ids = list(dict.fromkeys(ids))
        for offset in range(0, len(ids), 20):
            result.extend(self.query(f"{field}:" + "|".join(ids[offset:offset + 20])))
        return result

    def fetch(self, dynamic_works=()):
        if not self.config.enabled:
            return []
        configured = {s.openalex_id for s in self.config.seeds}
        records = self.resolve(sorted(configured)) if configured else []
        valid = [r for r in records if candidate_from_openalex(r) is not None]
        self.seed_records = {wid(r.get("id")): r for r in valid if wid(r.get("id")) in configured}
        if configured - self.seed_records.keys():
            self.warn("部分种子未解析或已撤稿，已跳过这些种子，未将缺失结果视为零引用。")
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=self.settings.sources.window_days)
        found = []
        seed_ids = sorted(self.seed_records)
        for offset in range(0, len(seed_ids), 6):
            group = seed_ids[offset:offset + 6]
            filt = (f"cites:{'|'.join(group)},from_publication_date:{since.date().isoformat()},"
                    f"to_publication_date:{now.date().isoformat()}")
            for raw in self.query(filt, sort="publication_date:desc", max_pages=self.config.max_pages):
                work = candidate_from_openalex(raw)
                matches = reference_ids(raw) & self.seed_records.keys()
                if work is None or not matches or not work.published or not since <= work.published <= now:
                    continue
                work.extra.update(discovery_route="citation_watch", referenced_works=sorted(reference_ids(raw)),
                                  cites_seeds=[evidence(self.seed_records[k]) for k in sorted(matches)])
                found.append(work)

        # Recent high-ranked candidates supply fresh reference lists, not unrestricted recursion.
        dois = ["https://doi.org/" + work_key(w) for w in dynamic_works if w.doi]
        parents = dict(self.seed_records)
        for raw in self.resolve(dois[:self.config.dynamic_seed_count], "doi") if dois else []:
            if candidate_from_openalex(raw) is not None:
                parents[wid(raw.get("id"))] = raw
        cited_by = defaultdict(list)
        for raw in parents.values():
            for ref in reference_ids(raw):
                if ref not in parents:
                    cited_by[ref].append(evidence(raw))
        # Shared references first; weekly rotation avoids permanently starving the long tail.
        ordered = sorted(cited_by, key=lambda key: (-len(cited_by[key]), key))
        cap = self.config.max_reference_candidates
        if len(ordered) > cap:
            self.warn("经典补漏候选超过上限：优先共享参考文献，其余按周轮换，非穷尽回溯。")
            priority = ordered[:cap // 2]
            tail = ordered[cap // 2:]
            start = (now.date().toordinal() // 7 * (cap - len(priority))) % len(tail)
            ordered = priority + (tail[start:] + tail[:start])[:cap - len(priority)]
        for raw in self.resolve(ordered):
            work = candidate_from_openalex(raw)
            ident = wid(raw.get("id"))
            if work is None or ident not in cited_by or not work.published or work.published > now:
                continue
            work.extra.update(discovery_route="citation_watch", referenced_by=cited_by[ident],
                              referenced_works=sorted(reference_ids(raw)))
            found.append(work)
        logger.info("Citation discovery: %d seeds, %d parents, %d candidates, %d queries",
                    len(self.seed_records), len(parents), len(found), self.requests)
        return self.annotate(merge_candidates(found))

    def annotate(self, works):
        result = []
        for work in works:
            extra = dict(work.extra)
            refs = {wid(x) for x in extra.get("referenced_works", []) if wid(x)}
            direct = refs & self.seed_records.keys()
            if direct:
                extra["cites_seeds"] = [evidence(self.seed_records[k]) for k in sorted(direct)]
            best = None
            for seed in self.seed_records.values():
                seed_refs = reference_ids(seed)
                shared = refs & seed_refs
                if len(shared) < 2:
                    continue
                score = len(shared) / math.sqrt(len(refs) * len(seed_refs))
                if best is None or score > best["score"]:
                    best = {"seed": evidence(seed), "count": len(shared), "score": score,
                            "reference_ids": sorted(shared)[:5]}
            if best:
                extra["bibliographic_coupling"] = best
            result.append(work.model_copy(update={"extra": extra}))
        return result


def citation_strength(extra):
    direct = bool(extra.get("cites_seeds") or extra.get("referenced_by"))
    coupling = min(1.0, max(0.0, extra.get("bibliographic_coupling", {}).get("score", 0)))
    return min(1.0, (2 / 3 if direct else 0) + coupling / 3)


def recommendation_reasons(work):
    """Deterministic evidence, not an LLM-invented explanation or quality guarantee."""
    rows = []
    nearest = work.extra.get("nearest_library_work")
    if nearest:
        rows.append(f"库内相似论文：{nearest['title']}（相似度 {work.similarity:.3f}）")
    for source in work.extra.get("cites_seeds", [])[:3]:
        rows.append(f"引用重点论文：{source['title']} [{source['url']}]")
    for source in work.extra.get("referenced_by", [])[:3]:
        rows.append(f"被相关论文引用：{source['title']} [{source['url']}]")
    coupling = work.extra.get("bibliographic_coupling")
    if coupling:
        rows.append(f"与《{coupling['seed']['title']}》共享 {coupling['count']} 篇参考文献；归一化重合度 {coupling['score']:.3f}")
        rows.append("共享参考文献示例：" + ", ".join("https://openalex.org/" + k for k in coupling["reference_ids"]))
    return rows
