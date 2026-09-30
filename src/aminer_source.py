"""AMiner candidate discovery; origin and generated prose remain separate from evidence."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from urllib.parse import quote
from datetime import datetime, timezone
from pathlib import Path

from .aminer_client import AMinerClient, AMinerError
from .citation_watch import merge_candidates
from .models import CandidateWork
from .topic_matching import matches_any
from .author_watch import candidate_from_openalex
import requests
from .query_feedback import schedule, query_id, annotate_counts

logger = logging.getLogger(__name__)


def clean_doi(value):
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", str(value or "").strip(), flags=re.I).casefold().rstrip(" .;")
    return value if re.fullmatch(r"10\.\d{4,9}/\S+", value) else None


def text(value):
    return value.strip() if isinstance(value, str) else ""


def align_identities(works, baseline):
    """Conservative no-DOI join: exact title + year + full first author; refuse ambiguity."""
    def signature(w):
        year = w.published.year if w.published else w.extra.get("publication_year")
        norm = lambda s: re.sub(r"[^\w]", "", s.casefold())
        author = norm(w.authors[0]) if w.authors else ""
        return (norm(w.title), year, author) if year and len(author) >= 5 else None
    index = {}
    for w in baseline:
        sig = signature(w)
        if sig and w.doi:
            index.setdefault(sig, set()).add(clean_doi(w.doi))
    result = []
    for w in works:
        matches = index.get(signature(w), set()) - {None}
        if not w.doi and len(matches) == 1:
            w = w.model_copy(deep=True)
            w.doi = next(iter(matches))
            w.extra.setdefault("external_ids", {})["doi"] = w.doi
            w.extra["identity_match"] = "exact_title_year_first_author"
        result.append(w)
    return result


def normalize_paper(raw, *, route, facet="", query=""):
    ident = text(raw.get("paper_id") or raw.get("id"))
    title = text(raw.get("title")) or text(raw.get("title_zh"))
    if not title or not re.fullmatch(r"[\w-]{1,128}", ident, flags=re.ASCII):
        return None
    names = raw.get("authors") or ([raw["first_author"]] if raw.get("first_author") else [])
    if not isinstance(names, list):
        names = [names]
    authors = [text(x.get("name") or x.get("name_zh")) if isinstance(x, dict) else text(x) for x in names]
    venue = raw.get("venue")
    venue = text(venue.get("raw")) if isinstance(venue, dict) else text(venue)
    abstract = text(raw.get("abstract")) or text(raw.get("abstract_slice"))
    published = None
    precision = "unknown"
    try:
        year = int(raw.get("year"))
        if not 1500 <= year <= datetime.now(timezone.utc).year + 1:
            year = None
    except (TypeError, ValueError):
        year = None
    date = text(raw.get("publication_date"))
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        try:
            published = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
            precision = "day"
        except ValueError:
            pass
    if published is None and year:
        # Keep year outside published: no invented January 1 date in weekly selection.
        precision = "year"
    extra = {
        "aminer_id": ident, "external_ids": {"aminer": ident},
        "publication_year": published.year if published else year, "date_precision": precision,
        "abstract_is_partial": bool(abstract and not text(raw.get("abstract"))),
        "abstract_source": "aminer" if abstract else None,
        "generated_summary": text(raw.get("summary")),
        "aminer_citation_bucket": text(raw.get("n_citation_bucket")),
        "aminer_facets": [facet] if facet else [],
        "is_retracted": raw.get("is_retracted") is True,
        "provenance": [{"provider": "aminer", "route": route, "facet": facet,
                        "query": query, "fetched_at": datetime.now(timezone.utc).isoformat()}],
        "field_sources": {"title": "aminer", "abstract": "aminer" if abstract else None,
                          "published": "aminer" if published else None},
    }
    valid_id = lambda value: isinstance(value, str) and bool(re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', value))
    extra['aminer_author_ids'] = sorted({x['id'] for x in names if isinstance(x, dict) and valid_id(x.get('id'))})
    orgs = []
    for author in names:
        if isinstance(author, dict):
            value = author.get('org_id', [])
            orgs.extend(value if isinstance(value, list) else [value])
    extra['aminer_org_ids'] = sorted({value for value in orgs if valid_id(value)})
    venue_record = raw.get('venue') if isinstance(raw.get('venue'), dict) else {}
    venue_id = raw.get('venue_id') or venue_record.get('id')
    if valid_id(venue_id):
        extra['aminer_venue_id'] = venue_id
    doi = clean_doi(raw.get("doi"))
    if doi:
        extra["external_ids"]["doi"] = doi
    return CandidateWork(source="aminer", identifier="aminer:" + ident, title=title,
        abstract=abstract or None, authors=[n for n in authors if n], doi=doi,
        url="https://www.aminer.cn/pub/" + ident, published=published,
        venue=venue or text(raw.get("venue_name")) or None, extra=extra)


def query_plan(settings, entities=()):
    """Round-robin stages prevent the first facet from consuming all search pages."""
    cfg = settings.sources.aminer
    facets = settings.research.facets
    recommendations = [(f.id, "recommend", {"topics": [f.semantic_query or f.name], "size": cfg.recommendation_size}) for f in facets]
    entity_recommendations = [('entity:' + e['key'], 'recommend', {'aminer_author_id': e['aminer_id'],
        'author_name': e['name'], 'topics': [(f.semantic_query or f.name)[:600] for f in facets[:3]],
        'size': cfg.recommendation_size}) for e in entities if e['enabled'] and e['kind'] == 'person']
    plan = []
    for page in range(1, cfg.search_pages + 1):
        for slot in range(cfg.phrases_per_facet):
            for f in facets:
                terms = list(dict.fromkeys(t.strip() for t in f.terms if len(t.strip()) >= 3))
                # Prefer meaningful phrases over abbreviations such as LPBF/SLM alone.
                phrases = list(dict.fromkeys(t.strip() for t in f.aminer_phrases if t.strip())) or (
                    [t for t in terms if " " in t] + [t for t in terms if " " not in t])
                if slot < len(phrases):
                    plan.append((f.id, "search", {"title": phrases[slot], "page": page, "size": cfg.search_size}))
    # One short-phrase pass for every direction before any slow recommendation call.
    first_pass = min(len(facets), len(plan))
    return plan[:first_pass] + entity_recommendations + recommendations + plan[first_pass:]


class AMinerSource:
    def __init__(self, settings, cache_dir, *, client=None, feedback=None, entities=()):
        self.settings = settings
        self.config = settings.sources.aminer
        self.cache_dir = Path(cache_dir)
        self.plan = query_plan(settings, entities)
        self.fingerprint = hashlib.sha256(json.dumps(self.plan, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.client = client if client is not None else AMinerClient(self.config, self.cache_dir, query_version=self.fingerprint)
        self.stats = {}
        self.feedback = feedback

    def fetch(self):
        if not self.config.enabled:
            return []
        try:
            self.client.check_cooldown()
        except AMinerError as e:
            self.client.warn(str(e))
            self.stats = self.client.summary()
            return []
        # Rotate whole query plan between capped runs; cursor is independent of credential.
        cursor_path = self.cache_dir / "rotation.json"
        start = 0
        rec_start = 0
        try:
            saved = json.loads(cursor_path.read_text("utf-8"))
            if saved.get("fingerprint") == self.fingerprint:
                start = int(saved["next"]) % max(1, len(self.plan))
                rec_start = int(saved.get("recommendation_next", 0))
        except (OSError, ValueError, KeyError, TypeError):
            pass
        expected = sum(p.get("size", 0) for _, _, p in self.plan)
        reserve = min(self.config.max_requests // 4, math.ceil(min(expected, self.config.max_enrich_items) / self.config.metadata_batch_size))
        works = []
        rec_positions = [i for i, (_, endpoint, _) in enumerate(self.plan) if endpoint == "recommend"]
        plan = list(self.plan)
        scheduling = {"feedback_active": False}
        if rec_positions:
            rec_start %= len(rec_positions)
            rec_rows = [self.plan[i] for i in rec_positions]
            ordered, scheduling = schedule(rec_rows, rec_start, self.config.max_recommendation_queries,
                                           getattr(self.feedback, "preferences", {}) or {})
            visit_positions = sorted(rec_positions, key=lambda i: (i - start) % len(plan))
            for slot, index in enumerate(visit_positions):
                plan[index] = ordered[slot]
        rec_seen = set()
        rec_attempts = 0
        rec_low_novelty = 0
        rec_observations = []
        rec_skipped = 0
        outcomes = []
        next_index = start
        for offset in range(len(plan)):
            index = (start + offset) % len(plan)
            facet, endpoint, params = plan[index]
            if endpoint == "recommend":
                if (rec_attempts >= self.config.max_recommendation_queries
                        or rec_low_novelty >= self.config.redundant_recommendation_streak):
                    rec_skipped += 1
                    next_index = (index + 1) % len(plan)
                    continue
                rec_attempts += 1
            try:
                rows = self.client.query(endpoint, params, reserve=reserve)
            except AMinerError as e:
                self.client.warn(str(e))
                if endpoint == "recommend" and str(e) in {"request_cap", "time_cap", "discovery_time_cap", "credential_missing", "source_stopped"}:
                    rec_attempts -= 1
                if self.client.stopped or str(e) in {"request_cap", "time_cap", "discovery_time_cap", "credential_missing"}:
                    break
                continue
            next_index = (index + 1) % len(self.plan)
            query = params.get("title") or next(iter(params.get("topics", [])), '') or params.get('author_name', '')
            if len(rows) >= params["size"]:
                self.client.warn("candidate_cap")
            batch = []
            ident = query_id(endpoint, params)
            for row in rows:
                work = normalize_paper(row, route=endpoint, facet='' if facet.startswith('entity:') else facet, query=query)
                if work:
                    work.extra["provenance"][-1]["query_id"] = ident
                    if facet.startswith('entity:'):
                        work.extra['provenance'][-1]['entity_key'] = facet.removeprefix('entity:')
                        work.extra['entity_discovery_note'] = '由已确认学者触发的推荐线索；并不证明该学者署名。'
                    works.append(work)
                    batch.append(work)
                else:
                    self.client.warn("invalid_candidate_record")
            outcomes.append({"query_id": ident, "facet": facet, "endpoint": endpoint, "returned": len(rows), "valid_candidates": len(batch)})
            if endpoint == "recommend":
                ids = {w.extra["aminer_id"] for w in batch}
                novel = ids - rec_seen
                ratio = len(novel) / len(ids) if ids else 0
                rec_observations.append({"facet": facet, "returned_unique": len(ids),
                                         "new_ids": len(novel), "novel_ratio": ratio})
                rec_seen.update(ids)
                rec_low_novelty = rec_low_novelty + 1 if ratio < self.config.min_recommendation_novel_ratio else 0
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cursor_path.write_text(json.dumps({"fingerprint": self.fingerprint, "next": next_index,
                "recommendation_next": (rec_start + (int(rec_attempts > 0) if scheduling["feedback_active"] else rec_attempts)) % max(1, len(rec_positions))}), "utf-8")
        except OSError:
            self.client.warn("rotation_write_failed")
        works = merge_candidates(works)
        # Free partial abstracts BEFORE strict topic filtering, prioritising mechanical titles.
        anchors = self.settings.sources.mechanics_anchor_keywords
        missing = sorted([w for w in works if not w.abstract], key=lambda w: not matches_any(w.title, anchors))
        selected = missing[:self.config.max_enrich_items]
        details = {}
        for offset in range(0, len(selected), self.config.metadata_batch_size):
            ids = [w.extra["aminer_id"] for w in selected[offset:offset + self.config.metadata_batch_size]]
            try:
                rows = self.client.query("info", {"ids": ids})
                for row in rows:
                    ident = text(row.get("id"))
                    if ident in ids:
                        details[ident] = row
            except AMinerError as e:
                self.client.warn(str(e))
                break
        if len(selected) < len(missing):
            self.client.warn("metadata_cap")
        result = []
        for work in works:
            row = details.get(work.extra["aminer_id"])
            if row:
                info = normalize_paper({**row, "title": row.get("title") or work.title}, route="info")
                if info:
                    # Info is joined only by returned AMiner ID, never by array position.
                    info.doi = work.doi
                    work = merge_candidates([work, info])[0]
            result.append(work)
        # Recommendation omits DOI. A small exact-ID title lookup makes useful
        # matches eligible for existing DOI metadata, dedupe and feedback paths.
        identity_count = 0
        for work in sorted(result, key=lambda w: not matches_any(w.title + " " + (w.abstract or ""), anchors)):
            if work.doi or identity_count >= self.config.max_identity_lookups:
                continue
            if not matches_any(work.title + " " + (work.abstract or ""), anchors):
                continue
            identity_count += 1
            try:
                rows = self.client.query("search", {"title": work.title, "page": 1, "size": 5})
                matches = {clean_doi(r.get("doi")) for r in rows if text(r.get("id")) == work.extra["aminer_id"]} - {None}
                if len(matches) == 1:
                    work.doi = next(iter(matches))
                    work.extra["external_ids"]["doi"] = work.doi
                    work.extra["identity_match"] = "aminer_id_title_lookup"
            except AMinerError as e:
                self.client.warn(str(e))
                if self.client.stopped or str(e) in {"request_cap", "time_cap"}:
                    break
        result = merge_candidates(result)
        self.stats = {**self.client.summary(), "query_fingerprint": self.fingerprint,
                      "planned_queries": len(self.plan), "raw_candidates": len(result),
                      "metadata_enriched": len(details), "identity_lookups": identity_count}
        self.stats.update(recommendation_queries=rec_attempts, recommendation_skipped=rec_skipped,
                          query_schedule=scheduling, queries=outcomes,
                          recommendation_observations=rec_observations,
                          recommendation_redundancy_stop=rec_low_novelty >= self.config.redundant_recommendation_streak)
        return result

    def record_topic_results(self, works):
        self.stats = annotate_counts(self.stats, works, "topic_pass")
        try:
            path = self.cache_dir / "query-results.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(self.stats, ensure_ascii=False, indent=2), "utf-8")
            temp.replace(path)
        except OSError:
            self.client.warn("query_observation_write_failed")

    def resolve_dates(self, works, session):
        """Bounded exact-DOI metadata lookup using the existing OpenAlex budget/cache."""
        if not self.settings.sources.openalex.enabled:
            return works
        results = []
        count = 0
        for work in works:
            if (not work.published and work.doi and count < self.config.max_doi_resolutions
                    and matches_any(work.title + " " + (work.abstract or ""), self.settings.sources.mechanics_anchor_keywords)):
                count += 1
                try:
                    response = session.request("GET", "https://api.openalex.org/works/" + quote("https://doi.org/" + work.doi, safe=""),
                        params={"mailto": self.settings.sources.openalex.mailto}, timeout=15, allow_redirects=False)
                    if response.status_code != 200:
                        raise AMinerError("doi_metadata_http")
                    data = response.json()
                    if not isinstance(data, dict):
                        raise AMinerError("doi_metadata_schema")
                    if clean_doi(data.get("doi")) == work.doi:
                        if data.get("is_retracted"):
                            work.extra["is_retracted"] = True
                        resolved = candidate_from_openalex(data)
                        if resolved:
                            work = merge_candidates([work, resolved])[0]
                except (requests.RequestException, ValueError, TypeError):
                    self.client.warn("doi_metadata_incomplete")
            results.append(work)
        self.stats["doi_metadata_requests"] = count
        self.stats["warnings"] = list(self.client.warnings)
        return results
