"""Research lineage for this week's papers, from their OpenAlex reference lists.

A reader meeting a new paper asks two things the ranking cannot answer:

1. Which papers already in my library does it build on?  That places it on the map
   the reader already has, and is the fastest way to judge whether it matters.
2. What do this week's papers have in common upstream?  References cited by several
   of them, laid out oldest first, are the lineage of the week's topics -- and the
   ones missing from the library are the classics worth reading next.

OpenAlex only: no key, priced per list request at 0.0001 USD in network.yaml, and
reachable from the GitHub runner, which AMiner has not reliably been. A reference
list is whatever the publisher deposited, so "no references" means unknown, never
"cites nothing", and the page says how many papers had a list at all.
"""
from __future__ import annotations

import logging
import re
import time
from collections import defaultdict

from .http_utils import request_with_retry

logger = logging.getLogger(__name__)

OPENALEX = "https://api.openalex.org/works"
BATCH = 50
WORK_ID = re.compile(r"W\d+")


def norm_doi(value) -> str:
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", str(value or "").strip(), flags=re.I)
    return doi.casefold().rstrip(" .;")


def wid(value) -> str:
    tail = str(value or "").rsplit("/", 1)[-1]
    return tail if WORK_ID.fullmatch(tail) else ""


class Lookup:
    """Batched OpenAlex list queries under a hard request cap.

    Uses the run's BudgetSession, so these calls share the per-run cap, the daily
    cost ledger and the 48 h GET cache with the main fetch -- a re-run in the same
    week costs nothing. When the budget is gone the remaining lookups are skipped
    and recorded, never retried in a loop.
    """

    def __init__(self, session, *, mailto="", max_requests=30, interval=0.2, max_seconds=120):
        self.session, self.mailto = session, mailto
        self.max_requests, self.interval = max_requests, interval
        self.deadline = time.monotonic() + max_seconds
        self.requests = 0
        self.incomplete = []
        self.stopped = False

    def query(self, filter_value, select):
        # Three stops, because this runs on the main thread before the page is
        # rendered and the email sent: a request cap, a wall-clock cap, and the first
        # failure. Without the last two a slow OpenAlex cost ~62 s per request (two 30 s
        # attempts plus backoff) across ~31 requests -- half an hour on top of a 30-35
        # minute run, enough to hit the job timeout before the email step.
        if self.stopped:
            return []
        if self.requests >= self.max_requests:
            self.incomplete.append("request_cap")
            return []
        if time.monotonic() >= self.deadline:
            self.incomplete.append("time_cap")
            self.stopped = True
            return []
        self.requests += 1
        params = {"filter": filter_value, "per-page": BATCH, "select": select}
        if self.mailto:
            params["mailto"] = self.mailto
        try:
            response = request_with_retry(self.session, "GET", OPENALEX, params=params, timeout=20,
                                          attempts=2, logger=logger, context="Lineage lookup")
            rows = response.json().get("results", [])
        except Exception as exc:  # budget exhaustion, transport, bad JSON: all non-fatal
            self.incomplete.append(type(exc).__name__)
            self.stopped = True
            return []
        time.sleep(self.interval)
        return rows

    def by_doi(self, dois, select):
        # '|' separates OR values and ',' separates filters, so a DOI containing either
        # cannot be expressed in the filter and is skipped rather than mangled.
        usable = [d for d in dict.fromkeys(dois) if d and "|" not in d and "," not in d]
        found = {}
        for start in range(0, len(usable), BATCH):
            chunk = usable[start:start + BATCH]
            for row in self.query("doi:" + "|".join("https://doi.org/" + d for d in chunk), select):
                if norm_doi(row.get("doi")):
                    found[norm_doi(row["doi"])] = row
        return found

    def by_id(self, ids, select):
        found = {}
        ids = [i for i in dict.fromkeys(ids) if WORK_ID.fullmatch(i)]
        for start in range(0, len(ids), BATCH):
            for row in self.query("openalex:" + "|".join(ids[start:start + BATCH]), select):
                if wid(row.get("id")):
                    found[wid(row["id"])] = row
        return found


WEEK_FIELDS = "id,doi,referenced_works,best_oa_location,locations"
REF_FIELDS = "id,doi,display_name,publication_year,cited_by_count"


def resolve_week(works, lookup):
    """One OpenAlex record per paper of the week, keyed by DOI.

    Supplies the reference lists Crossref and AMiner candidates lack, and the open
    access locations the method comparison reads full text from.
    """
    return lookup.by_doi([norm_doi(w.doi) for w in works if w.doi], WEEK_FIELDS)


def _short(title, limit=90):
    title = re.sub(r"\s+", " ", title or "").strip()
    return title if len(title) <= limit else title[:limit - 1].rstrip() + "…"


def build_lineage(works, library_items, records, lookup, *, max_references=1500,
                  library_limit=4, ancestor_limit=15, min_shared=2):
    """Per-paper links into the library, and references shared across the week.

    `works` are in report order, so position i is card #p(i+1) on the page.
    """
    library = {}
    for item in library_items:
        doi = norm_doi(getattr(item, "doi", None))
        if doi:
            library[doi] = {"title": item.title, "year": getattr(item, "year", None)}

    refs_of, cited_by = [], defaultdict(list)
    for rank, work in enumerate(works, start=1):
        refs = {wid(r) for r in work.extra.get("referenced_works") or [] if wid(r)}
        record = records.get(norm_doi(work.doi)) if work.doi else None
        if record:
            refs |= {wid(r) for r in record.get("referenced_works") or [] if wid(r)}
        refs_of.append(refs)
        for ref in refs:
            cited_by[ref].append(rank)

    # Shared references first: they feed both outputs. The long tail only feeds the
    # library links, so it is what gets cut when the cap bites.
    ordered = sorted(cited_by, key=lambda ref: (-len(cited_by[ref]), ref))
    resolved = lookup.by_id(ordered[:max_references], REF_FIELDS)
    truncated = len(ordered) > max_references

    papers = []
    for rank, (work, refs) in enumerate(zip(works, refs_of), start=1):
        links = []
        for ref in refs:
            doi = norm_doi((resolved.get(ref) or {}).get("doi"))
            if doi in library:
                links.append({"title": library[doi]["title"], "year": library[doi]["year"], "doi": doi})
        links.sort(key=lambda row: -(row["year"] or 0))
        papers.append({"rank": rank, "title": _short(work.title), "references": len(refs),
                       "library": links[:library_limit], "library_total": len(links)})

    ancestors = []
    for ref in ordered:
        if len(cited_by[ref]) < min_shared or len(ancestors) >= ancestor_limit:
            break
        row = resolved.get(ref)
        if not row:
            continue
        doi = norm_doi(row.get("doi"))
        ancestors.append({"title": _short(row.get("display_name")), "year": row.get("publication_year"),
                          "doi": doi, "url": f"https://doi.org/{doi}" if doi else row.get("id"),
                          "cited_by": cited_by[ref], "in_library": doi in library,
                          "citations": row.get("cited_by_count")})
    ancestors.sort(key=lambda row: (row["year"] or 9999, -len(row["cited_by"])))

    with_refs = sum(1 for refs in refs_of if refs)
    return {"papers": [p for p in papers if p["library"]], "ancestors": ancestors,
            "coverage": {"papers": len(works), "with_references": with_refs,
                         "linked_to_library": sum(1 for p in papers if p["library"]),
                         "library_dois": len(library), "references": len(ordered),
                         "resolved": len(resolved), "truncated": truncated,
                         "requests": lookup.requests,
                         "incomplete": sorted(set(lookup.incomplete))}}
