from __future__ import annotations

import logging
import re
from typing import Iterable, List, Set

from rapidfuzz import fuzz, process

from .models import CandidateWork
from .storage import ProfileStorage

logger = logging.getLogger(__name__)

# A shorter title that is a strict subset of a longer one is a different paper far
# more often than it is a duplicate: conference abstract vs. journal article,
# Part I vs. Part II, original vs. extended version. `token_set_ratio` scores every
# such pair at 100, so it used to discard the full paper whenever a stub was
# already in the library. `token_sort_ratio` is order-insensitive but not
# subset-blind, and the length guard below refuses to compare titles whose lengths
# are too far apart to plausibly be the same work.
MIN_LENGTH_RATIO = 0.75


class DedupeEngine:
    def __init__(self, storage: ProfileStorage, title_threshold: float = 0.9):
        self.storage = storage
        self.title_threshold = title_threshold
        self.existing_doi: Set[str] = set()
        self.existing_ids: Set[str] = set()
        self.existing_titles: List[str] = []
        self._load_existing()

    def _load_existing(self) -> None:
        for item in self.storage.iter_items():
            if item.doi:
                self.existing_doi.add(_normalize_identifier(item.doi))
            if item.url:
                self.existing_ids.add(_normalize_identifier(item.url))
            title = _normalize_title(item.title)
            if title:
                self.existing_titles.append(title)

    def filter(self, candidates: Iterable[CandidateWork]) -> List[CandidateWork]:
        source = list(candidates)
        deduped: List[CandidateWork] = []
        candidate_titles: List[str] = []
        seen_keys: Set[str] = set()
        suppressed: List[tuple[str, str]] = []

        for work in source:
            key = _normalize_identifier(work.identifier)
            doi = _normalize_identifier(work.doi) if work.doi else None
            title = _normalize_title(work.title)

            if doi and doi in self.existing_doi:
                logger.debug("Skipping %s due to DOI duplication", work.identifier)
                continue
            if doi and doi in seen_keys:
                continue
            if key in self.existing_ids or key in seen_keys:
                logger.debug("Skipping %s due to identifier duplication", work.identifier)
                continue
            match = _closest_title(title, self.existing_titles, self.title_threshold) or _closest_title(
                title, candidate_titles, self.title_threshold
            )
            if match:
                suppressed.append((work.title, match))
                continue

            deduped.append(work)
            if title:
                candidate_titles.append(title)
            seen_keys.add(key)
            if doi:
                seen_keys.add(doi)
        logger.info("Deduped candidates from %d to %d", len(source), len(deduped))
        # Title-similarity suppression is the only rule here that can discard a
        # genuinely new paper, so report it at INFO instead of hiding it in DEBUG.
        for title, match in suppressed:
            logger.info("Title-similarity suppressed: %r matched existing %r", title, match)
        return deduped

    def _is_title_duplicate(self, title: str) -> bool:
        return _closest_title(title, self.existing_titles, self.title_threshold) is not None


def _normalize_identifier(value: str) -> str:
    normalized = (value or "").lower().strip()
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", normalized)


def _normalize_title(title: str) -> str:
    normalized = re.sub(r"\s+", " ", title or "").strip().lower()
    return normalized


def _length_compatible(a: str, b: str) -> bool:
    if not a or not b:
        return False
    shorter, longer = sorted((len(a), len(b)))
    return shorter / longer >= MIN_LENGTH_RATIO


def _closest_title(title: str, title_list: Iterable[str], threshold: float) -> str | None:
    """Return the first existing title that is a plausible duplicate, else None.

    Uses rapidfuzz's vectorised extraction with a score cutoff instead of a Python
    loop over every stored title. The previous implementation compared every
    candidate against every library title one pair at a time, which grows as
    O(candidates x library) and became the dominant cost of a run.
    """
    if not title:
        return None
    pool = [existing for existing in title_list if existing and _length_compatible(title, existing)]
    if not pool:
        return None
    match = process.extractOne(
        title,
        pool,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold * 100.0,
    )
    return match[0] if match else None


__all__ = ["DedupeEngine"]
