"""Shared, boundary-aware matching for literature topics and ranking rules."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import List, Tuple

from .models import CandidateWork
from .settings import ScoringConfig


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"(?<!\w)ti[\s-]*6[\s-]*al[\s-]*4[\s-]*v(?!\w)", "ti6al4v", text)
    text = re.sub(r"(?<!\w)ti[\s-]*64(?!\w)", "ti6al4v", text)
    text = re.sub(
        r"(?<!\w)(?:l[\s-]*pbf|lpbf|slm|pbf[\s-]*lb\s*/\s*m|"
        r"selective laser melting|laser beam powder bed fusion|laser powder bed fusion)(?!\w)",
        "lpbf", text,
    )
    return " ".join(re.sub(r"[^\w\s]", " ", text).replace("_", " ").split())


@lru_cache(maxsize=2048)
def _term_pattern(term: str) -> re.Pattern:
    normalized = normalize_text(term)
    if re.search(r"[\u3400-\u9fff]", normalized):
        # Chinese scientific phrases are not separated by spaces in running text.
        return re.compile(re.escape(normalized))
    words = normalized.split()
    if not words:
        return re.compile(r"(?!)")
    parts = [re.escape(word) for word in words]
    # Allow ordinary trailing plurals, but never expand short model acronyms.
    last = term.split()[-1]
    if len(words[-1]) >= 4 and not words[-1].endswith("s") and not last.isupper():
        parts[-1] += "s?"
    return re.compile(r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)")


def matches_term(text: str, term: str) -> bool:
    return bool(_term_pattern(term).search(normalize_text(text)))


def matches_any(text: str, terms: List[str]) -> bool:
    normalized = normalize_text(text)
    return any(_term_pattern(term).search(normalized) for term in terms if term.strip())


def matches_groups(text: str, groups: List[List[str]]) -> bool:
    return all(matches_any(text, group) for group in groups if group)


def research_priority(work: CandidateWork, scoring: ScoringConfig) -> Tuple[str, float]:
    text = " ".join(filter(None, [work.title, work.abstract]))
    for rule in scoring.research_priorities:
        scope = work.title if rule.match_fields == "title" else text
        if rule.required_groups and matches_groups(scope, rule.required_groups):
            return rule.name, rule.multiplier
    return "其他相关研究", scoring.default_priority_multiplier
