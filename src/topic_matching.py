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


def matching_priorities(work: CandidateWork, scoring: ScoringConfig) -> List[dict]:
    """Every priority rule the work matches, in declaration order.

    The resolver below used to return on the first match, so the rule table encoded its
    meaning in file order and nothing could see the alternatives. That is how a demotion
    rule ran for weeks without ever firing: coating and hot-deformation papers were
    claimed at x1.0 by "metal + constitutive" further up, and a corrosion paper by the
    word "fracture energy" in the inverse-identification rule -- where it means the
    numerical regularisation parameter, not a material's toughness. Nobody could tell,
    because the losing matches were never computed.
    """
    text = " ".join(filter(None, [work.title, work.abstract]))
    hits = []
    for index, rule in enumerate(scoring.research_priorities):
        scope = work.title if rule.match_fields == "title" else text
        if rule.required_groups and matches_groups(scope, rule.required_groups):
            hits.append({"index": index, "name": rule.name, "multiplier": rule.multiplier,
                         "match_fields": rule.match_fields,
                         "authoritative": bool(getattr(rule, "authoritative", False))})
    return hits


def research_priority(work: CandidateWork, scoring: ScoringConfig) -> Tuple[str, float]:
    """Resolve the priority multiplier from all matching rules.

    Policy, stated here once instead of being implied by row order:

      1. An `authoritative` rule wins. Those state a positive identification -- this is
         my material and my research chain -- which a peripheral word in the title does
         not undo. "Ductile fracture of coated Ti-6Al-4V under dynamic loading" is core
         work that happens to involve a coating.
      2. Otherwise the LOWEST multiplier wins. A demotion says something about the paper
         makes it less useful, and that stays true however many generic rules also
         recognise it. "Hot Deformation Behavior of AA3102 Aluminum Alloy: Constitutive
         Modeling" matches the cross-metal rule, but hot working is still not this
         project's regime.

    Row order now only picks which NAME is reported among rules of equal standing, so
    the report can still distinguish core TC4 work from cross-metal method work.

    Deliberately not "first match wins", which is what this was: correctness then
    depended on placing every new rule correctly against every existing one, and it
    failed silently -- a demotion rule ran for weeks without once firing, because rules
    above it claimed the same papers at x1.0 first. Also not "highest wins", which would
    let one generous rule cancel every demotion.
    """
    hits = matching_priorities(work, scoring)
    if not hits:
        return "其他相关研究", scoring.default_priority_multiplier
    authoritative = [hit for hit in hits if hit.get("authoritative")]
    if authoritative:
        best = max(hit["multiplier"] for hit in authoritative)
        winner = next(hit for hit in authoritative if hit["multiplier"] == best)
    else:
        lowest = min(hit["multiplier"] for hit in hits)
        winner = next(hit for hit in hits if hit["multiplier"] == lowest)
    return winner["name"], winner["multiplier"]
