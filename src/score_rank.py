from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .faiss_store import FaissIndex
from .citation_watch import citation_strength
from .models import CandidateWork, RankedWork
from .settings import ScoreScales, Settings
from .topic_matching import normalize_text, research_priority
from .vectorizer import TextVectorizer

logger = logging.getLogger(__name__)


@dataclass
class RankerArtifacts:
    index_path: Path
    profile_path: Path


class WorkRanker:
    def __init__(self, base_dir: Path | str, settings: Settings, vectorizer: TextVectorizer | None = None):
        self.base_dir = Path(base_dir)
        self.settings = settings
        self.vectorizer = vectorizer or TextVectorizer.from_settings(settings)
        self.artifacts = RankerArtifacts(
            index_path=self.base_dir / "data" / "faiss.index",
            profile_path=self.base_dir / "data" / "profile.json",
        )
        self.index = FaissIndex.load(self.artifacts.index_path)
        self.profile = self._load_profile()
        self.journal_metrics = self._load_journal_metrics()

    def _load_profile(self) -> dict:
        path = self.artifacts.profile_path
        if not path.exists():
            raise FileNotFoundError("Profile JSON not found; run profile build first.")
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_journal_metrics(self) -> Dict[str, float]:
        path = self.base_dir / "data" / "journal_metrics.csv"
        metrics: Dict[str, float] = {}
        if not path.exists():
            logger.warning("Journal metrics file not found: %s", path)
            return metrics
        try:
            with path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    title = normalize_text(row.get("title") or "")
                    sjr = row.get("sjr")
                    if not title or not sjr:
                        continue
                    try:
                        metrics[title] = float(sjr)
                    except ValueError:
                        continue
        except Exception as exc:
            logger.warning("Failed to load journal metrics: %s", exc)
            return {}
        logger.info("Loaded %d journal SJR entries", len(metrics))
        return metrics

    def rank(self, candidates: List[CandidateWork]) -> List[RankedWork]:
        if not candidates:
            return []

        separator = getattr(self.vectorizer, "text_separator", "[SEP]")
        texts = [c.content_for_embedding(separator) for c in candidates]
        vectors = self.vectorizer.encode(texts)
        logger.info("Scoring %d candidate works", len(candidates))

        # Similarity is the mean over the nearest `neighbors` library items. Using
        # only the single nearest item let one accidental match against one paper
        # in the library spike a candidate's score.
        neighbors = max(1, int(getattr(self.settings.embedding, "neighbors", 5)))
        neighbors = min(neighbors, max(1, int(getattr(self.index, "ntotal", neighbors))))
        distances, indices = self.index.search(vectors, top_k=neighbors)
        weights = self.settings.scoring.weights
        thresholds = self.settings.scoring.thresholds
        scales = self.settings.scoring.scales

        ranked: List[RankedWork] = []
        for row, (candidate, vector, distance) in enumerate(zip(candidates, vectors, distances)):
            similarity = float(np.mean(distance)) if distance.size else 0.0
            nearest_similarity = float(distance[0]) if distance.size else 0.0
            profiles = getattr(self, "profile", {}).get("problem_profiles", {})
            problem_scores = {key: float(vector @ np.asarray(profile["centroid"])) for key, profile in profiles.items()}
            primary_problem = max(problem_scores, key=problem_scores.get) if problem_scores else ""
            affinity = problem_scores.get(primary_problem, similarity)
            semantic_score = _clamp01(
                0.4 * similarity + 0.6 * affinity if problem_scores else similarity
            )
            recency_score = _compute_recency(candidate.published, scales)
            citation_score, altmetric_score = _compute_metric(candidate, scales)
            journal_quality, journal_sjr = _journal_quality_score(
                candidate.venue, self.journal_metrics, scales
            )
            author_bonus = _bonus(candidate.authors, self.settings.scoring.whitelist_authors)
            venue_bonus = _bonus(
                [candidate.venue] if candidate.venue else [],
                self.settings.scoring.whitelist_venues,
            )

            # Every term above is in [0, 1], so with weights summing to 1.0 the
            # score is in [0, 1] and the label thresholds are directly comparable
            # across runs.
            score = (
                semantic_score * weights.similarity
                + recency_score * weights.recency
                + citation_score * weights.citations
                + altmetric_score * weights.altmetric
                + journal_quality * getattr(weights, "journal_quality", 0.0)
                + author_bonus * weights.author_bonus
                + venue_bonus * weights.venue_bonus
            )

            priority, multiplier = research_priority(candidate, self.settings.scoring)
            base_score = score
            score *= multiplier
            watched_bonus = (self.settings.author_watch.score_bonus
                             if self.settings.author_watch.enabled and candidate.extra.get("watched_authors") else 0.0)
            score += watched_bonus
            citation_bonus = (citation_strength(candidate.extra) * self.settings.citation_watch.score_bonus
                              if self.settings.citation_watch.enabled else 0.0)
            score += citation_bonus
            legacy_score = score + (similarity - semantic_score) * weights.similarity * multiplier
            payload = candidate.model_dump()
            payload["extra"] = {
                **candidate.extra,
                "research_priority": priority,
                "priority_multiplier": multiplier,
                "base_score": base_score,
                "watched_author_bonus": watched_bonus,
                "citation_bonus": citation_bonus,
                "legacy_score": legacy_score,
                "semantic_score": semantic_score,
                "problem_scores": problem_scores,
                "primary_problem": primary_problem,
                "nearest_similarity": nearest_similarity,
                "score_components": {
                    "semantic": semantic_score,
                    "recency": recency_score,
                    "citations": citation_score,
                    "altmetric": altmetric_score,
                    "journal_quality": journal_quality,
                    "author_bonus": author_bonus,
                    "venue_bonus": venue_bonus,
                },
            }
            index_items = getattr(self, "profile", {}).get("index_items", [])
            nearest = int(indices[row][0]) if indices is not None else -1
            if 0 <= nearest < len(index_items):
                payload["extra"]["nearest_library_work"] = index_items[nearest]

            label = "ignore"
            if score >= thresholds.must_read:
                label = "must_read"
            elif score >= thresholds.consider:
                label = "consider"
            if candidate.extra.get("semantic_facets") and affinity < self.settings.research.semantic_min_similarity:
                label = "ignore"
                payload["extra"]["semantic_gate_failed"] = True

            ranked.append(
                RankedWork(
                    **payload,
                    score=score,
                    similarity=similarity,
                    recency_score=recency_score,
                    metric_score=citation_score,
                    author_bonus=author_bonus,
                    venue_bonus=venue_bonus,
                    journal_quality=journal_quality,
                    journal_sjr=journal_sjr,
                    label=label,
                )
            )
        ranked.sort(key=lambda w: w.score, reverse=True)
        return ranked


def _clamp01(value: float) -> float:
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _bonus(values: List[str], whitelist: List[str]) -> float:
    whitelist_lower = {normalize_text(v) for v in whitelist}
    for value in values:
        if value and normalize_text(value) in whitelist_lower:
            return 1.0
    return 0.0


def _journal_quality_score(
    venue: Optional[str], metrics: Dict[str, float], scales: ScoreScales
) -> Tuple[float, Optional[float]]:
    """Map SJR onto [0, 1] between a floor and a ceiling.

    The previous version returned max(log1p(sjr), 1.0) and also returned 1.0 for
    unknown venues, so nearly every candidate received the identical value and the
    term discriminated nothing while still consuming its full weight.
    """
    if not venue:
        return scales.journal_unknown, None
    key = normalize_text(venue)
    value = metrics.get(key)
    if value is None:
        return scales.journal_unknown, None
    low = float(np.log1p(scales.sjr_floor))
    high = float(np.log1p(scales.sjr_ceiling))
    score = (float(np.log1p(value)) - low) / (high - low)
    return _clamp01(score), float(value)


def _compute_recency(published: datetime | None, scales: ScoreScales) -> float:
    """Continuous exponential decay.

    The previous step function dropped from 0.4 to 0.1 between day 30 and day 31,
    a 4x penalty for one day of age.
    """
    if not published:
        return 0.0
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta_days = max((now - published).total_seconds() / 86400.0, 0.0)
    half_life = scales.recency_half_life_days
    return _clamp01(float(np.exp(-np.log(2.0) * delta_days / half_life)))


def _compute_metric(candidate: CandidateWork, scales: ScoreScales) -> Tuple[float, float]:
    """Saturating citation and altmetric scores.

    log1p(citations) is unbounded: at the previous weight of 0.08 a paper with 200
    citations gained +0.42 while the entire semantic term could contribute at most
    0.68, so citation count could outrank topical relevance. In a feed whose whole
    purpose is new work -- which has zero citations by construction -- that is
    backwards. Saturating at `citation_saturation` keeps the term a tie-breaker.
    """
    citations = float(candidate.metrics.get("cited_by", candidate.metrics.get("is-referenced-by", 0.0)))
    altmetric = float(candidate.metrics.get("altmetric", 0.0))
    citation_score = _clamp01(np.log1p(max(citations, 0.0)) / np.log1p(scales.citation_saturation))
    altmetric_score = _clamp01(np.log1p(max(altmetric, 0.0)) / np.log1p(scales.altmetric_saturation))
    return citation_score, altmetric_score


__all__ = ["WorkRanker"]
