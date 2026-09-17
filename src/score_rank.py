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
from .settings import Settings
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
        self.vectorizer = vectorizer or TextVectorizer()
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

        texts = [c.content_for_embedding() for c in candidates]
        vectors = self.vectorizer.encode(texts)
        logger.info("Scoring %d candidate works", len(candidates))

        distances, indices = self.index.search(vectors, top_k=1)
        weights = self.settings.scoring.weights
        thresholds = self.settings.scoring.thresholds

        ranked: List[RankedWork] = []
        for row, (candidate, vector, distance) in enumerate(zip(candidates, vectors, distances)):
            similarity = float(distance[0]) if distance.size else 0.0
            profiles = getattr(self, "profile", {}).get("problem_profiles", {})
            problem_scores = {key: float(vector @ np.asarray(profile["centroid"])) for key, profile in profiles.items()}
            primary_problem = max(problem_scores, key=problem_scores.get) if problem_scores else ""
            affinity = problem_scores.get(primary_problem, similarity)
            semantic_score = 0.4 * similarity + 0.6 * affinity if problem_scores else similarity
            recency_score = _compute_recency(candidate.published, self.settings)
            citation_score, altmetric_score = _compute_metric(candidate)
            journal_quality, journal_sjr = _journal_quality_score(candidate.venue, self.journal_metrics)
            author_bonus = _bonus(candidate.authors, self.settings.scoring.whitelist_authors)
            venue_bonus = _bonus(
                [candidate.venue] if candidate.venue else [],
                self.settings.scoring.whitelist_venues,
            )

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


def _bonus(values: List[str], whitelist: List[str]) -> float:
    whitelist_lower = {normalize_text(v) for v in whitelist}
    for value in values:
        if value and normalize_text(value) in whitelist_lower:
            return 1.0
    return 0.0


def _journal_quality_score(venue: Optional[str], metrics: Dict[str, float]) -> Tuple[float, Optional[float]]:
    if not venue:
        return 1.0, None
    key = normalize_text(venue)
    value = metrics.get(key)
    if value is None:
        return 1.0, None
    score = float(np.log1p(value))
    if score < 1.0:
        score = 1.0
    return score, float(value)


def _compute_recency(published: datetime | None, settings: Settings) -> float:
    if not published:
        return 0.0
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta_days = max((now - published).days, 0)
    decay = settings.scoring.decay_days
    if delta_days <= decay.get("fast", 30):
        return 1.0
    if delta_days <= decay.get("medium", 60):
        return 0.7
    if delta_days <= decay.get("slow", 180):
        return 0.4
    return 0.1


def _compute_metric(candidate: CandidateWork) -> Tuple[float, float]:
    citations = float(candidate.metrics.get("cited_by", candidate.metrics.get("is-referenced-by", 0.0)))
    altmetric = float(candidate.metrics.get("altmetric", 0.0))
    citation_score = float(np.log1p(citations)) if citations else 0.0
    altmetric_score = float(np.log1p(altmetric)) if altmetric else 0.0
    return citation_score, altmetric_score


__all__ = ["WorkRanker"]
