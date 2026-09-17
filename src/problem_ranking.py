"""Independent research profiles and bounded diversity-aware selection."""
from collections import Counter
import math

import numpy as np

from .author_watch import work_key
from .research_features import facet_ids, normalize_doi


def build_problem_profiles(items, vectors, config, feedback=()):
    profiles = {}
    for facet in config.facets:
        selected, weights = [], []
        for i, item in enumerate(items):
            explicit = bool(facet.collection_keys or facet.seed_dois)
            matched = (bool(set(item.collections) & set(facet.collection_keys)) or
                       normalize_doi(item.doi) in {normalize_doi(d) for d in facet.seed_dois}) if explicit else facet.id in facet_ids(item, config)
            weight = 1.0
            for entry in feedback:
                if normalize_doi(item.doi) != entry.doi or entry.rating in {"reset", "read", "later", "reading"}:
                    continue
                if getattr(entry, "scope", "") not in {"", facet.id}:
                    continue
                if getattr(entry, "scope", "") == facet.id or facet.id in entry.facets:
                    if entry.rating == "irrelevant":
                        matched = False
                    else:
                        matched, weight = True, 2.0
            if matched:
                selected.append(i); weights.append(weight)
        if selected:
            centroid = np.average(vectors[selected], axis=0, weights=weights)
            centroid /= np.linalg.norm(centroid) + 1e-12
            profiles[facet.id] = {"name": facet.name, "count": len(selected), "centroid": centroid.tolist(),
                "examples": [{"title": items[i].title, "doi": items[i].doi} for i in selected[:3]]}
    return profiles


def diverse_select(works, top, config, vectorizer):
    if not top or len(works) <= top:
        return works
    pool = works[:max(top, config.diversity_pool)]
    vectors = vectorizer.encode([w.content_for_embedding() for w in pool])
    scores = np.array([w.score for w in pool], dtype=float)
    scores = (scores - scores.min()) / max(float(np.ptp(scores)), 1e-9)
    selected, remaining, counts = [], set(range(len(pool))), Counter()
    cap = max(1, math.ceil(top * 0.4))
    def group(i):
        return pool[i].extra.get("primary_problem") or next(iter(pool[i].extra.get("research_facets", [])), "other")
    def value(i):
        redundant = max((float(vectors[i] @ vectors[j]) for j in selected), default=0)
        return float(scores[i]) - config.diversity_penalty * redundant
    # Reserve a small, relevance-qualified lane for semantic discoveries without many citations.
    explorers = [i for i in remaining if pool[i].extra.get("semantic_facets") and pool[i].metrics.get("cited_by", 0) <= 5]
    for i in sorted(explorers, key=lambda i: (-pool[i].score, work_key(pool[i])))[:min(top, config.exploration_slots)]:
        selected.append(i); remaining.remove(i); counts[group(i)] += 1
    while remaining and len(selected) < top:
        eligible = [i for i in remaining if counts[group(i)] < cap] or list(remaining)
        best = max(eligible, key=lambda i: (value(i), pool[i].score, -i))
        selected.append(best); remaining.remove(best); counts[group(best)] += 1
    return [pool[i].model_copy(update={"extra": {**pool[i].extra, "selection": "分问题名额＋语义去冗余；探索位仍需通过相关性阈值"}}) for i in selected]


def local_evidence_graph(work):
    edges = []
    for row in work.extra.get("cites_seeds", [])[:2]:
        edges.append({"label": row["title"], "relation": "引用", "direction": "out", "url": row["url"]})
    for row in work.extra.get("referenced_by", [])[:2]:
        edges.append({"label": row["title"], "relation": "被引用", "direction": "in", "url": row["url"]})
    nearest = work.extra.get("nearest_library_work")
    if nearest:
        edges.append({"label": nearest["title"], "relation": "标题摘要相似（非引用）", "direction": "similar", "url": ""})
    return edges[:5]
