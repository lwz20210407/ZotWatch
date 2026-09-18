"""Discover research directions by clustering the Zotero library itself.

config/research.yaml currently lists seven hand-written facets. They are a lossy
summary: 329 of the 406 keywords in sources.yaml fall outside them, and one facet
("温度—应变率耦合") asserts a coupling the owner does not always study.

Clustering the library is better evidence than clustering the keyword list. The
keywords are what someone once intended to track; the library is what they
actually collected, weighted by how much of it there is. It also lives in exactly
the vector space the recommender scores against, so the facets that come out of it
line up with the ranking rather than sitting beside it.

    python tools/discover_facets.py                 # 12 clusters, named by the model
    python tools/discover_facets.py -k 16           # more granular
    python tools/discover_facets.py --no-name       # cluster only, skip the API

Prints a proposed facets block for config/research.yaml. Nothing is written: the
research taxonomy is the owner's to approve.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

BASE = Path(__file__).resolve().parent.parent

STOP = set("""a an and are as at be by for from has have in into is it its of on or that the to
with we our this these those which was were been using used use study studies paper based
new novel effect effects analysis investigation research results result approach method
methods model models modelling modeling behaviour behavior properties property under during
via towards toward their they can could also more most high low different various two three
first second between during after before within without over under about such than then when
where while both each other another same different""".split())


def load_vectors(db: Path) -> tuple:
    """Titles plus their cached embeddings, for items that have one."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT title, abstract, embedding, embedding_signature FROM items "
        "WHERE embedding IS NOT NULL AND title != ''").fetchall()
    conn.close()
    if not rows:
        raise SystemExit("profile.sqlite holds no cached embeddings; run "
                         "`python -m src.cli profile --local` first.")
    signature = Counter(r["embedding_signature"] for r in rows).most_common(1)[0][0]
    rows = [r for r in rows if r["embedding_signature"] == signature]
    vectors = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    titles = [r["title"] for r in rows]
    abstracts = [r["abstract"] or "" for r in rows]
    return titles, abstracts, vectors, signature


def kmeans(vectors: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    import faiss
    km = faiss.Kmeans(vectors.shape[1], k, niter=40, nredo=3, seed=seed, verbose=False,
                      spherical=True)
    km.train(vectors.astype(np.float32))
    _, assign = km.index.search(vectors.astype(np.float32), 1)
    return assign.ravel(), km.centroids


def top_terms(titles: list, members: list, all_counts: Counter, limit: int = 12) -> list:
    """Terms unusually frequent in this cluster relative to the whole library."""
    local = Counter()
    for i in members:
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", titles[i].lower()):
            if token not in STOP:
                local[token] += 1
    total_local = sum(local.values()) or 1
    total_all = sum(all_counts.values()) or 1
    scored = []
    for term, n in local.items():
        if n < 3:
            continue
        lift = (n / total_local) / ((all_counts[term] + 1) / total_all)
        scored.append((lift * math.log1p(n), term, n))
    scored.sort(reverse=True)
    return [(t, n) for _, t, n in scored[:limit]]


def representatives(vectors: np.ndarray, centroid: np.ndarray, members: list,
                    titles: list, limit: int = 6) -> list:
    sims = vectors[members] @ centroid
    order = np.argsort(-sims)[:limit]
    return [titles[members[i]] for i in order]


def name_clusters(clusters: list, config) -> dict:
    """Ask the model to name each cluster, in the owner's own terminology."""
    import requests
    from src.http_utils import request_with_retry
    import logging

    key = os.getenv(config.api_key_env, "")
    if not key:
        return {}
    payload = [{"i": c["id"], "terms": [t for t, _ in c["terms"]][:10],
                "titles": c["papers"][:5]} for c in clusters]
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    prompt = (
        "你在帮一位研究金属本构与延性断裂的博士生整理文献库的研究方向。"
        "下面每组是一个聚类，给出该类高区分度的英文词和最靠近中心的论文标题。\n\n"
        "对每一组输出：\n"
        "1. name：4-12 字的中文方向名。必须贴合该组的实际内容，不要编造组里没有的限定词"
        "（例如组里既有单独的温度效应也有单独的应变率效应时，不能命名为「温度—应变率耦合」）。\n"
        "2. id：英文小写下划线标识符。\n"
        "3. terms：8-14 个检索词，中英混合，取自该组实际出现的术语。\n"
        "4. summary：一句话说明这组在研究什么。\n\n"
        '严格输出 JSON：{"items":[{"i":序号,"id":"...","name":"...","terms":["..."],"summary":"..."}]}'
    )
    logger = logging.getLogger("discover")
    response = request_with_retry(
        session, "POST", f"{config.base_url.rstrip('/')}/chat/completions",
        logger=logger, context="name clusters",
        json={"model": config.model_name,
              "messages": [{"role": "system", "content": prompt},
                           {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
              "temperature": 0.3, "max_tokens": 400 * len(clusters) + 512,
              "response_format": {"type": "json_object"}, "enable_thinking": False},
        timeout=300)
    content = response.json()["choices"][0]["message"]["content"]
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    rows = json.loads(text[start:end + 1]).get("items", [])
    return {int(r["i"]): r for r in rows if "i" in r}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", type=int, default=12, help="number of clusters (default 12)")
    parser.add_argument("--no-name", action="store_true", help="skip the model naming step")
    parser.add_argument("--min-size", type=int, default=25,
                        help="report clusters smaller than this as fragments")
    args = parser.parse_args()

    load_dotenv(BASE / ".env")
    titles, abstracts, vectors, signature = load_vectors(BASE / "data" / "profile.sqlite")
    print(f"文库 {len(titles)} 篇，向量 {vectors.shape[1]} 维（{signature}）")
    print(f"聚成 {args.k} 类…\n")

    assign, centroids = kmeans(vectors, args.k)
    all_counts = Counter()
    for title in titles:
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", title.lower()):
            if token not in STOP:
                all_counts[token] += 1

    clusters = []
    for cid in range(args.k):
        members = [i for i, a in enumerate(assign) if a == cid]
        if not members:
            continue
        clusters.append({
            "id": cid,
            "size": len(members),
            "terms": top_terms(titles, members, all_counts),
            "papers": representatives(vectors, centroids[cid], members, titles),
        })
    clusters.sort(key=lambda c: -c["size"])

    named = {} if args.no_name else name_clusters(clusters, __import__(
        "src.settings", fromlist=["load_settings"]).load_settings(BASE).translation)

    for rank, c in enumerate(clusters, 1):
        meta = named.get(c["id"], {})
        flag = "  ← 偏小，可能是碎片" if c["size"] < args.min_size else ""
        print(f"─── {rank:2d}. {meta.get('name', '(未命名)')}   {c['size']} 篇{flag}")
        if meta.get("summary"):
            print(f"      {meta['summary']}")
        print(f"      高区分度词: {', '.join(f'{t}({n})' for t, n in c['terms'][:9])}")
        for title in c["papers"][:3]:
            print(f"      · {title[:92]}")
        print()

    if named:
        print("=" * 78)
        print("# 建议写入 config/research.yaml 的 facets（请你过目后再定稿）")
        print("facets:")
        for c in clusters:
            meta = named.get(c["id"])
            if not meta or c["size"] < args.min_size:
                continue
            terms = ", ".join(json.dumps(t, ensure_ascii=False) for t in meta.get("terms", []))
            print(f"  - id: {meta.get('id', 'cluster_%d' % c['id'])}")
            print(f"    name: \"{meta.get('name', '')}\"    # 库内 {c['size']} 篇")
            print(f"    terms: [{terms}]")
            print(f"    semantic_query: \"{meta.get('summary', '')}\"")
            print("    use: \"\"")
            print("    verify: \"\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
