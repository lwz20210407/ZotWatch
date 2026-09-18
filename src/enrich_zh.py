"""Chinese title and a one-sentence TLDR for each paper.

Borrowed from how commercial literature feeds present results: Semantic Scholar's
TLDR is what makes a list scannable, because a truncated abstract still forces you
to read prose to find out what the paper did. One sentence stating the object,
the method and the finding lets you triage twenty papers in a minute.

Both fields come from a single chat call per paper batch and are cached on disk by
a hash of title+abstract, so a paper is only ever paid for once. Everything is
best-effort: a failure leaves the English title and no TLDR rather than breaking
the digest.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests

from .http_utils import request_with_retry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是材料与冲击动力学方向的科研助手。对每篇论文输出三项：

1. title_zh：标题的简体中文翻译。
2. tldr：一句话中文概括，30-50 字，必须说清「对什么材料/对象、用什么方法、得到什么
   结果」。禁止「本文研究了…」「具有重要意义」这类空话。没有摘要时根据标题克制概括。
3. abstract_zh：摘要的完整简体中文翻译。逐句译全，不许概括、不许删句、不许添话；
   原文没有的结论不能写。没有摘要时输出空字符串。

三项共同的术语规则：专业术语用标准中文译法（应力三轴度、Lode 参数、延性断裂、
绝热剪切带、本构模型、颈缩后硬化、网格客观性、弹道极限）；合金牌号、化学式、模型名、
缩写和数值单位保留原文（Ti-6Al-4V、TC4、DP800、LPBF、SHPB、DIC、GTN、Johnson-Cook、
200-600 m/s）。

严格输出 JSON 对象：
{"items": [{"i": 序号, "title_zh": "...", "tldr": "...", "abstract_zh": "..."}, ...]}
items 顺序与输入一致、长度一致，不要输出任何其他文字。"""

CJK = re.compile(r"[㐀-鿿]")


def _key(title: str, abstract: Optional[str]) -> str:
    raw = f"{(title or '').strip()}\x00{(abstract or '').strip()[:1200]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _parse(content: str, expected: int) -> List[dict]:
    """Pull the rows out of a chat reply that may be fenced, chatty or wrapped."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.S)
    # Some models prepend reasoning; take the outermost JSON value.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start == -1 or end == -1:
            continue
        try:
            parsed = json.loads(text[start : end + 1])
        except ValueError:
            continue
        rows = parsed.get("items") if isinstance(parsed, dict) else parsed
        if isinstance(rows, list) and len(rows) == expected:
            return rows
        raise ValueError(
            f"expected {expected} rows, got {len(rows) if isinstance(rows, list) else '?'}")
    raise ValueError("no JSON payload in reply")


class ChineseEnricher:
    def __init__(self, config, cache_path: Path | str):
        self.config = config
        self.cache_path = Path(cache_path)
        self.cache: Dict[str, dict] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("Ignoring unreadable enrichment cache: %s", exc)
        self._session: Optional[requests.Session] = None
        self._dirty = False

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and os.getenv(self.config.api_key_env, ""))

    def _post(self, batch: List[dict]) -> List[dict]:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {os.getenv(self.config.api_key_env, '')}",
                "Content-Type": "application/json",
            })
        payload = [
            {"i": i, "title": row["title"], "abstract": (row.get("abstract") or "")[:2000]}
            for i, row in enumerate(batch)
        ]
        response = request_with_retry(
            self._session, "POST",
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            logger=logger, context=f"enrich({len(batch)} papers)",
            json={
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "temperature": 0.2,
                # A whole translated abstract per item, not just a title and a line.
                "max_tokens": 900 * len(batch) + 512,
                "response_format": {"type": "json_object"},
                # Qwen3 emits a long reasoning trace by default, which blows the
                # timeout and buries the JSON. This digest needs the answer only.
                "enable_thinking": False,
            },
            timeout=self.config.timeout_seconds,
        )
        content = response.json()["choices"][0]["message"]["content"]
        return _parse(content, len(batch))

    def enrich(self, works: Sequence) -> None:
        """Fill work.extra['title_zh'] and work.extra['tldr_zh'] for every work."""
        pending: List[dict] = []
        seen: set = set()
        for work in works:
            key = _key(work.title, work.abstract)
            if key in self.cache or key in seen:
                continue
            seen.add(key)
            pending.append({"key": key, "title": work.title, "abstract": work.abstract})

        if pending and not self.enabled:
            logger.info("Chinese enrichment disabled or %s unset; English only",
                        self.config.api_key_env)
        elif pending:
            logger.info("Generating Chinese title + TLDR for %d papers (%d cached)",
                        len(pending), len(self.cache))
            size = self.config.batch_size
            for start in range(0, len(pending), size):
                chunk = pending[start : start + size]
                try:
                    for row, out in zip(chunk, self._post(chunk)):
                        self.cache[row["key"]] = {
                            "title_zh": str(out.get("title_zh") or "").strip(),
                            "tldr": str(out.get("tldr") or "").strip(),
                            "abstract_zh": str(out.get("abstract_zh") or "").strip(),
                        }
                        self._dirty = True
                except Exception as exc:  # best effort; never break the digest
                    logger.warning("Chinese enrichment batch failed (%d papers): %s",
                                   len(chunk), exc)
                else:
                    logger.info("  enriched %d/%d", min(start + size, len(pending)), len(pending))
            self.save()

        for work in works:
            row = self.cache.get(_key(work.title, work.abstract)) or {}
            title_zh = row.get("title_zh") or ""
            # Guard against the model echoing the English title back.
            if title_zh and CJK.search(title_zh):
                work.extra["title_zh"] = title_zh
            if row.get("tldr"):
                work.extra["tldr_zh"] = row["tldr"]
            abstract_zh = row.get("abstract_zh") or ""
            if abstract_zh and CJK.search(abstract_zh):
                work.extra["abstract_zh"] = abstract_zh

    def save(self) -> None:
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=0, sort_keys=True),
            encoding="utf-8")
        logger.info("Chinese enrichment cache: %d papers at %s", len(self.cache), self.cache_path)


def enrich_chinese(works: Sequence, config, cache_path: Path | str) -> None:
    ChineseEnricher(config, cache_path).enrich(works)


__all__ = ["ChineseEnricher", "enrich_chinese"]
