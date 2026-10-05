"""Side-by-side method comparison of this week's top papers, from full text where open.

The cards answer "what is this paper about". This table answers the question a
researcher on a mechanics chain asks next: what material, which tests at which rates
and temperatures, which constitutive and failure model, calibrated how, implemented
in which code, validated against what. Lined up in one table, the week's papers can be
compared in a minute instead of opened one by one.

Full text comes only from open-access PDFs OpenAlex lists for the paper; nothing is
fetched past a paywall. Without one the abstract is used, and every row says which.
The model is told to answer only from the text it is given and to write 未报告 where
the text is silent -- an empty cell is information, an invented one is a liability.

Auxiliary throughout: a failure leaves one row marked as failed, and the whole
section is skipped, never the digest.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import requests

from .http_utils import request_with_retry
from .lineage import norm_doi

logger = logging.getLogger(__name__)

# Version 2 (2026-10-05). Version 1 gave a concrete example inside each field
# description, and on the first real run Qwen3-8B copied two of them verbatim into a
# paper that contains neither: a 34CrNi3MoA SHPB abstract came back with calibration
# "DIC + 有限元反演" and validation "与弹道极限速度对比". The descriptions now say what
# a field means without a copyable answer, and every value must carry a quote that is
# checked against the source text (see grounded()).
PROMPT_VERSION = 2
FIELDS = {
    "material": "研究的材料，以及制备工艺或热处理状态",
    "tests": "做了哪些力学试验（试验类型与试样类型）",
    "strain_rate": "试验覆盖的应变率范围",
    "temperature": "试验覆盖的温度范围",
    "stress_state": "覆盖的应力状态（应力三轴度、Lode 参数或所用试样几何）",
    "model": "采用或提出的本构模型、失效或损伤模型",
    "calibration": "模型参数是怎样标定的",
    "simulation": "数值实现：软件、求解方式、用户子程序、单元或网格",
    "validation": "模型或结论用什么数据、什么工况验证",
    "finding": "最关键的一条定量结论",
}
UNREPORTED = "未报告"

SYSTEM_PROMPT = (
    "你是冲击动力学与材料力学方向的科研助手。从给定论文文本中抽取方法信息，用于把多篇论文"
    "放在一张表里横向比较。\n规则：\n"
    "1. 只依据给定文本，不推测、不用常识或其他论文补全。文本没有明确写到的字段，value 填「"
    + UNREPORTED + "」，quote 填空字符串。\n"
    "2. 每个非「" + UNREPORTED + "」的字段必须给 quote：从给定文本中原样复制的一段连续原文"
    "（不超过 30 个词），它必须能直接支持 value。找不到这样的原文，就填「" + UNREPORTED + "」。\n"
    "3. value 用中文，不超过 40 个汉字；数值、单位、合金牌号、模型名、软件名保留原文写法。\n"
    "字段：\n" + "\n".join(f"- {k}：{v}" for k, v in FIELDS.items()) +
    "\n严格输出一个 JSON 对象：键为上述英文字段名，值为 {\"value\": \"...\", \"quote\": \"...\"}，"
    "不要输出其他文字。"
)

# Variants the model writes instead of the exact marker: "未报告应变率范围",
# "氢脆钢，未报告制备或热处理状态", "未提及", "N/A".
_SILENT = re.compile(r"未报告|未提及|未说明|未给出|未注明|未明确|不详|^(?:无|n/?a|none|unknown|not reported|-|—)$", re.I)
_WORD = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

METHOD_HEAD = re.compile(
    r"(?im)^[ \t]*(?:\d+(?:\.\d+)*\.?[ \t]+)?(?:materials? and methods?|experimental(?: procedures?| details| methods?| setup| work)?"
    r"|methods?|methodology|numerical (?:model(?:l?ing)?|simulations?|methods?)|finite element (?:model(?:l?ing)?|simulations?|analysis)"
    r"|constitutive model(?:l?ing)?)\b[^\n]{0,60}$")
CONCLUSION_HEAD = re.compile(r"(?im)^[ \t]*(?:\d+\.?[ \t]+)?(?:conclusions?|concluding remarks|summary and conclusions?)\b[^\n]{0,40}$")
END_HEAD = re.compile(r"(?im)^[ \t]*(?:references|acknowledg(?:e)?ments?|declaration of competing interest)\b[^\n]{0,40}$")

METHOD_CHARS, CONCLUSION_CHARS = 11000, 2500


def pdf_urls(work, record):
    """Open-access PDF links OpenAlex lists, best first; arXiv as a fallback."""
    urls = []
    record = record or {}
    for location in [record.get("best_oa_location") or {}, *(record.get("locations") or [])]:
        url = (location or {}).get("pdf_url")
        if isinstance(url, str) and url.startswith("http"):
            urls.append(url)
    match = re.search(r"arxiv\.org/abs/([\w.\-/]+)", work.url or "")
    if match:
        urls.append(f"https://arxiv.org/pdf/{match.group(1)}")
    return list(dict.fromkeys(urls))[:3]


def pdf_text(session, url, *, max_bytes=25_000_000, timeout=40, max_pages=40):
    """Text of a PDF, or None for anything that is not a readable PDF."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        with session.get(url, timeout=timeout, stream=True) as response:
            if response.status_code != 200:
                return None
            data = bytearray()
            for chunk in response.iter_content(65536):
                data.extend(chunk)
                if len(data) > max_bytes:
                    return None
        if not bytes(data[:1024]).lstrip().startswith(b"%PDF"):
            return None  # a landing page or a bot challenge, not the paper
        reader = PdfReader(io.BytesIO(bytes(data)))
        pages = [page.extract_text() or "" for page in reader.pages[:max_pages]]
    except Exception as exc:  # network, malformed PDF, pypdf internals
        logger.info("Full text unavailable from %s: %s", re.sub(r"\?.*", "", url), type(exc).__name__)
        return None
    text = "\n".join(pages).strip()
    return text if len(text) > 2000 else None


def method_excerpt(text):
    """The methods part and the conclusions -- what the table is filled from."""
    end = END_HEAD.search(text, len(text) // 2)
    body = text[:end.start()] if end else text
    head = next((m for m in METHOD_HEAD.finditer(body) if m.start() > 800), None)
    start = head.start() if head else 0
    methods = body[start:start + METHOD_CHARS]
    conclusion = ""
    tail = list(CONCLUSION_HEAD.finditer(body))
    if tail and tail[-1].start() > start + METHOD_CHARS:
        conclusion = body[tail[-1].start():tail[-1].start() + CONCLUSION_CHARS]
    return methods + ("\n\n[结论]\n" + conclusion if conclusion else "")


def _cache_key(work, source, model):
    ident = norm_doi(work.doi) or work.title.casefold()
    return hashlib.sha256(f"{PROMPT_VERSION}\x1f{model}\x1f{source}\x1f{ident}".encode()).hexdigest()[:24]


def _value(raw):
    """Collapse a field to its reported part, or the marker if nothing is reported."""
    value = re.sub(r"\s+", " ", str(raw or "")).strip()
    kept = [part.strip() for part in re.split(r"[，,；;]", value) if part.strip() and not _SILENT.search(part.strip())]
    return "，".join(kept)[:80] if kept else UNREPORTED


def grounded(quote, source, *, min_share=0.8):
    """Does the quote actually occur in the source text?

    Token containment rather than exact substring: PDF extraction breaks lines and
    hyphenates, and the model normalises whitespace and dashes. A quote with no
    alphanumeric token cannot be checked and does not count as evidence.
    """
    quote = str(quote or "")
    if re.search(r"[㐀-鿿]", quote):  # a Chinese quote: whitespace-free substring
        squeeze = lambda s: re.sub(r"\s+", "", s)
        return len(squeeze(quote)) >= 4 and squeeze(quote) in squeeze(source)
    words = _WORD.findall(quote.lower())
    if len(words) < 2:
        return False
    present = set(_WORD.findall(source.lower()))
    return sum(word in present for word in words) / len(words) >= min_share


def _clean(fields, source=None):
    """Normalise every field; with a source, drop values whose quote is not in it.

    Returns (fields, dropped). Cached rows were checked when written, so they are
    re-normalised without a source.
    """
    out, dropped = {}, 0
    for key in FIELDS:
        raw = fields.get(key)
        quote = raw.get("quote") if isinstance(raw, dict) else None
        value = _value(raw.get("value") if isinstance(raw, dict) else raw)
        if source is not None and value != UNREPORTED and not grounded(quote, source):
            value, dropped = UNREPORTED, dropped + 1
        out[key] = value
    return out, dropped


class MethodComparer:
    def __init__(self, translation, cache_dir, *, model="", pdf_session=None):
        self.config = translation
        self.model = model or translation.model_name
        self.cache_dir = Path(cache_dir)
        self.pdf_session = pdf_session or requests.Session()
        self.pdf_session.headers.setdefault("User-Agent", "ZotWatch/1.0 (weekly literature digest; open-access text only)")

    @property
    def enabled(self):
        return bool(self.config.enabled and os.getenv(self.config.api_key_env, ""))

    def _ask(self, work, source, text):
        user = f"标题：{work.title}\n来源：{source}\n\n{text}"
        response = request_with_retry(
            requests.Session(), "POST", f"{self.config.base_url.rstrip('/')}/chat/completions",
            logger=logger, context="method comparison", attempts=2,
            headers={"Authorization": f"Bearer {os.getenv(self.config.api_key_env, '')}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "temperature": 0.1, "max_tokens": 2400,
                  "response_format": {"type": "json_object"}, "enable_thinking": False,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": user}]},
            timeout=self.config.timeout_seconds)
        content = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.S)
        return json.loads(match.group(0) if match else content)

    def row(self, rank, work, record):
        text, source = None, "无"
        for url in pdf_urls(work, record):
            full = pdf_text(self.pdf_session, url)
            if full:
                text, source = method_excerpt(full), "全文"
                break
        if text is None and work.abstract and len(work.abstract) > 200:
            text, source = work.abstract, "摘要"
        base = {"rank": rank, "title": work.title, "source": source}
        if text is None:
            return {**base, "fields": None, "note": "无开放全文，摘要过短"}
        path = self.cache_dir / f"{_cache_key(work, source, self.model)}.json"
        try:
            cached = json.loads(path.read_text("utf-8"))
            return {**base, "fields": _clean(cached["fields"])[0], "dropped": cached.get("dropped", 0)}
        except (OSError, ValueError, KeyError, TypeError):
            pass
        # Title + text: the title is evidence too, and the model sees it.
        fields, dropped = _clean(self._ask(work, source, text), f"{work.title}\n{text}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"fields": fields, "dropped": dropped}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return {**base, "fields": fields, "dropped": dropped}

    def compare(self, works, records, *, deadline_seconds=480, workers=4):
        if not self.enabled or not works:
            return None
        started = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=workers)
        futures = {pool.submit(self.row, rank, work, records.get(norm_doi(work.doi)) if work.doi else None): (rank, work)
                   for rank, work in enumerate(works, start=1)}
        done, _ = wait(futures, timeout=deadline_seconds)
        pool.shutdown(wait=False, cancel_futures=True)
        rows = []
        for future, (rank, work) in futures.items():
            if future not in done:
                rows.append({"rank": rank, "title": work.title, "source": "无", "fields": None,
                             "note": "未完成：超过本轮时间上限"})
                continue
            try:
                rows.append(future.result())
            except Exception as exc:
                logger.warning("Method comparison failed for #%d: %s", rank, type(exc).__name__)
                rows.append({"rank": rank, "title": work.title, "source": "无", "fields": None,
                             "note": "抽取失败"})
        rows.sort(key=lambda row: row["rank"])
        return {"rows": rows, "model": self.model, "seconds": round(time.monotonic() - started),
                "dropped": sum(r.get("dropped", 0) for r in rows),
                "full_text": sum(1 for r in rows if r["source"] == "全文" and r["fields"]),
                "abstract_only": sum(1 for r in rows if r["source"] == "摘要" and r["fields"])}
