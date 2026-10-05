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
import ipaddress
import json
import logging
import os
import queue
import re
import threading
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit

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
# Hyphens split words ("finite-element" == "finite element", "Ti-6Al-4V" == ti 6al 4v)
# so PDF hyphenation and the model's re-spacing cannot break a genuine quote.
_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")
_STOP = frozenset("""a an the and or of in on at to for from by with was were is are be been being this that
    these those which as it its we our us using used use than then into under over between via also has have had
    not no can may such both each all any per their they there here while where when after before during""".split())

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
        if isinstance(url, str) and safe_url(url):
            urls.append(url)
    match = re.search(r"arxiv\.org/abs/([\w.\-/]+)", work.url or "")
    if match:
        urls.append(f"https://arxiv.org/pdf/{match.group(1)}")
    return list(dict.fromkeys(urls))[:3]


def safe_url(url):
    """https to a named host only.

    The URLs come from third-party metadata and redirects, and the runner should not
    be steered at a link-local metadata service or anything else by IP literal.
    """
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not host or host == "localhost" or host.endswith(".local"):
        return False
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return True
    return False


def _open(session, url, timeout, max_redirects=4):
    """GET with redirects followed by hand, so every hop passes safe_url."""
    for _ in range(max_redirects + 1):
        if not safe_url(url):
            return None
        response = session.get(url, timeout=timeout, stream=True, allow_redirects=False)
        location = (getattr(response, "headers", None) or {}).get("Location")
        if response.status_code in (301, 302, 303, 307, 308) and location:
            response.close()
            url = urljoin(url, location)
            continue
        return response
    return None


def pdf_text(session, url, *, deadline=None, max_bytes=25_000_000, timeout=(10, 20), max_pages=40):
    """Text of a PDF, or None for anything that is not a readable PDF.

    `deadline` is an absolute time.monotonic() value. The per-read timeout alone does
    not bound a download: a slow server trickling bytes never trips it.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    deadline = deadline or time.monotonic() + 60
    try:
        response = _open(session, url, timeout)
        if response is None:
            return None
        with response:
            if response.status_code != 200:
                return None
            data = bytearray()
            for chunk in response.iter_content(65536):
                data.extend(chunk)
                if len(data) > max_bytes or time.monotonic() > deadline:
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


def _content(text):
    return [w for w in _WORD.findall(str(text or "").lower()) if w not in _STOP]


def grounded(quote, source, *, min_share=0.8, min_pairs=0.7):
    """Does the quote actually occur in the source text?

    Not an exact substring -- PDF extraction breaks lines and hyphenates, and the model
    re-spaces dashes -- but close to one: the quote's content words must be in the
    source AND appear there in the same order, judged by adjacent pairs. A bag of words
    alone was too weak: "calibrated using DIC measurements and the finite element
    analysis" passed against a text that never mentions DIC, because every other word
    occurs somewhere in any FE paper. Stopwords are ignored for the same reason.
    """
    quote = str(quote or "")
    if re.search(r"[㐀-鿿]", quote):  # a Chinese quote: whitespace-free substring
        squeeze = lambda s: re.sub(r"\s+", "", s)
        return len(squeeze(quote)) >= 4 and squeeze(quote) in squeeze(source)
    words = _content(quote)
    if len(words) < 2:
        return False
    tokens = _content(source)
    present = set(tokens)
    if sum(word in present for word in words) / len(words) < min_share:
        return False
    pairs = set(zip(tokens, tokens[1:]))
    wanted = list(zip(words, words[1:]))
    return sum(pair in pairs for pair in wanted) / len(wanted) >= min_pairs


def named_terms_present(value, source, *, min_share=0.75):
    """The Latin-script names in a value (DIC, UMAT, Johnson-Cook) must be in the source.

    A value can outrun its quote: a correct quote about the test, attached to a value
    that adds a calibration method. Numbers are not checked -- the model legitimately
    rewrites 0.001 as 1e-3 -- only tokens containing a letter.
    """
    names = [w for w in _content(value) if len(w) >= 2 and re.search(r"[a-z]", w)]
    if not names:
        return True
    present = set(_content(source))
    return sum(name in present for name in names) / len(names) >= min_share


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
        if (source is not None and value != UNREPORTED
                and not (grounded(quote, source) and named_terms_present(value, source))):
            value, dropped = UNREPORTED, dropped + 1
        out[key] = value
    return out, dropped


class MethodComparer:
    def __init__(self, translation, cache_dir, *, model="", pdf_session=None):
        self.config = translation
        self.model = model or translation.model_name
        self.cache_dir = Path(cache_dir)
        self.pdf_session = pdf_session or requests.Session()
        # Seconds kept back from downloads for the model call, and the least time worth
        # starting a model call with. A call that cannot finish only wastes the budget.
        self.model_reserve, self.min_call_seconds = 30, 10
        self.pdf_session.headers.setdefault("User-Agent", "ZotWatch/1.0 (weekly literature digest; open-access text only)")

    @property
    def enabled(self):
        return bool(self.config.enabled and os.getenv(self.config.api_key_env, ""))

    def _ask(self, work, source, text, timeout=None):
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
            timeout=timeout or self.config.timeout_seconds)
        content = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.S)
        return json.loads(match.group(0) if match else content)

    def row(self, rank, work, record, deadline=None):
        deadline = deadline or time.monotonic() + 600
        text, source = None, "无"
        for url in pdf_urls(work, record):
            if time.monotonic() > deadline - self.model_reserve:
                break  # leave the remaining time for the model call
            full = pdf_text(self.pdf_session, url, deadline=deadline - self.model_reserve)
            if full:
                text, source = method_excerpt(full), "全文"
                break
        if text is None and work.abstract and len(work.abstract) > 200:
            text, source = work.abstract, "摘要"
        base = {"rank": rank, "title": work.title, "source": source}
        if text is None:
            # Elsevier abstracts are often absent from OpenAlex and Crossref alike; on
            # 2026-10-05 the two top-ranked papers had none, so say which case it is.
            note = "无开放全文，摘要过短" if work.abstract else "无开放全文，来源未提供摘要"
            return {**base, "fields": None, "note": note}
        path = self.cache_dir / f"{_cache_key(work, source, self.model)}.json"
        try:
            cached = json.loads(path.read_text("utf-8"))
            return {**base, "fields": _clean(cached["fields"])[0], "dropped": cached.get("dropped", 0)}
        except (OSError, ValueError, KeyError, TypeError):
            pass
        remaining = deadline - time.monotonic()
        if remaining < self.min_call_seconds:
            return {**base, "fields": None, "note": "未完成：超过本轮时间上限"}
        # Title + text: the title is evidence too, and the model sees it.
        reply = self._ask(work, source, text, timeout=min(self.config.timeout_seconds, remaining))
        fields, dropped = _clean(reply, f"{work.title}\n{text}")
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
        deadline = started + deadline_seconds
        jobs = queue.Queue()
        for rank, work in enumerate(works, start=1):
            jobs.put((rank, work))
        results = {}

        def worker():
            while time.monotonic() < deadline:
                try:
                    rank, work = jobs.get_nowait()
                except queue.Empty:
                    return
                record = records.get(norm_doi(work.doi)) if work.doi else None
                try:
                    results[rank] = self.row(rank, work, record, deadline=deadline)
                except Exception as exc:
                    logger.warning("Method comparison failed for #%d: %s", rank, type(exc).__name__)
                    results[rank] = {"rank": rank, "title": work.title, "source": "无",
                                     "fields": None, "note": "抽取失败"}

        # Daemon threads, not a ThreadPoolExecutor. The executor's workers are joined at
        # interpreter exit, so one download stuck past the deadline held the whole watch
        # step open -- measured in review: a 1 s deadline still took 6.4 s to exit, and a
        # slow server could push the job into its 60-minute timeout before the email
        # step. A daemon thread is abandoned at exit instead.
        threads = [threading.Thread(target=worker, daemon=True, name=f"method-compare-{i}")
                   for i in range(max(1, min(workers, len(works))))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        snapshot = dict(results)
        rows = [snapshot.get(rank) or {"rank": rank, "title": work.title, "source": "无", "fields": None,
                                       "note": "未完成：超过本轮时间上限"}
                for rank, work in enumerate(works, start=1)]
        return {"rows": rows, "model": self.model, "seconds": round(time.monotonic() - started),
                "dropped": sum(r.get("dropped", 0) for r in rows),
                "full_text": sum(1 for r in rows if r["source"] == "全文" and r["fields"]),
                "abstract_only": sum(1 for r in rows if r["source"] == "摘要" and r["fields"])}
