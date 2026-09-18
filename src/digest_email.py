"""The email: a briefing that carries the whole issue, not just a link.

The report is still the reading surface -- the email leads with a link to it. But
the email is what actually arrives, so it covers every paper rather than a
five-item teaser: must-reads in full (both titles, the one-line TLDR, byline,
venue, direction and relevance), the rest as a compact list. You can triage the
week without leaving the mailbox, and open the report when you want abstracts,
filters and the archive.

Rendered with inline styles and table layout because mail clients strip <style>,
drop <svg> and have no interactivity.
"""
from __future__ import annotations

import logging
from html import escape
from typing import List, Optional, Sequence

from .models import RankedWork
from .utils import beijing_now

logger = logging.getLogger(__name__)

FONT = ('-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",'
        '"Helvetica Neue",Arial,sans-serif')
INK, BODY, MUTED, FAINT = "#0d1520", "#3d4650", "#68727e", "#98a1ac"
LINE, PAGE, ACCENT, ACCENT_D = "#e7eaee", "#f6f7f9", "#ea7317", "#c85c08"
MUST, MUST_BG, SOFT = "#b3341f", "#fceee9", "#f3f1ed"


def _clip(text: Optional[str], limit: int) -> str:
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def _byline(work: RankedWork) -> str:
    authors = list(work.authors or [])
    who = f"{authors[0]} 等 {len(authors)} 人" if len(authors) > 2 else "、".join(authors)
    when = ""
    if work.published:
        precision = (work.extra or {}).get("date_precision") or "day"
        fmt = {"year": "%Y", "month": "%Y-%m"}.get(precision, "%Y-%m-%d")
        when = work.published.strftime(fmt)
    return " · ".join(b for b in (who, work.venue or "", when) if b)


def _direction(work: RankedWork, problem_names: dict) -> str:
    key = (work.extra or {}).get("primary_problem")
    if key:
        return problem_names.get(key, key)
    return (work.extra or {}).get("research_priority") or ""


def _tag(text: str, fg: str, bg: str) -> str:
    return (f'<span style="display:inline-block;padding:2px 8px;border-radius:5px;'
            f'font:600 11px/1.6 {FONT};color:{fg};background:{bg};'
            f'white-space:nowrap">{escape(text)}</span>')


def _full(work: RankedWork, rank: int, problem_names: dict) -> str:
    """A must-read: everything short of the abstract."""
    title_zh = work.extra.get("title_zh") or ""
    tldr = work.extra.get("tldr_zh") or ""
    direction = _direction(work, problem_names)
    tags = _tag("必读", MUST, MUST_BG)
    if direction:
        tags += "&nbsp;" + _tag(direction, ACCENT_D, "#fdf1e3")
    return f"""
    <tr><td style="padding:0 0 20px;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr>
      <td width="30" valign="top" style="padding:3px 0 0;font:700 12px/1.6 {FONT};color:{FAINT};">{rank:02d}</td>
      <td valign="top" style="padding:0">
        <div style="margin-bottom:7px">{tags}</div>
        <a href="{escape(work.url or '#')}" style="font:600 15.5px/1.5 {FONT};color:{INK};
           text-decoration:none;">{escape(work.title or '')}</a>
        {f'<div style="margin-top:5px;font:500 14px/1.55 {FONT};color:{MUTED};">{escape(title_zh)}</div>' if title_zh else ''}
        {f'<div style="margin-top:9px;padding:9px 12px;background:#fdf1e3;border-left:3px solid {ACCENT};font:400 13.5px/1.75 {FONT};color:#59341a;">{escape(tldr)}</div>' if tldr else ''}
        <div style="margin-top:8px;font:400 12px/1.6 {FONT};color:{FAINT};">
          {escape(_byline(work))}&nbsp;·&nbsp;相关度 {work.score:.2f}
        </div>
      </td></tr></table>
    </td></tr>"""


def _compact(work: RankedWork, rank: int, problem_names: dict) -> str:
    """Everything else: one entry, two or three lines."""
    title_zh = work.extra.get("title_zh") or ""
    direction = _direction(work, problem_names)
    return f"""
    <tr><td style="padding:0 0 13px;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr>
      <td width="30" valign="top" style="padding:2px 0 0;font:600 11.5px/1.6 {FONT};color:{FAINT};">{rank:02d}</td>
      <td valign="top" style="padding:0">
        <a href="{escape(work.url or '#')}" style="font:600 14px/1.5 {FONT};color:{INK};
           text-decoration:none;">{escape(title_zh or work.title or '')}</a>
        {f'<div style="margin-top:3px;font:400 12.5px/1.5 {FONT};color:{MUTED};">{escape(_clip(work.title, 110))}</div>' if title_zh else ''}
        <div style="margin-top:5px;font:400 11.5px/1.6 {FONT};color:{FAINT};">
          {f'{_tag(direction, MUTED, SOFT)}&nbsp;' if direction else ''}{escape(_byline(work))}
        </div>
      </td></tr></table>
    </td></tr>"""


def _rule(label: str, count: int, note: str = "") -> str:
    return f"""
    <tr><td style="padding:24px 0 14px;">
      <div style="font:700 11.5px/1.5 {FONT};letter-spacing:.12em;color:{MUTED};">
        {escape(label)}&nbsp;&nbsp;{count}</div>
      {f'<div style="margin-top:4px;font:400 11.5px/1.6 {FONT};color:{FAINT};">{escape(note)}</div>' if note else ''}
      <div style="margin-top:9px;height:1px;background:{LINE};font-size:0;line-height:0">&nbsp;</div>
    </td></tr>"""


def render_digest(works: Sequence[RankedWork], *, report_url: str = "", feed_url: str = "",
                  problem_names: Optional[dict] = None, issue_no: int = 0,
                  extras: Optional[dict] = None) -> str:
    problem_names = problem_names or {}
    now = beijing_now()
    must = [w for w in works if w.label == "must_read"]
    rest = [w for w in works if w.label != "must_read"]

    rows: List[str] = []
    if must:
        rows.append(_rule("必读", len(must)))
        rows += [_full(w, i, problem_names) for i, w in enumerate(must, 1)]
    if rest:
        rows.append(_rule("其余推荐", len(rest), "按相关度排序，完整摘要见网页版。"))
        rows += [_compact(w, i, problem_names) for i, w in enumerate(rest, len(must) + 1)]
    if not works:
        rows.append(f'<tr><td style="padding:8px 0 18px;font:400 14px/1.7 {FONT};color:{MUTED};">'
                    f"本轮无通过筛选且尚未推送的文献。</td></tr>")

    button = ""
    if report_url:
        button = f"""
      <tr><td style="padding:4px 0 2px;">
        <table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr>
        <td bgcolor="{INK}" style="border-radius:9px;">
          <a href="{escape(report_url)}" style="display:inline-block;padding:12px 26px;
             font:600 14px/1 {FONT};color:#ffffff;text-decoration:none;">阅读完整周报 →</a>
        </td></tr></table>
      </td></tr>"""

    # Other channels are named and counted; they live in full on the web page.
    channel_line = " · ".join(f"{name} {len(items)} 篇"
                              for name, items in (extras or {}).items() if items)
    channels = ""
    if channel_line:
        more = (f' — <a href="{escape(report_url)}" style="color:{ACCENT_D};'
                f'text-decoration:none">在完整周报中查看</a>') if report_url else ""
        channels = (f'<tr><td style="padding:20px 0 0;border-top:1px solid {LINE};'
                    f'font:400 12.5px/1.7 {FONT};color:{MUTED};">'
                    f"其他频道：{escape(channel_line)}{more}</td></tr>")

    feed_link = (f' · <a href="{escape(feed_url)}" style="color:{ACCENT_D};text-decoration:none">RSS</a>'
                 if feed_url else "")
    issue = f"第 {issue_no} 期 · " if issue_no else ""

    return f"""<!DOCTYPE html>
<html lang="zh"><head>
<meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" />
<meta name="color-scheme" content="light only" />
<title>文献周刊 {now:%Y-%m-%d}</title></head>
<body style="margin:0;padding:0;background:{PAGE};">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
  style="background:{PAGE};"><tr><td align="center" style="padding:30px 14px 36px;">
<table role="presentation" width="640" cellpadding="0" cellspacing="0" border="0"
  style="width:100%;max-width:640px;background:#ffffff;border:1px solid {LINE};border-radius:13px;">
<tr><td style="padding:30px 30px 28px;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">

  <tr><td align="center" style="padding:0 0 4px;font:700 10.5px/1 {FONT};
      letter-spacing:.2em;color:{ACCENT};">ZOTWATCH</td></tr>
  <tr><td align="center" style="padding:0 0 6px;font:700 22px/1.3 {FONT};color:{INK};
      letter-spacing:.02em;">每周最新文献订阅与追踪</td></tr>
  <tr><td align="center" style="padding:0 0 18px;font:400 12.5px/1.7 {FONT};color:{MUTED};">
    {issue}{now:%Y年%m月%d日} · 本期 {len(works)} 篇{f"，其中必读 {len(must)} 篇" if must else ""}
  </td></tr>
  <tr><td align="center">{button and '<table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr><td>' or ''}</td></tr>
  {button}

  {"".join(rows)}
  {channels}

  <tr><td style="padding:18px 0 0;border-top:1px solid {LINE};font:400 11.5px/1.8 {FONT};color:{FAINT};">
    {f'<a href="{escape(report_url)}" style="color:{ACCENT_D};text-decoration:none">完整周报与运行诊断</a>' if report_url else ''}{feed_link}<br />
    中文标题与摘要由模型生成，仅供快速筛选，以原文为准。
  </td></tr>

</table></td></tr></table>
</td></tr></table>
</body></html>"""


def render_text(works: Sequence[RankedWork], *, report_url: str = "",
                problem_names: Optional[dict] = None, issue_no: int = 0) -> str:
    problem_names = problem_names or {}
    now = beijing_now()
    must = [w for w in works if w.label == "must_read"]
    rest = [w for w in works if w.label != "must_read"]
    issue = f"第 {issue_no} 期 · " if issue_no else ""
    lines = [f"每周最新文献订阅与追踪 · {issue}{now:%Y-%m-%d}",
             f"本期 {len(works)} 篇" + (f"，其中必读 {len(must)} 篇" if must else ""), ""]
    if report_url:
        lines += [f"完整周报：{report_url}", ""]

    def block(work, i, detailed):
        out = [f"{'★' if detailed else ' '}{i:02d}. {work.title}"]
        if work.extra.get("title_zh"):
            out.append(f"     {work.extra['title_zh']}")
        if detailed and work.extra.get("tldr_zh"):
            out.append(f"     {work.extra['tldr_zh']}")
        meta = _byline(work)
        direction = _direction(work, problem_names)
        if direction:
            meta = f"[{direction}] {meta}"
        out.append(f"     {meta}")
        out.append(f"     {work.url or ''}")
        out.append("")
        return out

    if must:
        lines.append(f"── 必读 {len(must)} 篇 " + "─" * 30)
        lines.append("")
        for i, work in enumerate(must, 1):
            lines += block(work, i, True)
    if rest:
        lines.append(f"── 其余推荐 {len(rest)} 篇 " + "─" * 26)
        lines.append("")
        for i, work in enumerate(rest, len(must) + 1):
            lines += block(work, i, False)
    if not works:
        lines += ["本轮无通过筛选且尚未推送的文献。", ""]
    return "\n".join(lines)


__all__ = ["render_digest", "render_text"]
