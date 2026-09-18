"""The email: a short reminder that points at the web report.

The report is the reading surface. The email exists to say "there are N papers
this week, here are the top few, go look" -- so it carries a preview and a link,
not the whole digest. Rendered with inline styles and table layout because mail
clients strip <style>, <svg> and interactivity.
"""
from __future__ import annotations

import logging
from html import escape
from typing import List, Optional, Sequence

from .models import RankedWork
from .utils import beijing_now

logger = logging.getLogger(__name__)

PREVIEW = 5
FONT = ('-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",'
        '"Helvetica Neue",Arial,sans-serif')
INK, BODY, MUTED, FAINT = "#0d1520", "#3d4650", "#68727e", "#98a1ac"
LINE, PAGE, ACCENT = "#e7eaee", "#f6f7f9", "#1550c8"


def _clip(text: Optional[str], limit: int) -> str:
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def _row(work: RankedWork, rank: int) -> str:
    title_zh = work.extra.get("title_zh") or ""
    tldr = work.extra.get("tldr_zh") or ""
    venue = work.venue or ""
    return f"""
    <tr><td style="padding:0 0 17px;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr>
      <td width="30" valign="top" style="padding:2px 0 0;font:650 12px/1.6 {FONT};color:{FAINT};">{rank:02d}</td>
      <td valign="top" style="padding:0">
        <a href="{escape(work.url or '#')}" style="font:600 15px/1.5 {FONT};color:{INK};
           text-decoration:none;">{escape(work.title or '')}</a>
        {f'<div style="margin-top:5px;font:500 14px/1.55 {FONT};color:{MUTED};">{escape(title_zh)}</div>' if title_zh else ''}
        {f'<div style="margin-top:8px;padding:9px 12px;background:#f4f8ff;border-left:3px solid #4a7fe0;font:400 13px/1.7 {FONT};color:#24344d;">{escape(tldr)}</div>' if tldr else ''}
        {f'<div style="margin-top:7px;font:400 12px/1.6 {FONT};color:{FAINT};">{escape(venue)}</div>' if venue else ''}
      </td></tr></table>
    </td></tr>"""


def render_digest(works: Sequence[RankedWork], *, report_url: str = "", feed_url: str = "") -> str:
    now = beijing_now()
    must = [w for w in works if w.label == "must_read"]
    preview: List[RankedWork] = (must or list(works))[:PREVIEW]
    remaining = len(works) - len(preview)

    button = ""
    if report_url:
        button = f"""
      <tr><td style="padding:6px 0 4px;">
        <table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr>
        <td bgcolor="{INK}" style="border-radius:9px;">
          <a href="{escape(report_url)}" style="display:inline-block;padding:12px 26px;
             font:600 14px/1 {FONT};color:#ffffff;text-decoration:none;">阅读完整周报 →</a>
        </td></tr></table>
      </td></tr>"""

    rows = "".join(_row(w, i) for i, w in enumerate(preview, 1))
    more = ""
    if remaining > 0:
        more = (f'<tr><td style="padding:0 0 18px;font:400 13px/1.7 {FONT};color:{MUTED};">'
                f'另有 {remaining} 篇在完整周报中。</td></tr>')
    if not works:
        rows = (f'<tr><td style="padding:6px 0 18px;font:400 14px/1.7 {FONT};color:{MUTED};">'
                f"本轮无通过筛选且尚未推送的文献。</td></tr>")

    feed_link = (f' · <a href="{escape(feed_url)}" style="color:{ACCENT};text-decoration:none">RSS</a>'
                 if feed_url else "")

    return f"""<!DOCTYPE html>
<html lang="zh"><head>
<meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" />
<meta name="color-scheme" content="light only" />
<title>文献周报 {now:%Y-%m-%d}</title></head>
<body style="margin:0;padding:0;background:{PAGE};">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
  style="background:{PAGE};"><tr><td align="center" style="padding:30px 14px 36px;">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0"
  style="width:100%;max-width:600px;background:#ffffff;border:1px solid {LINE};border-radius:13px;">
<tr><td style="padding:30px 30px 26px;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">

  <tr><td style="padding:0 0 3px;font:700 10.5px/1 {FONT};letter-spacing:.18em;color:{FAINT};">ZOTWATCH</td></tr>
  <tr><td style="padding:0 0 4px;font:700 21px/1.3 {FONT};color:{INK};letter-spacing:-.02em;">文献周报</td></tr>
  <tr><td style="padding:0 0 20px;font:400 13px/1.7 {FONT};color:{MUTED};">
    {now:%Y年%m月%d日} · 本期 {len(works)} 篇{f"，其中必读 {len(must)} 篇" if must else ""}
  </td></tr>
  {button}
  <tr><td style="padding:22px 0 14px;"><div style="height:1px;background:{LINE};font-size:0;line-height:0">&nbsp;</div></td></tr>
  <tr><td style="padding:0 0 14px;font:700 11px/1 {FONT};letter-spacing:.12em;color:{MUTED};">
    {"必读预览" if must else "本期预览"}
  </td></tr>
  {rows}
  {more}

  <tr><td style="padding:16px 0 0;border-top:1px solid {LINE};font:400 11.5px/1.8 {FONT};color:{FAINT};">
    {f'<a href="{escape(report_url)}" style="color:{ACCENT};text-decoration:none">完整周报与运行诊断</a>' if report_url else ''}{feed_link}<br />
    中文标题与摘要由模型生成，仅供快速筛选，以原文为准。
  </td></tr>

</table></td></tr></table>
</td></tr></table>
</body></html>"""


def render_text(works: Sequence[RankedWork], *, report_url: str = "") -> str:
    now = beijing_now()
    must = [w for w in works if w.label == "must_read"]
    lines = [f"文献周报 {now:%Y-%m-%d} · 本期 {len(works)} 篇"
             + (f"，其中必读 {len(must)} 篇" if must else ""), ""]
    if report_url:
        lines += [f"完整周报：{report_url}", ""]
    for i, work in enumerate((must or list(works))[:PREVIEW], 1):
        lines.append(f"{i:02d}. {work.title}")
        if work.extra.get("title_zh"):
            lines.append(f"    {work.extra['title_zh']}")
        if work.extra.get("tldr_zh"):
            lines.append(f"    {work.extra['tldr_zh']}")
        lines.append(f"    {work.url or ''}")
        lines.append("")
    if not works:
        lines += ["本轮无通过筛选且尚未推送的文献。", ""]
    elif len(works) > PREVIEW:
        lines.append(f"另有 {len(works) - PREVIEW} 篇在完整周报中。")
    return "\n".join(lines)


__all__ = ["render_digest", "render_text"]
