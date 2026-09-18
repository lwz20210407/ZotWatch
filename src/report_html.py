"""The web report: the primary reading surface.

Not the email body -- that is a short reminder rendered by digest_email.py.

The visual language is taken from OpenScholar's results page, measured from the
live site rather than guessed:

  page background  #faf9f7 (warm off-white, not a cool grey)
  ink              #241f1b (warm near-black)
  accent           #ea7317 (orange, used for actions and the "show more" control)
  card             white, 16px radius, 1px hairline border,
                   shadow 0 10px 15px -3px rgba(0,0,0,.08), 0 4px 6px -4px rgba(0,0,0,.05)
  title            18px / 600 / line-height 1.375, with a grey "(year)" suffix
  abstract         16px / line-height 1.5, shown inline and expanded on demand
  layout           sticky filter sidebar on the left, result column on the right

Structural borrowings from the same page: an icon-led metadata line for authors
and for the journal/volume/pages, the abstract printed in full rather than hidden
behind a disclosure, a "显示更多" expander, and a metric row at the foot of each
card. What this report adds on top is a Chinese title and a one-sentence Chinese
TLDR, because the reader here is triaging rather than searching.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from jinja2 import Environment, Template
from markupsafe import Markup

from .models import RankedWork
from .citation_watch import recommendation_reasons
from .utils import beijing_now

logger = logging.getLogger(__name__)

ABSTRACT_CLAMP = 420

ICONS = {
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "book": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>'
            '<path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    "link": '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>'
            '<path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
}


def icon(name: str, size: int = 15) -> Markup:
    body = ICONS.get(name, "")
    return Markup(
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true" class="ico">{body}</svg>'
    )


def clamp(text: str | None, limit: int = ABSTRACT_CLAMP) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


def split_abstract(text: str | None, limit: int = ABSTRACT_CLAMP) -> tuple:
    """Split into (visible, hidden) at a word boundary.

    A hard character cut lands mid-word on English abstracts ("petalling at lo").
    CJK has no spaces, so fall back to the hard cut when no boundary is near.
    """
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned, ""
    cut = cleaned.rfind(" ", int(limit * 0.8), limit)
    if cut == -1:
        cut = limit
    return cleaned[:cut].rstrip(), cleaned[cut:].lstrip()


def authors_line(work: RankedWork) -> str:
    authors = list(work.authors or [])
    if len(authors) > 8:
        return ", ".join(authors[:8]) + f" 等 {len(authors)} 人"
    return ", ".join(authors)


def source_line(work: RankedWork) -> str:
    """Journal, volume(issue), pages -- the citation line."""
    bits = [work.venue or ""]
    extra = work.extra or {}
    volume, issue, pages = extra.get("volume"), extra.get("issue"), extra.get("pages")
    if volume:
        bits.append(f"卷 {volume}" + (f" ({issue})" if issue else ""))
    if pages:
        bits.append(f"页 {pages}")
    return ", ".join(b for b in bits if b)


def direction(work: RankedWork, problem_names: dict) -> str:
    key = work.extra.get("primary_problem")
    if key:
        return problem_names.get(key, key)
    return work.extra.get("research_priority") or ""


# feedback_links() emits fourteen options (eight global ratings plus scoped ones).
# A card shows the three that a reader actually uses while triaging; the rest stay
# reachable from the repository's issue templates.
RATING_BUTTONS = (("直接有用", "有用"), ("不相关", "无用"), ("稍后看", "稍后读"))


def rating_buttons(work: RankedWork) -> List[dict]:
    by_name = {link["name"]: link["url"] for link in work.extra.get("feedback_links") or []}
    return [{"label": label, "url": by_name[source]}
            for source, label in RATING_BUTTONS if source in by_name]


def citations(work: RankedWork) -> int:
    metrics = work.metrics or {}
    return int(metrics.get("cited_by", metrics.get("is-referenced-by", 0)) or 0)


def anchor(work: RankedWork) -> str:
    nearest = (work.extra.get("nearest_library_work") or {}).get("title")
    if nearest:
        short = " ".join(str(nearest).split())
        if len(short) > 44:
            short = short[:43] + "…"
        return f"最接近库内《{short}》"
    if work.extra.get("cites_seeds"):
        return f"引用了 {len(work.extra['cites_seeds'])} 篇你关注的论文"
    if work.extra.get("referenced_by"):
        return f"被 {len(work.extra['referenced_by'])} 篇相关论文引用"
    return ""


_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>文献周报 {{ date_label }}</title>
<style>
  :root{
    --page:#faf9f7; --card:#fff; --ink:#241f1b; --body:#4a423c; --muted:#7c736c;
    --faint:#a49b93; --line:#e8e5e1; --soft:#f3f1ed;
    --accent:#ea7317; --accent-d:#c85c08; --accent-soft:#fdf1e3;
    --must:#b3341f; --must-bg:#fceee9;
    --radius:16px;
    --shadow:0 10px 15px -3px rgba(0,0,0,.08),0 4px 6px -4px rgba(0,0,0,.05);
    --font:"Outfit",system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",
           "Helvetica Neue",Arial,sans-serif;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--page);color:var(--ink);font:400 16px/1.6 var(--font);
    -webkit-font-smoothing:antialiased}
  a{color:inherit}
  .ico{flex:none;vertical-align:-2px;color:var(--faint)}

  .shell{max-width:1180px;margin:0 auto;padding:30px 22px 80px;
    display:grid;grid-template-columns:270px minmax(0,1fr);gap:26px;align-items:start}

  /* ---------------- sidebar ---------------- */
  aside{position:sticky;top:26px;display:flex;flex-direction:column;gap:14px}
  .panel{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
    box-shadow:var(--shadow);padding:17px 18px}
  .panel h3{margin:0 0 12px;font-size:13.5px;font-weight:600;color:var(--ink);
    display:flex;align-items:center;gap:7px}
  .kv{display:flex;justify-content:space-between;align-items:baseline;
    padding:5px 0;font-size:13.5px;color:var(--muted)}
  .kv b{font-size:17px;font-weight:600;color:var(--ink)}
  .fitem{display:flex;justify-content:space-between;align-items:center;gap:8px;
    width:100%;padding:7px 10px;margin:2px 0;border:0;border-radius:9px;background:transparent;
    font:inherit;font-size:13.5px;color:var(--body);text-align:left;cursor:pointer;
    transition:background .12s,color .12s}
  .fitem:hover{background:var(--soft)}
  .fitem[aria-pressed="true"]{background:var(--accent-soft);color:var(--accent-d);font-weight:600}
  .fitem span:last-child{font-size:12px;color:var(--faint);font-variant-numeric:tabular-nums}
  .fitem[aria-pressed="true"] span:last-child{color:var(--accent-d)}
  .sw{display:flex;align-items:center;justify-content:space-between;gap:10px;
    font-size:13.5px;color:var(--body);cursor:pointer}
  .sw input{position:absolute;opacity:0;pointer-events:none}
  .sw i{flex:none;width:38px;height:22px;border-radius:999px;background:var(--line);
    position:relative;transition:background .15s}
  .sw i::after{content:"";position:absolute;top:3px;left:3px;width:16px;height:16px;
    border-radius:50%;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.2);transition:transform .15s}
  .sw input:checked+i{background:var(--accent)}
  .sw input:checked+i::after{transform:translateX(16px)}

  /* ---------------- main ---------------- */
  .mast{margin-bottom:18px}
  .kicker{font-size:11.5px;font-weight:700;letter-spacing:.18em;color:var(--accent);
    text-transform:uppercase}
  h1{margin:9px 0 0;font-size:29px;line-height:1.2;font-weight:700;letter-spacing:-.02em}
  .sub{margin-top:7px;font-size:14px;color:var(--muted)}
  .count{margin:18px 0 14px;font-size:14px;color:var(--muted)}
  .count b{font-size:26px;font-weight:700;color:var(--ink);margin-right:5px}

  h2{margin:34px 0 12px;font-size:15px;font-weight:600;color:var(--ink);
    display:flex;align-items:center;gap:9px}
  h2 em{font-style:normal;font-size:12.5px;font-weight:500;color:var(--faint)}
  .note{margin:-6px 0 14px;font-size:13px;line-height:1.7;color:var(--faint)}

  /* ---------------- card ---------------- */
  article{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
    box-shadow:var(--shadow);padding:24px 26px;margin-bottom:18px}
  article[hidden]{display:none}
  .badges{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin-bottom:11px}
  .rank{font-size:12px;font-weight:700;color:var(--faint);font-variant-numeric:tabular-nums}
  .pill{display:inline-block;padding:3px 10px;border-radius:7px;font-size:11.5px;
    font-weight:600;line-height:1.55;white-space:nowrap}
  .p-must{color:var(--must);background:var(--must-bg)}
  .p-dir{color:var(--accent-d);background:var(--accent-soft)}

  .title{display:block;font-size:18px;font-weight:600;line-height:1.375;color:var(--ink);
    text-decoration:none}
  .title:hover{color:var(--accent-d);text-decoration:underline;text-underline-offset:3px}
  .title .yr{color:var(--faint);font-weight:400}
  .title-zh{margin-top:7px;font-size:16px;line-height:1.5;color:var(--body);font-weight:500}

  .line{display:flex;align-items:flex-start;gap:8px;margin-top:9px;
    font-size:14px;line-height:1.55;color:var(--muted)}
  .line em{font-style:italic}

  .tldr{margin:14px 0 0;padding:12px 15px;background:var(--accent-soft);
    border-left:3px solid var(--accent);border-radius:0 10px 10px 0;
    font-size:15px;line-height:1.7;color:#59341a}

  .abs{margin-top:14px;font-size:16px;line-height:1.5;color:var(--body)}
  .abs .lead::before{content:"摘要：";color:var(--muted)}
  .abs .rest{display:none}
  .abs.open .rest{display:inline}
  .more{margin-top:9px;background:none;border:0;padding:0;cursor:pointer;
    font:inherit;font-size:14px;color:var(--accent);display:inline-flex;align-items:center;gap:5px}
  .more::before{content:"⌄";font-size:15px;line-height:1}
  .abs.open+.more::before{content:"⌃"}
  .more:hover{color:var(--accent-d)}

  /* method-transfer note: project-specific, no public search engine has it */
  .transfer{margin-top:12px;padding:10px 13px;background:var(--soft);border-radius:10px;
    font-size:13px;line-height:1.7;color:var(--muted)}
  .transfer b{color:var(--ink);font-weight:600;margin-right:6px}

  /* why this paper is here -- the thing a public search engine cannot show */
  .why{display:flex;align-items:center;gap:7px;margin-top:13px;font-size:13.5px;
    color:var(--accent-d);background:var(--accent-soft);border-radius:9px;padding:8px 12px}
  .why .ico{color:var(--accent)}

  .foot{display:flex;flex-wrap:wrap;align-items:center;gap:14px;margin-top:16px;
    padding-top:14px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}
  .foot a{color:var(--muted);text-decoration:none}
  .foot a:hover{color:var(--accent-d)}
  .foot .m{display:inline-flex;align-items:center;gap:6px}
  .foot .m b{font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums}
  .bar{display:inline-block;width:52px;height:5px;border-radius:3px;background:var(--line);
    overflow:hidden;margin-left:2px}
  .bar i{display:block;height:100%;background:var(--accent);border-radius:3px}

  .rate{margin-left:auto;display:flex;gap:7px}
  .rb{padding:5px 12px;border:1px solid var(--line);border-radius:8px;
    font-size:12.5px;color:var(--muted);text-decoration:none;transition:.12s;background:var(--card)}
  .rb:hover{border-color:var(--accent);color:var(--accent-d);background:var(--accent-soft)}
  .rb-1:hover{border-color:#2f8f4e;color:#246b3b;background:#eaf6ee}
  .rb-2:hover{border-color:#b3341f;color:var(--must);background:var(--must-bg)}

  details.diag{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
    box-shadow:var(--shadow);padding:18px 22px}
  details.diag summary{cursor:pointer;list-style:none;font-size:14px;color:var(--muted)}
  details.diag summary::-webkit-details-marker{display:none}
  details.diag summary::before{content:"⌄ ";color:var(--faint)}
  details.diag[open] summary::before{content:"⌃ "}
  table{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}
  th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line)}
  th{color:var(--muted);font-weight:600;font-size:12px}
  td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
  .warn{background:var(--accent-soft);color:var(--accent-d);border-radius:9px;
    padding:10px 13px;font-size:13px;margin:10px 0}
  .empty{text-align:center;color:var(--faint);padding:40px 0;font-size:15px}
  footer{grid-column:1/-1;margin-top:40px;padding-top:22px;border-top:1px solid var(--line);
    text-align:center;font-size:12.5px;line-height:2;color:var(--faint)}
  footer a{color:var(--accent-d);text-decoration:none}

  @media (max-width:900px){
    .shell{grid-template-columns:1fr;padding:22px 14px 60px;gap:16px}
    aside{position:static;flex-direction:row;overflow-x:auto;padding-bottom:4px}
    aside .panel{min-width:230px;flex:none}
    article{padding:19px 17px}
    h1{font-size:24px}.title{font-size:16.5px}.abs{font-size:15px}
  }
</style>
</head>
<body>
<div class="shell">

<aside>
  <div class="panel">
    <h3>{{ icon('target') }} 本期概览</h3>
    <div class="kv"><span>推送</span><b>{{ counts.total }}</b></div>
    {% if counts.must_read %}<div class="kv"><span>必读</span><b>{{ counts.must_read }}</b></div>{% endif %}
    {% if funnel %}
    <div class="kv"><span>抓取候选</span><b>{{ funnel.raw }}</b></div>
    <div class="kv"><span>去重后</span><b>{{ funnel.dedup }}</b></div>
    {% endif %}
  </div>

  {% if directions %}
  <div class="panel" id="dirFilter">
    <h3>{{ icon('book') }} 研究方向</h3>
    <button class="fitem" aria-pressed="true" data-dir=""><span>全部</span><span>{{ counts.total }}</span></button>
    {% for name, n in directions %}
    <button class="fitem" aria-pressed="false" data-dir="{{ name }}"><span>{{ name }}</span><span>{{ n }}</span></button>
    {% endfor %}
  </div>
  {% endif %}

  {% if counts.must_read %}
  <div class="panel">
    <h3>{{ icon('target') }} 快速筛选</h3>
    <label class="sw"><span>仅看必读</span>
      <input type="checkbox" id="onlyMust" /><i></i></label>
  </div>
  {% endif %}

  <div class="panel">
    <h3>{{ icon('clock') }} 关于本期</h3>
    <div style="font-size:13px;line-height:1.75;color:var(--muted)">
      画像来自你的 Zotero 文库{% if library_size %} {{ library_size }}{% endif %}。<br />
      中文标题与一句话摘要由模型生成，仅供快速筛选，**以原文为准**。
    </div>
  </div>
</aside>

<main>
  <div class="mast">
    <div class="kicker">ZotWatch</div>
    <h1>文献周报</h1>
    <div class="sub">{{ generated_at }} · 北京时间</div>
  </div>
  <div class="count"><b>{{ counts.total }}</b>篇推送{% if funnel %}，来自 {{ funnel.raw }} 篇候选{% endif %}</div>

{% macro card(work, rank) %}
{% set dir = direction(work, problem_names) %}
{% set abs_lead, abs_rest = split_abstract(work.abstract) %}
<article data-dir="{{ dir }}" data-must="{{ 1 if work.label == 'must_read' else 0 }}">
  <div class="badges">
    {% if rank %}<span class="rank">{{ '%02d'|format(rank) }}</span>{% endif %}
    {% if work.label == 'must_read' %}<span class="pill p-must">必读</span>{% endif %}
    {% if dir %}<span class="pill p-dir">{{ dir }}</span>{% endif %}
  </div>

  <a class="title" href="{{ work.url or '#' }}" target="_blank" rel="noopener">{{ work.title }}{%
    if work.published %} <span class="yr">({{ work.published.year }})</span>{% endif %}</a>
  {% if work.extra.get('title_zh') %}<div class="title-zh">{{ work.extra.title_zh }}</div>{% endif %}

  {% if authors_line(work) %}<div class="line">{{ icon('user') }}<span>{{ authors_line(work) }}</span></div>{% endif %}
  {% if source_line(work) %}<div class="line">{{ icon('book') }}<span><em>{{ source_line(work) }}</em></span></div>{% endif %}

  {% if work.extra.get('tldr_zh') %}<div class="tldr">{{ work.extra.tldr_zh }}</div>{% endif %}

  {% if abs_lead %}
  <div class="abs"><span class="lead">{{ abs_lead }}</span>{%
    if abs_rest %}<span class="rest"> {{ abs_rest }}</span>{% endif %}</div>
  {% if abs_rest %}<button class="more" type="button">显示更多</button>{% endif %}
  {% endif %}

  {% for c in work.extra.get('transfer_cards', [])[:2] %}
  <div class="transfer"><b>方法迁移说明卡 · {{ c.topic }}</b>{{ c.use }}<br />
    <span>迁移前核对：{{ c.verify }}</span></div>
  {% endfor %}

  {% set a = anchor(work) %}
  {% if a %}<div class="why">{{ icon('target', 14) }}<span>{{ a }}</span></div>{% endif %}

  <div class="foot">
    <span class="m rel" title="综合相关度，0–1">{{ icon('target', 14) }}相关度
      <b>{{ '%.2f'|format(work.score) }}</b>
      <span class="bar"><i style="width:{{ (work.score * 100)|round|int }}%"></i></span></span>
    <span class="m">被引用 <b>{{ citations(work) }}</b></span>
    {% if work.doi %}<span class="m">{{ icon('link', 14) }}<a href="https://doi.org/{{ work.doi }}"
       target="_blank" rel="noopener">{{ work.doi }}</a></span>{% endif %}
    {% set rates = rating_buttons(work) %}
    {% if rates %}
    <span class="rate" title="将打开公开 GitHub Issue，需你确认提交，请勿填写私人笔记">
      {% for r in rates %}<a class="rb rb-{{ loop.index }}" href="{{ r.url }}" target="_blank" rel="noopener">{{ r.label }}</a>{% endfor %}
    </span>
    {% endif %}
  </div>
</article>
{% endmacro %}

{% if not works and not watched_works and not classic_works and not update_works and not exploration_works %}
<div class="empty">本轮无通过筛选且尚未推送的文献。</div>
{% endif %}

{% if works %}
{% for work in works %}{{ card(work, loop.index) }}{% endfor %}
<div class="empty" id="noMatch" hidden>该方向本期没有推荐。</div>
{% endif %}

{% if watched_works %}
<h2>重点作者新作 <em>{{ watched_works|length }} 篇</em></h2>
<p class="note">已通过主题筛选，按发表时间排列；不代表都达到优先阅读阈值。</p>
{% for work in watched_works %}{{ card(work, 0) }}{% endfor %}
{% endif %}

{% if classic_works %}
<h2>经典文献补漏 <em>{{ classic_works|length }} 篇</em></h2>
<p class="note">来自重点论文或本轮高相关论文的参考文献，不受近期窗口限制。“经典”是栏目名，不等于已人工认定为奠基性论文。</p>
{% for work in classic_works %}{{ card(work, 0) }}{% endfor %}
{% endif %}

{% if exploration_works %}
<h2>跨圈方法发现 <em>{{ exploration_works|length }} 篇</em></h2>
<p class="note">独立语义检索所得，不要求与种子有引文关系；超出近期窗口，不能当作新发表。</p>
{% for work in exploration_works %}{{ card(work, 0) }}{% endfor %}
{% endif %}

{% if update_works %}
<h2>版本与更正提醒 <em>{{ update_works|length }} 篇</em></h2>
<p class="note">独立于普通去重与已读标记；日期为系统发现日期，不代表事件刚发生。</p>
{% for work in update_works %}
<article>
  <a class="title" href="{{ work.url or '#' }}" target="_blank" rel="noopener">{{ work.title }}</a>
  {% if work.extra.get('title_zh') %}<div class="title-zh">{{ work.extra.title_zh }}</div>{% endif %}
  <div class="line">{{ icon('link') }}<span>原论文 DOI {{ work.extra.original_doi }}｜{{ work.extra.update_source }}</span></div>
</article>
{% endfor %}
{% endif %}

{% if coverage_warnings or diagnostics.get('coverage') or diagnostics.get('proposals')
      or diagnostics.get('collaboration_groups') or diagnostics.get('network') %}
<h2>运行诊断</h2>
<details class="diag">
  <summary>展开本轮运行详情（供排查用，不是推荐内容）</summary>
  {% for warning in coverage_warnings %}<div class="warn">{{ warning }}</div>{% endfor %}
  {% if diagnostics.get('coverage') %}
  <table>
    <tr><th>研究方向</th><th class="n">抓取</th><th class="n">主题通过</th>
        <th class="n">去重后</th><th class="n">推送</th><th>状态</th></tr>
    {% for row in diagnostics.coverage %}
    <tr><td>{{ row.facet }}</td><td class="n">{{ row.raw }}</td><td class="n">{{ row.topic }}</td>
        <td class="n">{{ row.dedup }}</td><td class="n">{{ row.delivered }}</td>
        <td>{{ row.status }}{% if row.zero_runs %}（连续 {{ row.zero_runs }} 轮无推送）{% endif %}</td></tr>
    {% endfor %}
  </table>
  {% endif %}
  {% if diagnostics.get('proposals') %}
  <p style="font-size:13px;color:var(--muted)"><b>建议新增关注</b>（待确认，尚未自动加入）：
  {% for p in diagnostics.proposals %}{{ p.name }}（{{ p.count }} 篇）{% if not loop.last %} · {% endif %}{% endfor %}</p>
  {% endif %}
  {% if diagnostics.get('collaboration_groups') %}
  <p style="font-size:13px;color:var(--muted)"><b>持续合作组合</b>　仅表示共同署名证据，不推断师生关系或课题组。<br />
  {% for g in diagnostics.collaboration_groups %}{{ g.names|join(' — ') }}：{{ g.count }} 篇<br />{% endfor %}</p>
  {% endif %}
  {% if diagnostics.get('network') %}
  <p style="font-size:13px;color:var(--muted)"><b>请求与预算</b>　请求 {{ diagnostics.network.requests }}｜缓存命中 {{ diagnostics.network.cache_hits }}｜
  OpenAlex 本地估算余额 {{ diagnostics.network.local_openalex_remaining_usd }}（非账户实时余额）</p>
  {% endif %}
</details>
{% endif %}
</main>

<footer>
  <a href="feed.xml">RSS 订阅</a> · <a href="archive.html">历史推送</a><br />
  全流程只使用标题、摘要与引文元数据，不获取正文。
</footer>

</div>
<script>
document.addEventListener('click', function (e) {
  var more = e.target.closest('.more');
  if (more) {
    var box = more.previousElementSibling;
    box.classList.toggle('open');
    more.textContent = box.classList.contains('open') ? '收起' : '显示更多';
    return;
  }
  var btn = e.target.closest('#dirFilter .fitem');
  if (!btn) return;
  document.querySelectorAll('#dirFilter .fitem').forEach(function (b) {
    b.setAttribute('aria-pressed', String(b === btn));
  });
  apply();
});

function apply() {
  var active = document.querySelector('#dirFilter .fitem[aria-pressed="true"]');
  var want = active ? active.dataset.dir : '';
  var sw = document.getElementById('onlyMust');
  var mustOnly = sw && sw.checked;
  var shown = 0;
  document.querySelectorAll('article[data-dir]').forEach(function (card) {
    var hit = (!want || card.dataset.dir === want) && (!mustOnly || card.dataset.must === '1');
    card.hidden = !hit;
    if (hit) shown++;
  });
  var empty = document.getElementById('noMatch');
  if (empty) empty.hidden = shown > 0;
}
var mustSwitch = document.getElementById('onlyMust');
if (mustSwitch) mustSwitch.addEventListener('change', apply);
</script>
</body>
</html>
"""


def funnel_totals(diagnostics: dict | None) -> dict:
    rows = (diagnostics or {}).get("coverage") or []
    if not rows:
        return {}
    keys = ("raw", "topic", "dedup", "delivered")
    return {key: sum(int(row.get(key) or 0) for row in rows) for key in keys}


def render_html(works: List[RankedWork], output_path: Path | str, *, watched_works: List[RankedWork] | None = None,
                classic_works: List[RankedWork] | None = None, coverage_warnings: List[str] | None = None,
                diagnostics: dict | None = None, update_works: List[RankedWork] | None = None,
                exploration_works: List[RankedWork] | None = None,
                problem_names: dict | None = None, library_size: str = "") -> Path:
    env = Environment(autoescape=True)
    template: Template = env.from_string(_TEMPLATE)
    problem_names = problem_names or {}
    every = [*(works or []), *(watched_works or []), *(classic_works or []), *(exploration_works or [])]

    seen: dict = {}
    for work in every:
        name = direction(work, problem_names)
        if name:
            seen[name] = seen.get(name, 0) + 1

    now = beijing_now()
    rendered = template.render(
        works=works,
        watched_works=watched_works or [],
        classic_works=classic_works or [],
        exploration_works=exploration_works or [],
        update_works=update_works or [],
        coverage_warnings=coverage_warnings or [],
        diagnostics=diagnostics or {},
        funnel=funnel_totals(diagnostics),
        counts={"total": len(every), "must_read": sum(1 for w in every if w.label == "must_read")},
        directions=sorted(seen.items(), key=lambda kv: (-kv[1], kv[0])),
        problem_names=problem_names,
        library_size=library_size,
        generated_at=now.strftime("%Y-%m-%d %H:%M"),
        date_label=now.strftime("%Y-%m-%d"),
        clamp_at=ABSTRACT_CLAMP,
        recommendation_reasons=recommendation_reasons,
        authors_line=authors_line, source_line=source_line, direction=direction,
        anchor=anchor, clamp=clamp, split_abstract=split_abstract, icon=icon,
        rating_buttons=rating_buttons, citations=citations,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    logger.info("Wrote HTML report to %s (%.0f KB)", path, len(rendered.encode("utf-8")) / 1024)
    return path


__all__ = ["render_html", "funnel_totals", "clamp", "split_abstract", "authors_line", "source_line",
           "direction", "anchor", "icon", "rating_buttons", "citations"]
