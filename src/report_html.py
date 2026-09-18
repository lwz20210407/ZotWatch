"""The web report: the primary reading surface.

Not the email body -- that is a short reminder rendered by digest_email.py. This
runs in a real browser, so a <style> block, collapsible detail and a client-side
filter all work.

Design borrows from how literature feeds actually get read:

* Semantic Scholar's TLDR is the hero. A truncated abstract still makes you read
  prose to learn what a paper did; one sentence naming material, method and
  result lets you triage twenty papers in a minute. The abstract moves into a
  collapsed block underneath.
* Feedly's "why this matched" chip: each card says which research direction it
  hit and which library paper it sits closest to.
* Researcher/Stork's scan density: venue and date as quiet tags, title dominant,
  everything else subordinate.

Order matters as much as styling. The previous version opened with a coverage
table and three administrative sections, so the papers began below the fold.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from jinja2 import Environment, Template

from .models import RankedWork
from .citation_watch import recommendation_reasons
from .utils import beijing_now

logger = logging.getLogger(__name__)


def truncate(text: str | None, limit: int = 460) -> str:
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def byline(work: RankedWork) -> str:
    authors = list(work.authors or [])
    if not authors:
        return ""
    return f"{authors[0]} 等 {len(authors)} 人" if len(authors) > 2 else "、".join(authors)


def direction(work: RankedWork, problem_names: dict) -> str:
    key = work.extra.get("primary_problem")
    if key:
        return problem_names.get(key, key)
    return work.extra.get("research_priority") or ""


def anchor(work: RankedWork) -> str:
    """Why this paper sits where it does, in one clause."""
    nearest = (work.extra.get("nearest_library_work") or {}).get("title")
    if nearest:
        return f"最接近库内《{truncate(nearest, 42)}》"
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
    --ink:#0d1520; --body:#3d4650; --muted:#68727e; --faint:#98a1ac;
    --line:#e7eaee; --soft:#f0f3f7; --page:#f6f7f9; --card:#fff;
    --link:#1550c8; --lead:#c0392b; --tldr-bg:#f4f8ff; --tldr-bar:#4a7fe0;
    --shadow:0 1px 2px rgba(16,24,40,.03),0 2px 6px rgba(16,24,40,.05);
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{margin:0;background:var(--page);color:var(--body);
    font:400 15px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei","Helvetica Neue",Arial,sans-serif;
    -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
  .wrap{max-width:800px;margin:0 auto;padding:44px 20px 80px}
  a{color:var(--link)}

  /* ---------- masthead ---------- */
  header{margin-bottom:26px}
  .kicker{font:700 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
    letter-spacing:.18em;color:var(--faint);text-transform:uppercase}
  h1{margin:11px 0 0;font-size:30px;line-height:1.2;color:var(--ink);
    letter-spacing:-.025em;font-weight:700}
  .sub{margin-top:8px;font-size:13.5px;color:var(--faint)}
  .stats{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
  .stat{background:var(--card);border:1px solid var(--line);border-radius:10px;
    padding:10px 15px;box-shadow:var(--shadow);min-width:78px}
  .stat b{display:block;font:650 20px/1.2 -apple-system,"Segoe UI",Arial,sans-serif;color:var(--ink)}
  .stat span{font-size:11.5px;color:var(--faint)}

  /* ---------- filter ---------- */
  .filter{display:flex;flex-wrap:wrap;gap:7px;margin:22px 0 4px}
  .chip{border:1px solid var(--line);background:var(--card);border-radius:999px;
    padding:5px 13px;font-size:12.5px;color:var(--muted);cursor:pointer;
    transition:.12s;user-select:none}
  .chip:hover{border-color:#c3ccd8;color:var(--ink)}
  .chip[aria-pressed="true"]{background:var(--ink);border-color:var(--ink);color:#fff}
  .chip i{font-style:normal;opacity:.55;margin-left:5px}

  /* ---------- sections ---------- */
  h2{margin:40px 0 3px;font-size:12px;font-weight:700;letter-spacing:.12em;
    text-transform:uppercase;color:var(--muted)}
  h2 em{font-style:normal;color:var(--faint);font-weight:600;margin-left:3px}
  .note{margin:0 0 15px;font-size:12.5px;line-height:1.7;color:var(--faint)}

  /* ---------- paper card ---------- */
  article{background:var(--card);border:1px solid var(--line);border-radius:13px;
    padding:20px 22px;margin-bottom:12px;box-shadow:var(--shadow);
    transition:border-color .12s,box-shadow .12s}
  article:hover{border-color:#d3dae3;box-shadow:0 2px 4px rgba(16,24,40,.05),0 6px 16px rgba(16,24,40,.07)}
  article.lead{border-left:3px solid var(--lead)}
  article[hidden]{display:none}

  .tags{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin-bottom:11px}
  .rank{font:650 11.5px/1 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--faint)}
  .pill{display:inline-block;padding:2.5px 9px;border-radius:6px;
    font-size:11px;font-weight:600;line-height:1.6;white-space:nowrap}
  .p-must{color:#a02419;background:#fdecea}
  .p-consider{color:#8a5406;background:#fdf3e2}
  .p-dir{color:#2e3d92;background:#edf1fd}
  .p-venue{color:var(--muted);background:var(--soft)}
  .when{margin-left:auto;font-size:11.5px;color:var(--faint);white-space:nowrap}

  .t-en{display:block;font-size:16px;font-weight:600;line-height:1.45;
    color:var(--ink);text-decoration:none;letter-spacing:-.004em}
  .t-en:hover{color:var(--link);text-decoration:underline;text-underline-offset:2px}
  .t-zh{margin-top:6px;font-size:15px;line-height:1.55;color:var(--muted);font-weight:500}

  .tldr{margin-top:13px;background:var(--tldr-bg);border-left:3px solid var(--tldr-bar);
    border-radius:0 8px 8px 0;padding:11px 14px;font-size:14px;line-height:1.75;color:#24344d}

  .meta{margin-top:12px;font-size:12.5px;line-height:1.7;color:var(--faint)}
  .meta .dot{margin:0 6px;opacity:.5}
  .meta .anchor{color:var(--muted)}

  details{margin-top:11px}
  summary{cursor:pointer;font-size:12px;color:var(--faint);list-style:none;user-select:none;
    display:inline-block;padding:2px 0}
  summary::-webkit-details-marker{display:none}
  summary::before{content:"▸";display:inline-block;width:13px;color:var(--faint)}
  details[open] summary::before{content:"▾"}
  summary:hover{color:var(--link)}
  .panel{margin-top:9px;padding:12px 14px;background:var(--soft);border-radius:8px;
    font-size:13px;line-height:1.75;color:var(--body)}
  .panel .ev{font-size:12px;color:var(--muted);margin-top:9px}
  .fb{margin-top:12px;font-size:12px;color:var(--faint)}
  .fb a{text-decoration:none;margin-right:10px}

  table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:10px}
  th,td{padding:7px 9px;text-align:left;border-bottom:1px solid var(--line)}
  th{color:var(--muted);font-weight:600;font-size:11.5px}
  td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
  .warn{color:#8a5406;background:#fdf3e2;border-radius:8px;padding:10px 13px;
    font-size:12.5px;margin-bottom:8px}
  .empty{text-align:center;color:var(--faint);padding:34px 0;font-size:14px}

  footer{margin-top:50px;padding-top:22px;border-top:1px solid var(--line);
    font-size:12px;color:var(--faint);text-align:center;line-height:2}
  footer a{text-decoration:none}

  @media (max-width:640px){
    .wrap{padding:28px 14px 60px}
    h1{font-size:24px}
    article{padding:17px 16px}
    .t-en{font-size:15px}.t-zh{font-size:14px}.tldr{font-size:13.5px}
    .when{margin-left:0;width:100%}
  }
</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="kicker">ZotWatch</div>
  <h1>文献周报</h1>
  <div class="sub">{{ generated_at }} · 北京时间 · 基于你 Zotero 文库 {{ library_size or '' }} 的兴趣画像</div>
  <div class="stats">
    <div class="stat"><b>{{ counts.total }}</b><span>本期推送</span></div>
    {% if counts.must_read %}<div class="stat"><b>{{ counts.must_read }}</b><span>必读</span></div>{% endif %}
    {% if funnel %}<div class="stat"><b>{{ funnel.raw }}</b><span>抓取候选</span></div>
    <div class="stat"><b>{{ funnel.dedup }}</b><span>去重后</span></div>{% endif %}
  </div>
  {% if directions %}
  <div class="filter" id="filter">
    <button class="chip" aria-pressed="true" data-dir="">全部<i>{{ counts.total }}</i></button>
    {% for name, n in directions %}<button class="chip" aria-pressed="false" data-dir="{{ name }}">{{ name }}<i>{{ n }}</i></button>{% endfor %}
  </div>
  {% endif %}
</header>

{% macro card(work, rank, lead) %}
{% set dir = direction(work, problem_names) %}
<article class="{{ 'lead' if lead else '' }}" data-dir="{{ dir }}">
  <div class="tags">
    {% if rank %}<span class="rank">{{ '%02d'|format(rank) }}</span>{% endif %}
    {% if work.label == 'must_read' %}<span class="pill p-must">必读</span>
    {% elif work.label == 'consider' %}<span class="pill p-consider">可看</span>{% endif %}
    {% if dir %}<span class="pill p-dir">{{ dir }}</span>{% endif %}
    {% if work.venue %}<span class="pill p-venue">{{ work.venue }}</span>{% endif %}
    <span class="when">{{ work.published.strftime('%Y-%m-%d') if work.published else '' }}</span>
  </div>

  <a class="t-en" href="{{ work.url or '#' }}" target="_blank" rel="noopener">{{ work.title }}</a>
  {% if work.extra.get('title_zh') %}<div class="t-zh">{{ work.extra.title_zh }}</div>{% endif %}

  {% if work.extra.get('tldr_zh') %}<div class="tldr">{{ work.extra.tldr_zh }}</div>{% endif %}

  {% set who = byline(work) %}{% set why = anchor(work) %}
  {% if who or why %}
  <div class="meta">{{ who }}{% if who and why %}<span class="dot">·</span>{% endif %}<span class="anchor">{{ why }}</span></div>
  {% endif %}

  {% set evidence = recommendation_reasons(work) %}
  {% if work.abstract or evidence %}
  <details>
    <summary>摘要原文{% if evidence %}与引用证据{% endif %}</summary>
    <div class="panel">
      {% if work.abstract %}{{ truncate(work.abstract) }}{% endif %}
      {% if evidence %}<div class="ev">{% for row in evidence %}{{ row }}<br />{% endfor %}</div>{% endif %}
      {% for c in work.extra.get('transfer_cards', []) %}
        <div class="ev"><b>方法迁移说明卡 · {{ c.topic }}</b>：{{ c.use }}｜迁移前核对：{{ c.verify }}</div>
      {% endfor %}
    </div>
  </details>
  {% endif %}

  {% if work.extra.get('feedback_links') %}
  <div class="fb" title="将打开公开 GitHub Issue，请勿填写私人笔记">这篇{% for l in work.extra.feedback_links %}<a href="{{ l.url }}" target="_blank" rel="noopener">{{ l.name }}</a>{% endfor %}<span style="opacity:.75">（打开公开 GitHub Issue，需你确认提交）</span></div>
  {% endif %}
</article>
{% endmacro %}

{% if not works and not watched_works and not classic_works and not update_works and not exploration_works %}
<div class="empty">本轮无通过筛选且尚未推送的文献。</div>
{% endif %}

{% if works %}
<h2>本期推荐<em>{{ works|length }}</em></h2>
<p class="note">按相关度排序。相似度与引用关系是阅读线索，不代表结论正确或方法可直接迁移。</p>
{% for work in works %}{{ card(work, loop.index, work.label == 'must_read') }}{% endfor %}
<div class="empty" id="noMatch" hidden>该方向本期没有推荐。</div>
{% endif %}

{% if watched_works %}
<h2>重点作者新作<em>{{ watched_works|length }}</em></h2>
<p class="note">已通过主题筛选，按发表时间排列；不代表都达到优先阅读阈值。</p>
{% for work in watched_works %}{{ card(work, 0, false) }}{% endfor %}
{% endif %}

{% if classic_works %}
<h2>经典文献补漏<em>{{ classic_works|length }}</em></h2>
<p class="note">来自重点论文或本轮高相关论文的参考文献，不受近期窗口限制。“经典”是栏目名，不等于已人工认定为奠基性论文。</p>
{% for work in classic_works %}{{ card(work, 0, false) }}{% endfor %}
{% endif %}

{% if exploration_works %}
<h2>跨圈方法发现<em>{{ exploration_works|length }}</em></h2>
<p class="note">独立语义检索所得，不要求与种子有引文关系；超出近期窗口，不能当作新发表。</p>
{% for work in exploration_works %}{{ card(work, 0, false) }}{% endfor %}
{% endif %}

{% if update_works %}
<h2>版本与更正提醒<em>{{ update_works|length }}</em></h2>
<p class="note">独立于普通去重与已读标记；日期为系统发现日期，不代表事件刚发生。</p>
{% for work in update_works %}
<article>
  <a class="t-en" href="{{ work.url or '#' }}" target="_blank" rel="noopener">{{ work.title }}</a>
  {% if work.extra.get('title_zh') %}<div class="t-zh">{{ work.extra.title_zh }}</div>{% endif %}
  <div class="meta">原论文 DOI {{ work.extra.original_doi }}<span class="dot">·</span>{{ work.extra.update_source }}</div>
</article>
{% endfor %}
{% endif %}

{% if coverage_warnings or diagnostics.get('coverage') or diagnostics.get('proposals')
      or diagnostics.get('collaboration_groups') or diagnostics.get('network') %}
<h2>运行诊断</h2>
<p class="note">供排查用，不是推荐内容。</p>
<article>
  <details>
    <summary>展开本轮运行详情</summary>
    <div class="panel">
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
      <div class="ev"><b>建议新增关注</b>（待确认，尚未自动加入）：
      {% for p in diagnostics.proposals %}{{ p.name }}（{{ p.count }} 篇）{% if not loop.last %} · {% endif %}{% endfor %}</div>
      {% endif %}
      {% if diagnostics.get('collaboration_groups') %}
      <div class="ev"><b>持续合作组合</b>　仅表示共同署名证据，不推断师生关系或课题组。<br />
      {% for g in diagnostics.collaboration_groups %}{{ g.names|join(' — ') }}：{{ g.count }} 篇<br />{% endfor %}</div>
      {% endif %}
      {% if diagnostics.get('network') %}
      <div class="ev"><b>请求与预算</b>　请求 {{ diagnostics.network.requests }}｜缓存命中 {{ diagnostics.network.cache_hits }}｜
      OpenAlex 本地估算余额 {{ diagnostics.network.local_openalex_remaining_usd }}（非账户实时余额）</div>
      {% endif %}
    </div>
  </details>
</article>
{% endif %}

<footer>
  <a href="feed.xml">RSS 订阅</a> · <a href="archive.html">历史推送</a><br />
  中文标题与摘要由模型生成，仅供快速筛选，以原文为准。<br />
  全流程只使用标题、摘要与引文元数据，不获取正文。
</footer>

</div>
<script>
(function () {
  var bar = document.getElementById('filter');
  if (!bar) return;
  var cards = document.querySelectorAll('article[data-dir]');
  var empty = document.getElementById('noMatch');
  bar.addEventListener('click', function (e) {
    var btn = e.target.closest('.chip');
    if (!btn) return;
    var want = btn.dataset.dir;
    bar.querySelectorAll('.chip').forEach(function (c) {
      c.setAttribute('aria-pressed', String(c === btn));
    });
    var shown = 0;
    cards.forEach(function (card) {
      var hit = !want || card.dataset.dir === want;
      card.hidden = !hit;
      if (hit) shown++;
    });
    if (empty) empty.hidden = shown > 0;
  });
})();
</script>
</body>
</html>
"""


def funnel_totals(diagnostics: dict | None) -> dict:
    """Aggregate the per-facet coverage table into one retrieval funnel."""
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
    directions = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))

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
        directions=directions,
        problem_names=problem_names,
        library_size=library_size,
        generated_at=now.strftime("%Y-%m-%d %H:%M"),
        date_label=now.strftime("%Y-%m-%d"),
        recommendation_reasons=recommendation_reasons,
        byline=byline, direction=direction, anchor=anchor, truncate=truncate,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    logger.info("Wrote HTML report to %s (%.0f KB)", path, len(rendered.encode("utf-8")) / 1024)
    return path


__all__ = ["render_html", "funnel_totals", "truncate", "byline", "direction", "anchor"]
