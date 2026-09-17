from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

from jinja2 import Environment, Template

from .models import RankedWork
from .citation_watch import recommendation_reasons

logger = logging.getLogger(__name__)

_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ZotWatcher Report {{ generated_at }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; }
    h1 { border-bottom: 2px solid #444; padding-bottom: 0.5rem; }
    article { margin-bottom: 1.5rem; }
    .meta { color: #555; font-size: 0.9rem; }
    table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
    th, td { border: 1px solid #ddd; padding: .5rem; text-align: left; }
    details { background: #f5f7fa; padding: .7rem; margin-top: .6rem; }
  </style>
</head>
<body>
  <h1>ZotWatcher Recommendations</h1>
  <p>Generated at {{ generated_at }}</p>
  <p>近期新作与经典补漏分开呈现；引用关系和相似度是阅读线索，不代表结论正确或方法可直接迁移。</p>
  {% for warning in coverage_warnings %}<p class="meta">覆盖提示：{{ warning }}</p>{% endfor %}
  {% if not works and not watched_works and not classic_works and not update_works %}<p>本轮无通过筛选且尚未推送的文献。</p>{% endif %}
  {% if update_works %}
  <h2>版本与更正提醒</h2>
  <p>独立于普通文献去重及已读标记；以下日期为系统发现日期，不代表事件刚刚发生。</p>
  {% for work in update_works %}<article><h3><a href="{{ work.url }}">{{ work.title }}</a></h3>
    <p>原论文 DOI：{{ work.extra.original_doi }}；证据：{{ work.extra.update_source }}</p></article>{% endfor %}
  {% endif %}
  {% if diagnostics.get('coverage') %}
  <h2>研究问题覆盖诊断</h2>
  <p>同一论文可命中多个方向；数量是本轮候选集统计，不是全网召回率。反馈样本：{{ diagnostics.get('feedback_count', 0) }}。</p>
  <table><thead><tr><th>方向</th><th>抓取</th><th>主题通过</th><th>库内去重后</th><th>推送</th><th>诊断</th></tr></thead><tbody>
  {% for row in diagnostics.coverage %}<tr><td>{{ row.facet }}</td><td>{{ row.raw }}</td><td>{{ row.topic }}</td><td>{{ row.dedup }}</td><td>{{ row.delivered }}</td><td>{{ row.status }}；连续无推送 {{ row.zero_runs }} 个运行日</td></tr>{% endfor %}
  </tbody></table>
  {% endif %}
  {% if diagnostics.get('proposals') %}
  <h2>建议新增关注（待你确认，尚未自动加入）</h2>
  {% for proposal in diagnostics.proposals %}<article><h3>{{ proposal.kind }}：{{ proposal.name }}（{{ proposal.id }}）</h3>
  <p>{{ proposal.reason }}；不同论文数：{{ proposal.count }}</p>
  <ul>{% for paper in proposal.papers %}<li><a href="{{ paper.url or '#' }}">{{ paper.title }}</a> DOI: {{ paper.doi or '未知' }}</li>{% endfor %}</ul></article>{% endfor %}
  {% endif %}
  {% macro reasons(work) %}
  <ul>{% for reason in recommendation_reasons(work) %}<li>{{ reason }}</li>{% endfor %}</ul>
  {% if work.extra.get('transfer_cards') or work.extra.get('abstract_method_signals') %}
  <details><summary>方法迁移说明卡与摘要线索（{{ work.extra.get('evidence_level') }}）</summary>
  {% for card in work.extra.get('transfer_cards', []) %}
    <p><strong>{{ card.topic }}</strong>：{{ card.use }}<br>迁移前检查：{{ card.verify }}<br>
    {{ card.level }}片段：“{{ card.evidence }}”<br>{{ card.status }}</p>
  {% endfor %}
  {% for signal in work.extra.get('abstract_method_signals', []) %}<p>{{ signal.methods|join('、') }}：{{ signal.role }}；“{{ signal.evidence }}”<br>{{ signal.caveat }}</p>{% endfor %}
  <p>{{ work.extra.get('citation_context_status') }}</p>
  </details>
  {% endif %}
  {% if work.extra.get('feedback_links') %}<p>阅读反馈（打开公开 GitHub Issue，需你确认提交；勿附私人笔记）：
  {% for link in work.extra.feedback_links %}<a href="{{ link.url }}">{{ link.name }}</a>{% if not loop.last %} · {% endif %}{% endfor %}</p>{% endif %}
  {% endmacro %}
  {% if classic_works %}
  <section>
    <h2>经典文献补漏</h2>
    <p>来自重点论文或本轮高相关论文的参考文献；不受近期窗口限制，已与文献库及推送历史去重。“经典”是栏目名，不等于已经人工认定为奠基性论文。</p>
    {% for work in classic_works %}
    <article>
      <h3><a href="{{ work.url or '#' }}">{{ work.title }}</a></h3>
      <div class="meta">{{ work.published.strftime('%Y-%m-%d') if work.published else 'Unknown' }} | {{ work.venue or 'Unknown' }} | {{ work.extra.get('research_priority', '') }} | 相似度 {{ '%.3f'|format(work.similarity) }}</div>
      {{ reasons(work) }}
      {% if work.abstract %}<p>{{ work.abstract }}</p>{% endif %}
    </article>
    {% endfor %}
  </section>
  {% endif %}
  {% if watched_works %}
  <section>
    <h2>重点作者新作</h2>
    <p>已通过主题筛选，按发表时间排列；独立于综合推荐前 20 篇，不代表全部达到优先阅读评分阈值。</p>
    {% for work in watched_works %}
    <article>
      <h3><a href="{{ work.url or '#' }}">{{ work.title }}</a></h3>
      <div class="meta">关注作者：{% for author in work.extra.get('watched_authors', []) %}{{ author.name }}{% if not loop.last %}、{% endif %}{% endfor %}
      | {{ work.published.strftime('%Y-%m-%d') if work.published else 'Unknown' }}
      | {{ work.extra.get('research_priority', '其他相关研究') }}</div>
      {% if work.abstract %}<p>{{ work.abstract }}</p>{% endif %}
      {{ reasons(work) }}
    </article>
    {% endfor %}
  </section>
  {% endif %}
  <h2>综合推荐</h2>
  {% for work in works %}
    <article>
      <h2>{{ loop.index }}. <a href="{{ work.url or '#' }}">{{ work.title }}</a></h2>
      <div class="meta">
        <span>Label: {{ work.label }}</span> |
        <span>研究类型: {{ work.extra.get('research_priority', '其他相关研究') }}</span> |
        <span>Score: {{ '%.3f'|format(work.score) }}</span> |
        <span>Similarity: {{ '%.3f'|format(work.similarity) }}</span> |
        <span>Published: {{ work.published.strftime('%Y-%m-%d') if work.published else 'Unknown' }}</span> |
        <span>Venue: {{ work.venue or 'Unknown' }}</span>
      </div>
      {% if work.abstract %}<p>{{ work.abstract }}</p>{% endif %}
      {{ reasons(work) }}
      <div class="meta">
        Authors: {{ work.authors|join(', ') if work.authors else 'Unknown' }}
      </div>
    </article>
  {% endfor %}
</body>
</html>
"""


def render_html(works: List[RankedWork], output_path: Path | str, *, watched_works: List[RankedWork] | None = None,
                classic_works: List[RankedWork] | None = None, coverage_warnings: List[str] | None = None,
                diagnostics: dict | None = None, update_works: List[RankedWork] | None = None) -> Path:
    env = Environment(autoescape=True)
    template: Template = env.from_string(_TEMPLATE)
    rendered = template.render(works=works, watched_works=watched_works or [], classic_works=classic_works or [],
                               coverage_warnings=coverage_warnings or [], recommendation_reasons=recommendation_reasons,
                               diagnostics=diagnostics or {}, update_works=update_works or [],
                               generated_at=datetime.utcnow().isoformat())
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    logger.info("Wrote HTML report to %s", path)
    return path


__all__ = ["render_html"]
