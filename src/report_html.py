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
  </style>
</head>
<body>
  <h1>ZotWatcher Recommendations</h1>
  <p>Generated at {{ generated_at }}</p>
  <p>近期新作与经典补漏分开呈现；引用关系和相似度是阅读线索，不代表结论正确或方法可直接迁移。</p>
  {% for warning in coverage_warnings %}<p class="meta">覆盖提示：{{ warning }}</p>{% endfor %}
  {% if not works and not watched_works and not classic_works %}<p>本轮无通过筛选且尚未推送的文献。</p>{% endif %}
  {% macro reasons(work) %}
  <ul>{% for reason in recommendation_reasons(work) %}<li>{{ reason }}</li>{% endfor %}</ul>
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
                classic_works: List[RankedWork] | None = None, coverage_warnings: List[str] | None = None) -> Path:
    env = Environment(autoescape=True)
    template: Template = env.from_string(_TEMPLATE)
    rendered = template.render(works=works, watched_works=watched_works or [], classic_works=classic_works or [],
                               coverage_warnings=coverage_warnings or [], recommendation_reasons=recommendation_reasons,
                               generated_at=datetime.utcnow().isoformat())
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    logger.info("Wrote HTML report to %s", path)
    return path


__all__ = ["render_html"]
