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
import re
import zlib
from pathlib import Path
from urllib.parse import urlencode
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


def published_label(work: RankedWork) -> str:
    """Publication date at the precision the source actually gave.

    Crossref records that carry only a year are stored as YYYY-01-01, so printing
    a full date for every paper would claim a day the publisher never stated.
    """
    if not work.published:
        return ""
    precision = (work.extra or {}).get("date_precision") or "day"
    if precision == "year":
        return work.published.strftime("%Y")
    if precision == "month":
        return work.published.strftime("%Y-%m")
    return work.published.strftime("%Y-%m-%d")


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


# One hue per research direction, assigned by a stable hash of the name so the
# same direction keeps its colour across runs and any new facet still gets one.
# Colour carries meaning here -- which of the seven directions a paper belongs to
# -- rather than alternating just for variety.
DIRECTION_HUES = (
    ("#d9541f", "#fdeee6"),
    ("#1f6fb5", "#e9f2fb"),
    ("#2f8f4e", "#e9f6ee"),
    ("#8a4bb8", "#f3ecfa"),
    ("#b3341f", "#fceee9"),
    ("#0f8a8a", "#e6f6f6"),
    ("#a8781a", "#fbf3e1"),
    ("#5c6bc0", "#eceefb"),
)


def direction_hue(name: str, order: dict | None = None) -> dict:
    """Colour for a research direction.

    When the caller knows every direction on the page it passes `order`, which
    assigns colours by position and so keeps them distinct. A name-derived hash is
    only the fallback; with seven directions over an eight-colour palette it
    collided, and two directions sharing a colour defeats the point.
    """
    if not name:
        return {"fg": "#7c736c", "bg": "#f3f1ed"}
    if order and name in order:
        index = order[name] % len(DIRECTION_HUES)
    else:
        index = zlib.crc32(name.encode("utf-8")) % len(DIRECTION_HUES)
    fg, bg = DIRECTION_HUES[index]
    return {"fg": fg, "bg": bg}


def anchor(work: RankedWork) -> str:
    nearest = (work.extra.get("nearest_library_work") or {}).get("title")
    if nearest:
        # Shown in full: a truncated title is not enough to recognise which of
        # your own papers the recommendation is anchored to.
        return "最接近库内《" + " ".join(str(nearest).split()) + "》"
    if work.extra.get("cites_seeds"):
        return f"引用了 {len(work.extra['cites_seeds'])} 篇你关注的论文"
    if work.extra.get("referenced_by"):
        return f"被 {len(work.extra['referenced_by'])} 篇相关论文引用"
    return ""


def past_issues(current: Path, limit: int = 8) -> list:
    """Previously published issues, newest first.

    Each report embeds its issue number and item count as meta tags, so the
    archive can be listed without re-deriving anything. Reports written before
    those tags existed fall back to the date in the filename.
    """
    directory = Path(current).parent
    if not directory.is_dir():
        return []
    rows = []
    for path in sorted(directory.glob("report-*.html"), key=lambda f: f.name, reverse=True):
        if path.name == Path(current).name:
            continue
        stamp = path.stem.replace("report-", "")
        if len(stamp) != 8 or not stamp.isdigit():
            continue
        head = path.read_text(encoding="utf-8", errors="replace")[:2500]
        meta = dict(re.findall(r'name="zotwatch:(\w+)"\s+content="([^"]*)"', head))
        rows.append({
            "file": path.name,
            "date": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}",
            "short": f"{int(stamp[4:6])} 月 {int(stamp[6:])} 日",
            "no": meta.get("issue") or "",
            "count": meta.get("count") or "",
        })
        if len(rows) >= limit:
            break
    return rows


def author_candidates(diagnostics: dict | None, repository: str = "", limit: int = 5) -> list:
    """Author proposals, with a prefilled link that requests adding them.

    propose_tracking already accumulates these across issues; they were only ever
    shown inside the collapsed diagnostics block, where an actionable suggestion
    is of no use.
    """
    rows = [row for row in (diagnostics or {}).get("proposals") or []
            if row.get("kind") == "作者候选"][:limit]
    for row in rows:
        if repository and row.get("id"):
            body = (f"请把以下作者加入 config/authors.yaml：\n\n"
                    f"- name: {row.get('name', '')}\n"
                    f"  openalex_ids: [{row['id']}]\n"
                    f"  reason: 本期推荐中出现 {row.get('count', 0)} 篇高相关论文\n")
            row["add_url"] = (f"https://github.com/{repository}/issues/new?"
                              + urlencode({"title": f"[ZotWatch] 关注作者 {row.get('name', '')}",
                                           "body": body}))
    return rows


def library_anchors(works, limit: int = 5) -> list:
    """Which of the user's own papers this batch clusters around.

    Each card names its single nearest library paper; aggregating them answers a
    question no card can: what in my library is this week's literature actually
    circling. Not available anywhere else on the page.
    """
    counts: dict = {}
    for work in works:
        title = (work.extra.get("nearest_library_work") or {}).get("title")
        if title:
            title = " ".join(str(title).split())
            counts[title] = counts.get(title, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def watched_hits(works, limit: int = 6) -> list:
    """Tracked authors who published in this batch."""
    counts: dict = {}
    for work in works:
        for author in work.extra.get("watched_authors") or []:
            name = author.get("name") if isinstance(author, dict) else str(author)
            if name:
                counts[name] = counts.get(name, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def scatter(works, problem_names: dict, *, width: int = 252, height: int = 158,
            hue_order: dict | None = None) -> dict:
    """Points for the recency/relevance bubble chart.

    Borrowed in spirit from Connected Papers, whose graph encodes several
    dimensions at once: here x is how recently a paper appeared, y is its
    relevance to the library, bubble area is citation count and colour is the
    research direction -- the same hue the card and the sidebar use.
    """
    rows = [w for w in works if w.published and w.score is not None]
    if not rows:
        return {}
    now = beijing_now()
    ages = [max((now - w.published.astimezone(now.tzinfo)).days, 0) for w in rows]
    scores = [float(w.score) for w in rows]
    max_age = max(max(ages), 1)
    lo, hi = min(scores), max(scores)
    span = max(hi - lo, 0.06)
    pad_l, pad_r, pad_t, pad_b = 8, 10, 10, 16
    points = []
    for work, age, score in zip(rows, ages, scores):
        x = pad_l + (width - pad_l - pad_r) * (1 - age / max_age)
        y = pad_t + (height - pad_t - pad_b) * (1 - (score - lo) / span)
        radius = 3.2 + min(float(citations(work)), 200.0) ** 0.42
        hue = direction_hue(direction(work, problem_names), hue_order)
        points.append({"x": round(x, 1), "y": round(y, 1), "r": round(min(radius, 11.0), 1),
                       "color": hue["fg"], "title": work.title, "score": score,
                       "cites": citations(work), "age": age})
    return {"points": points, "width": width, "height": height,
            "max_age": max_age, "lo": lo, "hi": hi}


def venue_counts(works, limit: int = 5) -> list:
    counts: dict = {}
    for work in works:
        if work.venue:
            counts[work.venue] = counts.get(work.venue, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>文献周刊{% if issue_no %} 第 {{ issue_no }} 期{% endif %} · {{ date_label }}</title>
{# Machine-readable so the archive card can list past issues without reparsing HTML. #}
<meta name="zotwatch:issue" content="{{ issue_no }}" />
<meta name="zotwatch:count" content="{{ counts.total }}" />
<meta name="zotwatch:date" content="{{ date_label }}" />
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

  /* Fill the viewport instead of stranding a fixed 1180px column in the middle of
     a wide monitor, but cap it so text lines stay readable. */
  /* Three columns: filters left, the paper column in the middle at a readable
     width, data cards right. A single centred column stranded most of a wide
     monitor as empty margin. */
  .shell{width:100%;max-width:1780px;margin:0 auto;padding:28px clamp(14px,2vw,34px) 80px;
    display:grid;grid-template-columns:250px minmax(0,1fr) 292px;
    gap:clamp(16px,1.5vw,26px);align-items:start}
  .cards{display:grid;grid-template-columns:1fr;gap:16px;align-content:start}
  @media (max-width:1400px){
    .shell{grid-template-columns:240px minmax(0,1fr)}
    aside.right{grid-column:1/-1;display:grid;
      grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px;
      position:static;max-height:none;overflow:visible}
  }

  /* ---------------- sidebar ---------------- */
  aside{position:sticky;top:22px;max-height:calc(100vh - 44px);
    display:flex;flex-direction:column;gap:14px;
    overflow-y:auto;overscroll-behavior:contain;padding-right:5px;margin-right:-5px}
  aside::-webkit-scrollbar{width:7px}
  aside::-webkit-scrollbar-thumb{background:var(--line);border-radius:4px}
  aside::-webkit-scrollbar-thumb:hover{background:#d5d0ca}
  aside{scrollbar-width:thin;scrollbar-color:var(--line) transparent}
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
  .fitem span:first-child{display:flex;align-items:center;gap:8px;min-width:0}
  .dot{width:8px;height:8px;border-radius:50%;flex:none}
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
  /* Masthead, built like a periodical's: a wordmark, a bilingual title lockup,
     an issue line and a standfirst. "文献周报" alone was four characters floating
     at the top of a page this wide. */
  .mast{margin:0 0 26px;padding-bottom:24px;border-bottom:1px solid var(--line);
    position:relative;text-align:center}
  .mast::after{content:"";position:absolute;left:50%;transform:translateX(-50%);
    bottom:-1px;width:96px;height:3px;border-radius:2px;background:var(--accent)}

  .brand{display:flex;align-items:center;justify-content:center;gap:11px;flex-wrap:wrap}
  .mark{font-size:13px;font-weight:700;letter-spacing:.02em;color:var(--accent);
    padding:3px 10px;border:1.5px solid var(--accent);border-radius:7px}
  .brand-sep{width:22px;height:1px;background:var(--line)}
  .brand-note{font-size:12.5px;color:var(--faint)}

  .lockup{display:flex;flex-direction:column;align-items:center;gap:10px;margin-top:18px}
  h1{margin:0;font-size:clamp(27px,3vw,40px);line-height:1.15;font-weight:700;
    letter-spacing:.03em;color:var(--ink)}
  .latin{font-size:clamp(11px,.85vw,13px);font-weight:600;letter-spacing:.26em;
    text-transform:uppercase;color:var(--faint);white-space:nowrap}

  .issue{margin-top:14px;font-size:14px;color:var(--body);font-weight:500}
  .issue .no{color:var(--accent-d);font-weight:650}
  .issue .sep{margin:0 9px;color:var(--faint);font-weight:400}

  .lede{margin:13px auto 0;max-width:58ch;font-size:14.5px;line-height:1.85;
    color:var(--muted)}
  .lede b{color:var(--ink);font-weight:650}

  h2{margin:34px 0 12px;font-size:15px;font-weight:600;color:var(--ink);
    display:flex;align-items:center;gap:9px}
  h2 em{font-style:normal;font-size:12.5px;font-weight:500;color:var(--faint)}
  .note{margin:-6px 0 14px;font-size:13px;line-height:1.7;color:var(--faint)}

  /* ---------------- card ---------------- */
  article{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
    border-left:4px solid var(--hue,var(--line));
    box-shadow:var(--shadow);padding:22px clamp(18px,1.6vw,28px);margin:0;
    display:flex;flex-direction:column;break-inside:avoid}
  article .foot{margin-top:auto}
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

  .tldr{margin:14px 0 0;padding:12px 15px;background:var(--hue-soft,var(--accent-soft));
    border-left:3px solid var(--hue,var(--accent));border-radius:0 10px 10px 0;
    font-size:15px;line-height:1.7;color:#33291f}

  .abs{margin-top:14px;font-size:16px;line-height:1.5;color:var(--body)}
  .abs .lead::before{content:"摘要：";color:var(--muted)}
  .abs .rest{display:none}
  .abs.open .rest{display:inline}
  .absbar{display:flex;align-items:center;gap:17px;margin-top:9px}
  .more,.alt{background:none;border:0;padding:0;cursor:pointer;font:inherit;
    display:inline-flex;align-items:center;gap:5px}
  .more{font-size:14px;color:var(--accent)}
  .more::before{content:"⌄";font-size:15px;line-height:1}
  .more.open::before{content:"⌃"}
  .alt{font-size:13px;color:var(--muted);gap:6px}
  .alt::before{content:"译";font-size:10.5px;font-weight:700;border:1px solid currentColor;
    border-radius:3px;padding:0 3.5px;line-height:1.4}
  .more:hover,.alt:hover{color:var(--accent-d)}
  .abs.zh{color:var(--body);font-size:15px;border-left:2px solid var(--hue,var(--line));
    padding-left:13px;margin-top:11px}

  /* method-transfer note: project-specific, no public search engine has it */
  .transfer{margin-top:12px;padding:10px 13px;background:var(--soft);border-radius:10px;
    font-size:13px;line-height:1.7;color:var(--muted)}
  .transfer b{color:var(--ink);font-weight:600;margin-right:6px}

  /* why this paper is here -- the thing a public search engine cannot show */
  .why{display:flex;align-items:flex-start;gap:7px;margin-top:13px;font-size:13.5px;
    line-height:1.6;
    color:var(--hue,var(--accent-d));background:var(--hue-soft,var(--accent-soft));
    border-radius:9px;padding:8px 12px}
  .why .ico{color:var(--hue,var(--accent))}

  .foot{display:flex;flex-wrap:wrap;align-items:center;gap:14px;margin-top:16px;
    padding-top:14px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}
  .foot a{color:var(--muted);text-decoration:none}
  .foot a:hover{color:var(--accent-d)}
  .foot .m{display:inline-flex;align-items:center;gap:6px}
  .foot .m b{font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums}
  .bar{display:inline-block;width:52px;height:5px;border-radius:3px;background:var(--line);
    overflow:hidden;margin-left:2px}
  .bar i{display:block;height:100%;background:var(--hue,var(--accent));border-radius:3px}

  .rate{margin-left:auto;display:flex;gap:7px}
  .rb{padding:5px 12px;border:1px solid var(--line);border-radius:8px;
    font-size:12.5px;color:var(--muted);text-decoration:none;transition:.12s;background:var(--card)}
  .rb:hover{border-color:var(--accent);color:var(--accent-d);background:var(--accent-soft)}
  .rb-1:hover{border-color:#2f8f4e;color:#246b3b;background:#eaf6ee}
  .rb-2:hover{border-color:#b3341f;color:var(--must);background:var(--must-bg)}

  /* ---------------- right rail ---------------- */
  aside.right{top:22px}
  .hint{font-size:11.5px;line-height:1.65;color:var(--faint);margin:-5px 0 10px}
  .toc{display:flex;flex-direction:column;gap:1px;margin:-4px -6px 0}
  .toc a{display:flex;align-items:baseline;gap:7px;padding:7px 8px;border-radius:8px;
    font-size:12.5px;line-height:1.5;color:var(--body);text-decoration:none;
    transition:background .12s}
  .toc a:hover{background:var(--soft);color:var(--ink)}
  .toc .dot{flex:none;align-self:center}
  .toc .n{flex:none;font-size:11px;font-weight:600;color:var(--faint);
    font-variant-numeric:tabular-nums}
  .toc .tt{flex:1;min-width:0;overflow:hidden;display:-webkit-box;
    -webkit-line-clamp:2;-webkit-box-orient:vertical}
  .toc em{flex:none;font-style:normal;font-size:10px;font-weight:600;color:var(--must);
    background:var(--must-bg);border-radius:4px;padding:1px 5px}
  .brow.off{opacity:.42}
  .row a.rt{color:var(--accent-d);text-decoration:none}
  .row a.rt:hover{text-decoration:underline}
  .sub-h{font-size:12.5px;font-weight:600;color:var(--ink);margin:0 0 7px}
  .cands{display:flex;flex-direction:column;gap:11px}
  .cand{padding:10px 12px;background:var(--soft);border-radius:10px}
  .cand-h{display:flex;align-items:baseline;gap:7px;flex-wrap:wrap}
  .cand-h .nm{font-size:13px;font-weight:600;color:var(--ink)}
  .cand-h .new{font-style:normal;font-size:10px;font-weight:600;color:#246b3b;
    background:#eaf6ee;border-radius:4px;padding:1px 5px}
  .cand-h b{margin-left:auto;font-size:12px;font-weight:600;color:var(--muted);
    font-variant-numeric:tabular-nums}
  .aff{margin-top:3px;font-size:11.5px;color:var(--faint);overflow-wrap:anywhere}
  .cand-p{margin-top:6px;display:flex;flex-direction:column;gap:4px}
  .cand-p a{font-size:11.5px;line-height:1.55;color:var(--muted);text-decoration:none;
    overflow-wrap:anywhere;display:block}
  .cand-p a:hover{color:var(--accent-d)}
  .cand-p a::before{content:"· "}
  .add{display:inline-block;margin-top:8px;font-size:11.5px;color:var(--accent-d);
    text-decoration:none;font-weight:600}
  .add:hover{text-decoration:underline}
  .issues{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:1px}
  .issues a{display:flex;align-items:baseline;gap:8px;padding:7px 8px;margin:0 -6px;
    border-radius:8px;text-decoration:none;font-size:12.5px;transition:background .12s}
  .issues a:hover{background:var(--soft)}
  .issues .no{flex:none;font-weight:600;color:var(--ink)}
  .issues .when{flex:1;min-width:0;color:var(--faint)}
  .issues .cnt{flex:none;color:var(--muted);font-variant-numeric:tabular-nums}
  .links{display:flex;gap:14px;margin-top:13px;padding-top:11px;
    border-top:1px solid var(--line);font-size:12.5px}
  .links a{color:var(--accent-d);text-decoration:none}
  .links a:hover{text-decoration:underline}
  article{scroll-margin-top:22px}
  .rows{display:flex;flex-direction:column;gap:9px}
  /* Titles and journal names wrap instead of being cut with an ellipsis; a
     clipped title is not enough to recognise the paper it refers to. */
  .row{display:flex;align-items:flex-start;gap:9px;font-size:12.5px;line-height:1.6;
    color:var(--muted)}
  .row .rt{flex:1;min-width:0;overflow-wrap:anywhere}
  .row .dot{display:inline-block;margin-right:7px;vertical-align:1px}
  .row b{flex:none;font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums}
  .bars{display:flex;flex-direction:column;gap:9px;margin-top:2px}
  .brow{font-size:12.5px;color:var(--muted)}
  .brow .t{display:flex;justify-content:space-between;gap:8px;margin-bottom:4px}
  .brow .t b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
  .track{height:6px;border-radius:4px;background:var(--soft);overflow:hidden}
  .track i{display:block;height:100%;border-radius:4px}
  .funnel{display:flex;flex-direction:column;gap:7px;margin-top:2px}
  .fstep{font-size:12.5px;color:var(--muted)}
  .fstep .t{display:flex;justify-content:space-between;margin-bottom:3px}
  .fstep .t b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
  .fbar{height:22px;border-radius:6px;background:var(--accent-soft);position:relative}
  .fbar i{position:absolute;inset:0 auto 0 0;border-radius:6px;background:var(--accent);opacity:.82}

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

  @media (max-width:900px){
    .shell{grid-template-columns:1fr;padding:22px 14px 60px;gap:16px}
    .cards{gap:14px}
    aside{position:static;max-height:none;overflow:visible;padding-right:0;margin-right:0}
    aside.left{flex-direction:column}
    aside.left .panel{min-width:0}
    aside{position:static;flex-direction:row;overflow-x:auto;padding-bottom:4px}
    aside .panel{min-width:230px;flex:none}
    article{padding:19px 17px}
    h1{font-size:24px}.title{font-size:16.5px}.abs{font-size:15px}
  }
</style>
</head>
<body>
<div class="shell">

<aside class="left">
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
    {% set h = direction_hue(name, hue_order) %}
    <button class="fitem" aria-pressed="false" data-dir="{{ name }}">
      <span><i class="dot" style="background:{{ h.fg }}"></i>{{ name }}</span><span>{{ n }}</span></button>
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
      中文标题与一句话摘要由模型生成，仅供快速筛选，<b style="color:var(--ink)">以原文为准</b>。
    </div>
  </div>
</aside>

<main>
  <header class="mast">
    <div class="brand">
      <span class="mark">ZotWatch</span>
      <span class="brand-sep"></span>
      <span class="brand-note">基于 Zotero 文库画像的文献追踪</span>
    </div>

    <div class="lockup">
      <h1>每周最新文献订阅与追踪</h1>
      <div class="latin">Weekly&nbsp;Research&nbsp;Digest</div>
    </div>

    <div class="issue">
      {% if issue_no %}<span class="no">第 {{ issue_no }} 期</span><span class="sep">·</span>{% endif %}
      <span>{{ date_cn }}</span><span class="sep">·</span>
      <span>{{ generated_at_time }} 北京时间</span>
    </div>

    <p class="lede">
      本期 <b>{{ counts.total }}</b> 篇{% if counts.must_read %}，其中 <b>{{ counts.must_read }}</b> 篇必读{% endif %}，
      从近 {{ window_days }} 天的{% if funnel %} <b>{{ funnel.raw }}</b> {% else %} {% endif %}篇候选中筛出，
      按你 Zotero 文库{% if library_size %} <b>{{ library_size }}</b> {% endif %}的兴趣画像排序。
    </p>
  </header>

{% macro card(work, rank) %}
{% set dir = direction(work, problem_names) %}
{# The original is what is shown; the translation is one click away. #}
{% set zh = work.extra.get('abstract_zh') %}
{% set abs_lead, abs_rest = split_abstract(work.abstract) %}
{% set hue = direction_hue(dir, hue_order) %}
<article id="p{{ rank }}" data-dir="{{ dir }}" data-must="{{ 1 if work.label == 'must_read' else 0 }}"
         style="--hue:{{ hue.fg }};--hue-soft:{{ hue.bg }}">
  <div class="badges">
    {% if rank %}<span class="rank">{{ '%02d'|format(rank) }}</span>{% endif %}
    {% if work.label == 'must_read' %}<span class="pill p-must">必读</span>{% endif %}
    {% if dir %}<span class="pill p-dir" style="color:{{ hue.fg }};background:{{ hue.bg }}">{{ dir }}</span>{% endif %}
  </div>

  <a class="title" href="{{ work.url or '#' }}" target="_blank" rel="noopener">{{ work.title }}{%
    set when = published_label(work) %}{% if when %} <span class="yr">({{ when }})</span>{% endif %}</a>
  {% if work.extra.get('title_zh') %}<div class="title-zh">{{ work.extra.title_zh }}</div>{% endif %}

  {% if authors_line(work) %}<div class="line">{{ icon('user') }}<span>{{ authors_line(work) }}</span></div>{% endif %}
  {% if source_line(work) %}<div class="line">{{ icon('book') }}<span><em>{{ source_line(work) }}</em></span></div>{% endif %}

  {% if work.extra.get('tldr_zh') %}<div class="tldr">{{ work.extra.tldr_zh }}</div>{% endif %}

  {% if abs_lead %}
  <div class="abs"><span class="lead">{{ abs_lead }}</span>{%
    if abs_rest %}<span class="rest"> {{ abs_rest }}</span>{% endif %}</div>
  <div class="absbar">
    {% if abs_rest %}<button class="more" type="button">显示更多</button>{% endif %}
    {% if zh %}<button class="alt" type="button">翻译</button>{% endif %}
  </div>
  {% if zh %}<div class="abs zh" hidden>{{ zh }}</div>{% endif %}
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
<div class="cards">{% for work in works %}{{ card(work, loop.index) }}{% endfor %}</div>
<div class="empty" id="noMatch" hidden>该方向本期没有推荐。</div>
{% endif %}

{% if watched_works %}
<h2>重点作者新作 <em>{{ watched_works|length }} 篇</em></h2>
<p class="note">已通过主题筛选，按发表时间排列；不代表都达到优先阅读阈值。</p>
<div class="cards">{% for work in watched_works %}{{ card(work, 0) }}{% endfor %}</div>
{% endif %}

{% if classic_works %}
<h2>经典文献补漏 <em>{{ classic_works|length }} 篇</em></h2>
<p class="note">来自重点论文或本轮高相关论文的参考文献，不受近期窗口限制。“经典”是栏目名，不等于已人工认定为奠基性论文。</p>
<div class="cards">{% for work in classic_works %}{{ card(work, 0) }}{% endfor %}</div>
{% endif %}

{% if exploration_works %}
<h2>跨圈方法发现 <em>{{ exploration_works|length }} 篇</em></h2>
<p class="note">独立语义检索所得，不要求与种子有引文关系；超出近期窗口，不能当作新发表。</p>
<div class="cards">{% for work in exploration_works %}{{ card(work, 0) }}{% endfor %}</div>
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

<aside class="right">
  {# A periodical, not a search page: the reader's problem is getting through one
     issue, not narrowing thousands of hits. So the rail leads with navigation. #}
  {% if works %}
  <div class="panel">
    <h3>{{ icon('book') }} 本期目录</h3>
    <nav class="toc">
      {% for work in works %}
      {% set h = direction_hue(direction(work, problem_names), hue_order) %}
      <a href="#p{{ loop.index }}">
        <i class="dot" style="background:{{ h.fg }}"></i>
        <span class="n">{{ '%02d'|format(loop.index) }}</span>
        <span class="tt">{{ work.extra.get('title_zh') or work.title }}</span>
        {% if work.label == 'must_read' %}<em>必读</em>{% endif %}
      </a>
      {% endfor %}
    </nav>
  </div>
  {% endif %}

  {% if venues %}
  <div class="panel">
    <h3>{{ icon('book') }} 本期期刊</h3>
    <div class="rows">
      {% for name, n in venues %}
      <div class="row"><span class="rt">{{ name }}</span><b>{{ n }}</b></div>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  {% if library_directions %}
  <div class="panel">
    <h3>{{ icon('book') }} 你的兴趣画像</h3>
    <div class="hint">推荐是按这个结构排序的。库里某个方向积累得多，本期该方向自然推得多。</div>
    <div class="bars">
      {% for name, n in library_directions %}
      {% set h = direction_hue(name, hue_order) %}
      <div class="brow{{ ' off' if name in quiet_directions else '' }}">
        <div class="t"><span>{{ name }}</span><b>{{ n }}</b></div>
        <div class="track"><i style="width:{{ (n / library_directions[0][1] * 100)|round|int }}%;background:{{ h.fg }}"></i></div>
      </div>
      {% endfor %}
    </div>
    {% if quiet_directions %}
    <div class="hint" style="margin:11px 0 0">
      灰掉的 {{ quiet_directions|length }} 个方向本期一篇都没有。连续几期为空就该检查检索词了。
    </div>
    {% endif %}
  </div>
  {% endif %}

  {% if watched or author_candidates %}
  <div class="panel">
    <h3>{{ icon('user') }} 作者动向</h3>

    {% if watched %}
    <div class="sub-h">本期发文的已关注作者</div>
    <div class="rows">
      {% for name, n in watched %}
      <div class="row"><span class="rt">{{ name }}</span>{% if n > 1 %}<b>{{ n }}</b>{% endif %}</div>
      {% endfor %}
    </div>
    {% endif %}

    {% if author_candidates %}
    <div class="sub-h"{% if watched %} style="margin-top:15px"{% endif %}>
      查漏补缺 · 你还没关注的人</div>
    <div class="hint">按跨期累积的高相关论文数排出，不会自动加入关注。</div>
    <div class="cands">
      {% for c in author_candidates %}
      <div class="cand">
        <div class="cand-h">
          <span class="nm">{{ c.name }}</span>
          {% if c.is_new %}<em class="new">新涌现</em>{% endif %}
          <b>{{ c.count }} 篇</b>
        </div>
        {% if c.affiliation %}<div class="aff">{{ c.affiliation }}</div>{% endif %}
        <div class="cand-p">
          {% for paper in c.papers[:2] %}<a href="{{ paper.url or '#' }}" target="_blank" rel="noopener">{{ paper.title }}</a>{% endfor %}
        </div>
        {% if c.add_url %}<a class="add" href="{{ c.add_url }}" target="_blank" rel="noopener">加入关注 →</a>{% endif %}
      </div>
      {% endfor %}
    </div>
    {% endif %}
  </div>
  {% endif %}

  <div class="panel">
    <h3>{{ icon('clock') }} 往期</h3>
    {% if past %}
    <div class="hint">每期留档，链接不会因为下一期发布而失效。</div>
    <ol class="issues">
      {% for row in past %}
      <li><a href="{{ row.file }}">
        <span class="no">{% if row.no %}第 {{ row.no }} 期{% else %}{{ row.date }}{% endif %}</span>
        <span class="when">{{ row.short }}</span>
        {% if row.count %}<span class="cnt">{{ row.count }} 篇</span>{% endif %}
      </a></li>
      {% endfor %}
    </ol>
    {% else %}
    <div class="hint">这是第一期，还没有往期。以后每期都会留档在这里。</div>
    {% endif %}
    <div class="links">
      <a href="archive.html">全部往期 →</a>
      <a href="feed.xml">RSS 订阅 →</a>
    </div>
  </div>
</aside>

</div>
<script>
document.addEventListener('click', function (e) {
  var more = e.target.closest('.more');
  if (more) {
    var box = more.closest('article').querySelector('.abs:not(.zh)');
    var open = box.classList.toggle('open');
    more.classList.toggle('open', open);
    more.textContent = open ? '收起' : '显示更多';
    return;
  }
  var alt = e.target.closest('.alt');
  if (alt) {
    var zh = alt.closest('article').querySelector('.abs.zh');
    zh.hidden = !zh.hidden;
    alt.textContent = zh.hidden ? '翻译' : '隐藏译文';
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
                problem_names: dict | None = None, library_size: str = "",
                window_days: int = 30, library_directions: list | None = None,
                issue_no: int = 0, feedback_repository: str = "") -> Path:
    env = Environment(autoescape=True)
    template: Template = env.from_string(_TEMPLATE)
    problem_names = problem_names or {}
    every = [*(works or []), *(watched_works or []), *(classic_works or []), *(exploration_works or [])]

    seen: dict = {}
    for work in every:
        name = direction(work, problem_names)
        if name:
            seen[name] = seen.get(name, 0) + 1

    ordered = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
    hue_order = {name: i for i, (name, _) in enumerate(ordered)}

    path = Path(output_path)
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
        directions=ordered,
        problem_names=problem_names,
        library_size=library_size,
        generated_at=now.strftime("%Y-%m-%d %H:%M"),
        date_label=now.strftime("%Y-%m-%d"),
        date_cn=f"{now.year} 年 {now.month} 月 {now.day} 日",
        generated_at_time=now.strftime("%H:%M"),
        window_days=window_days,
        issue_no=issue_no,
        clamp_at=ABSTRACT_CLAMP,
        recommendation_reasons=recommendation_reasons,
        authors_line=authors_line, source_line=source_line, direction=direction,
        anchor=anchor, clamp=clamp, split_abstract=split_abstract, icon=icon,
        published_label=published_label,
        direction_hue=direction_hue,
        hue_order=hue_order,
        venues=venue_counts(every),
        past=past_issues(path),
        watched=watched_hits(every),
        author_candidates=author_candidates(diagnostics, feedback_repository),
        library_directions=library_directions or [],
        quiet_directions=[n for n, _ in (library_directions or []) if n not in seen],
        rating_buttons=rating_buttons, citations=citations,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    logger.info("Wrote HTML report to %s (%.0f KB)", path, len(rendered.encode("utf-8")) / 1024)
    return path


__all__ = ["render_html", "funnel_totals", "clamp", "split_abstract", "authors_line", "source_line",
           "direction", "direction_hue", "anchor", "icon", "rating_buttons", "citations",
           "published_label",
           "scatter", "venue_counts", "library_anchors", "watched_hits", "past_issues",
           "author_candidates"]
