"""Conservative numeric citations from explicit local documents, without network calls."""
import re
import html
import hashlib
import json
from pathlib import Path

from .aminer_source import clean_doi
from .topic_matching import normalize_text

MARKER = re.compile(r"\[\s*\d{1,4}(?:\s*(?:,|;|[-–])\s*\d{1,4})*\s*\]")
HEADING = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\d+\.?\s+)?(?:references|bibliography|参考文献)\s*:?\s*$", re.I)
ENTRY = re.compile(r"^\s*(\[\d{1,4}\])\s+(.+)$")


def marker_numbers(marker):
    if not isinstance(marker, str) or not MARKER.fullmatch(marker):
        return []
    numbers = []
    for part in re.split(r"[,;]", marker[1:-1]):
        bounds = re.split(r"[-–]", part)
        if len(bounds) > 2:
            return []
        start, end = int(bounds[0]), int(bounds[-1])
        if not 1 <= start <= end <= 9999 or end - start >= 20:
            return []
        numbers.extend(range(start, end + 1))
    return sorted(set(numbers)) if len(numbers) <= 20 else []


def reference_dois(reference):
    values = set()
    for raw in re.findall(r'10\.\d{4,9}/[^\s<>"\]]+', reference, re.I):
        raw = raw.rstrip('.,;:')
        while raw.endswith(')') and raw.count(')') > raw.count('('):
            raw = raw[:-1]
        value = clean_doi(raw)
        if value:
            values.add(value)
    return values


def reference_matches(reference, work):
    dois = reference_dois(reference)
    if dois:
        # Conflicting DOI evidence cannot be overridden by a similar title.
        return len(dois) == 1 and clean_doi(work.doi) in dois
    title = normalize_text(work.title)
    return len(title) >= 12 and title in normalize_text(reference)


def propose_citations(documents, papers, limit=100):
    """Return candidate annotations; callers must still verify original anchors.

    References must have an explicit heading and bracket-numbered entries.
    Entries spanning different evidence locators remain unresolved.
    """
    proposals, unresolved = [], []
    seen = set()
    truncated = False
    for source, document in documents.items():
        lines = [(b['locator'], line) for b in document.get('blocks', []) for line in b['text'].splitlines()]
        heading = next((i for i, (_, line) in enumerate(lines) if HEADING.fullmatch(line)), None)
        if heading is None:
            unresolved.append({'from': source, 'reason': 'reference_heading_not_found'})
            continue
        refs = {}
        current = None
        for locator, line in lines[heading + 1:]:
            if re.match(r'^\s*(?:#{1,6}\s*)?(?:appendix|appendices|supplement|附录)\b', line, re.I):
                break
            if HEADING.fullmatch(line) or not line.strip():
                continue
            match = ENTRY.match(line)
            if match:
                number = int(match[1][1:-1])
                current = {'marker': match[1], 'locator': locator, 'quote': line.strip(), 'cross_locator': False}
                refs.setdefault(number, []).append(current)
            elif current is not None:
                current['cross_locator'] |= locator != current['locator']
                if len(current['quote']) <= 2000:
                    current['quote'] += '\n' + line
        for line_index, (locator, line) in enumerate(lines[:heading]):
            for match in MARKER.finditer(line):
                marker = match[0]
                numbers = marker_numbers(marker)
                if not numbers:
                    unresolved.append({'from': source, 'locator': locator, 'marker': marker, 'reason': 'unsupported_citation_range'})
                for number in numbers:
                    key = (source, line_index, locator, match.start(), marker, number)
                    if key in seen:
                        continue
                    seen.add(key)
                    if len(seen) > limit:
                        truncated = True
                        break
                    candidates = refs.get(number, [])
                    reason = ''
                    if len(candidates) != 1:
                        reason = 'missing_or_duplicate_reference_number'
                    elif candidates[0]['cross_locator'] or len(candidates[0]['quote']) > 2000:
                        reason = 'reference_requires_manual_anchor'
                    else:
                        ref = candidates[0]
                        targets = [key for key, work in papers.items() if key != source and reference_matches(ref['quote'], work)]
                        if len(targets) != 1:
                            reason = 'target_not_selected_or_ambiguous'
                    if reason:
                        unresolved.append({'from': source, 'locator': locator, 'marker': marker,
                                           'reference_number': number, 'reason': reason})
                        continue
                    quote = line[max(0, match.start() - 250):match.end() + 250].strip()
                    proposals.append({'from': source, 'to': targets[0], 'relation': 'cites',
                        'marker': marker, 'reference_marker': ref['marker'], 'locator': locator, 'quote': quote,
                        'reference_locator': ref['locator'], 'reference_quote': ref['quote'], 'origin': 'local_numeric_detection'})
                if truncated:
                    break
            if truncated:
                break
        if truncated:
            break
    if truncated:
        unresolved.append({'reason': 'automatic_citation_limit_reached', 'limit': limit})
    return proposals, unresolved[:limit + len(documents) + 1]


def paper_anchor(work_id):
    return 'paper-' + hashlib.sha256(work_id.encode()).hexdigest()[:16]


def export_traces(ledger, directory, *, imported=False):
    """A local evidence browser, not a inferred claim/support or influence graph."""
    directory = Path(directory)
    esc = lambda value: html.escape(str(value), quote=True)
    nodes = {p['work_id']: p['title'] for p in ledger['papers']}
    rows = []
    for edge in ledger['citations']:
        source, target = edge['from'], edge['to']
        rows.append(f'<article><h2><a href="dossier.html#{paper_anchor(source)}">{esc(nodes[source])}</a> → '
                    f'<a href="dossier.html#{paper_anchor(target)}">{esc(nodes[target])}</a></h2>'
                    f'<p>引用标记 {esc(edge["marker"])} · 原文位置 {esc(edge["locator"])}</p>'
                    f'<blockquote>{esc(edge["quote"])}</blockquote><details><summary>核对参考文献条目</summary>'
                    f'<p>{esc(edge["reference_locator"])}</p><blockquote>{esc(edge["reference_quote"])}</blockquote></details></article>')
    gaps = ''.join(f'<li>{esc(nodes.get(row.get("from"), "专题"))} · {esc(row.get("locator", ""))} · {esc(row.get("reason", "待核查"))}</li>'
                   for row in ledger['unresolved_citations'])
    notice = '展示导入证据包记录的引用，本次归档未重新核验原文。' if imported else '边仅表示已匹配本地引用语境与参考文献，不证明支持、改进、反驳或方法采用。'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>引文脉络与原文依据</title>'
    page += '<style>body{font:16px/1.7 sans-serif;max-width:1050px;margin:36px auto;padding:0 20px;color:#243247}article{padding:20px;margin:20px 0;border:1px solid #d9e1e8;border-radius:12px}h2{font-size:18px}blockquote{margin:12px 0;padding:12px;background:#f4f7f9;overflow-wrap:anywhere}input{padding:12px;width:90%;max-width:600px}a{color:#17617d}</style>'
    page += f'<a href="dossier.html">返回证据工作台</a><h1>{esc(ledger["topic"])}：引文脉络</h1><p>{esc(notice)}</p><p>{len(nodes)} 篇文献 · {len(rows)} 条引用记录 · {len(ledger["unresolved_citations"])} 项待核查</p>'
    page += '<label>按题名或原文筛选 <input id="filter" type="search" placeholder="输入论文题名、方法或关键词"></label>'
    page += '<main>' + (''.join(rows) or '<p>未发现可靠的本地引文关系。请提供全文及明确的参考文献对应关系。</p>') + '</main>'
    page += f'<details><summary>未匹配或需要人工核查的记录</summary><ul>{gaps or "<li>无待核查记录；这不代表引用已穷尽。</li>"}</ul></details>'
    page += '<script>document.getElementById("filter").addEventListener("input",function(){const q=this.value.toLowerCase();document.querySelectorAll("article").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></html>'
    (directory / 'citation-traces.html').write_text(page, 'utf-8')
    (directory / 'citation-traces.json').write_text(json.dumps({'schema_version': 1,
        'nodes': [{'id': k, 'title': v} for k, v in nodes.items()], 'edges': ledger['citations'],
        'unresolved': ledger['unresolved_citations'], 'notice': notice}, ensure_ascii=False, indent=2), 'utf-8')
