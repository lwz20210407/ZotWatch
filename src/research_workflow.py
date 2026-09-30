"""One local research run: bounded discovery, profile filtering, evidence and archive."""
import argparse
import csv
import hashlib
import html
import json
import subprocess
import sys
import uuid
from pathlib import Path

from .aminer_acceptance import eligible_candidates, profile_fingerprint
from .aminer_source import AMinerSource, clean_doi
from .aminer_policy import screen_aminer_delivery
from .models import RankedWork
from .research_archive import import_dossier, publication_lock
from .research_dossier import build_dossier, export_dossier, read_snapshot, read_document, paper_id
from .research_features import FeedbackModel, load_feedback, facet_ids
from .settings import load_settings
from .utils import atomic_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_inputs(args):
    files = profile_fingerprint(args.base_dir)
    # Include all config files, including network/source budgets and translation changes.
    files.update({p.relative_to(args.base_dir).as_posix(): sha(p) for p in sorted((args.base_dir / 'config').glob('*.yaml'))})
    if args.snapshot:
        files['snapshot'] = sha(args.snapshot)
    if args.evidence_map:
        files['evidence_map'] = sha(args.evidence_map)
        mapping = json.loads(args.evidence_map.read_text('utf-8'))
        if not isinstance(mapping, dict) or not isinstance(mapping.get('documents', []), list) or len(mapping.get('documents', [])) > 20:
            raise ValueError('Evidence map must contain at most 20 local documents')
        root = args.evidence_root or args.evidence_map.parent
        for i, spec in enumerate(mapping.get('documents', [])):
            files['document:' + str(i)] = read_document(spec['path'], root)['sha256']
    return files


def run_ranking(args, snapshot, output):
    command = [sys.executable, '-B', '-m', 'src.aminer_acceptance', '--base-dir', str(args.base_dir),
        '--snapshot', str(snapshot), '--cohort', 'combined', '--output-dir', str(output),
        '--max-new-texts', str(args.max_new_texts), '--deadline', str(args.deadline)]
    if args.cache_only:
        command.append('--cache-only')
    output.mkdir(parents=True, exist_ok=True)
    # The worker reports sanitized status; raw subprocess output is not persisted.
    try:
        subprocess.run(command, cwd=Path(__file__).resolve().parents[1], timeout=args.deadline + 15,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'ranked': []}
    status = json.loads((output / 'status.json').read_text('utf-8'))
    if status.get('status') not in {'complete', 'partial'}:
        return {'status': status.get('status', 'failed'), 'ranked': []}
    result = json.loads((output / 'acceptance.json').read_text('utf-8'))
    if not result.get('inputs_unchanged'):
        raise ValueError('Ranking inputs changed')
    return result


def export_comparison(ledger, path):
    def safe(value):
        text = str(value or '')
        return "'" + text if text.lstrip().startswith(('=', '+', '-', '@')) or text.startswith(('\t', '\r')) else text
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['work_id', 'title', 'dimension', 'status', 'mentions', 'evidence_ids', 'evidence_level'])
        for paper in ledger['papers']:
            for dimension, field in paper['fields'].items():
                writer.writerow([safe(v) for v in [paper['work_id'], paper['title'], dimension, field['status'],
                    '; '.join(field['mentions']), '; '.join(field['evidence_ids']), paper['evidence_level']]])


def write_index(output, state):
    esc = lambda value: html.escape(str(value), quote=True)
    links = '<a href="selection.json">候选与筛选记录</a>' if (output / 'selection.json').exists() else ''
    if state.get('dossier'):
        links += ' · <a href="dossier/dossier.html">证据工作台</a> · <a href="dossier/citation-traces.html">引文脉络</a> · <a href="comparison.csv">比较表 CSV</a>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>专题研究运行</title><style>body{font:16px/1.8 sans-serif;max-width:900px;margin:40px auto;padding:20px;color:#25394b}li{margin:12px 0}a{color:#17617d}pre{white-space:pre-wrap}</style>'
    status = {'complete': '已完成', 'partial': '尚未完成，可恢复', 'failed': '本次运行失败', 'empty': '暂无符合条件的候选', 'running': '进行中'}.get(state['status'], state['status'])
    stage = {'discovery': '文献发现', 'selection': '候选筛选', 'ranking': '画像评分', 'dossier': '证据整理', 'archive': '私有归档', 'finished': '已完成'}.get(state.get('stage'), '')
    page += f'<h1>{esc(state["spec"]["topic"])}</h1><p>状态：{esc(status)} · 当前步骤：{esc(stage)}</p><p>本地候选与证据整理，尚待人工核查；不发送邮件，不发布网页，不更改推送历史。</p><p>{links}</p>'
    page += f'<p>候选 {state.get("candidate_count", 0)} 篇 · 通过主题/文库/历史筛选 {state.get("eligible_count", 0)} 篇 · 进入专题 {state.get("selected_count", 0)} 篇</p>'
    page += '<h2>本次说明</h2><ul>' + ''.join('<li>' + esc(w) + '</li>' for w in state.get('warnings', [])) + '</ul></html>'
    (output / 'index.html').write_text(page, 'utf-8')


def execute(args):
    args.base_dir = args.base_dir.resolve()
    output = args.output_dir.resolve()
    settings = load_settings(args.base_dir)
    known = {f.id for f in settings.research.facets}
    if set(args.facet) - known or (args.discover and not args.facet):
        raise ValueError('Discovery requires existing facet IDs')
    inputs = capture_inputs(args)
    spec = {'version': 1, 'topic': args.topic, 'discover': args.discover, 'facets': sorted(args.facet),
        'cohort': args.cohort, 'rank': args.rank, 'max_papers': args.max_papers,
        'max_requests': args.max_requests, 'auto_citations': args.auto_citations, 'archive': args.archive, 'inputs': inputs}
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Output is not empty; use a new directory or --resume')
    if args.resume and not (output / 'run.json').is_file():
        raise ValueError('Resume requires an existing run manifest')
    output.mkdir(parents=True, exist_ok=True)
    # This lock belongs to this run only; an existing lock is never removed automatically.
    with publication_lock(output):
        state = json.loads((output / 'run.json').read_text('utf-8')) if args.resume else {
            'spec': spec, 'run_id': uuid.uuid4().hex, 'status': 'running', 'stage': 'discovery', 'warnings': [], 'artifacts': {}}
        if state.get('spec') != spec:
            raise ValueError('Run inputs/configuration changed; create a new output directory')
        for name, digest in state.get('artifacts', {}).items():
            path = (output / name).resolve()
            if not path.is_relative_to(output) or not path.is_file() or sha(path) != digest:
                raise ValueError('A completed run artifact was changed or removed')
        if state['status'] == 'complete':
            return state
        def checkpoint():
            atomic_json(output / 'run.json', state)
            write_index(output, state)
        def finish():
            if capture_inputs(args) != inputs:
                raise ValueError('Inputs changed during the run')
            if args.archive:
                state['stage'] = 'archive'; checkpoint()
                state['archive'] = import_dossier(output / 'dossier', output / 'private-archive')
            state.update(status='complete', stage='finished', inputs_unchanged=True)
            checkpoint()
            return state
        try:
            state['status'] = 'running'; checkpoint()
            if state.get('dossier'):
                return finish()
            snapshot = output / 'candidates.json'
            if not state.get('discovery_done'):
                works = read_snapshot(args.snapshot, args.cohort) if args.snapshot else []
                if args.discover:
                    cfg = settings.model_copy(deep=True)
                    cfg.sources.aminer.enabled = True
                    cfg.sources.aminer.max_requests = args.max_requests
                    cfg.sources.aminer.max_recommendation_queries = 2
                    cfg.sources.aminer.max_run_seconds = min(cfg.sources.aminer.max_run_seconds, 120)
                    cfg.research.facets = [f for f in cfg.research.facets if f.id in args.facet]
                    for facet in cfg.research.facets:
                        facet.semantic_query = args.topic + '. ' + (facet.semantic_query or facet.name)
                    from .watch_history import WatchHistory
                    history = WatchHistory(args.base_dir / 'data/watch-state')
                    feedback = FeedbackModel(load_feedback(args.base_dir, settings.research, history.state.get('feedback_entries', [])), settings.research)
                    source = AMinerSource(cfg, output / 'api-cache', feedback=feedback)
                    from .citation_watch import merge_candidates
                    discovered = source.fetch()
                    works = merge_candidates([*works, *discovered])
                    state['discovery'] = source.stats
                    state['warnings'] = list(dict.fromkeys([*state['warnings'], *source.stats.get('warnings', [])]))
                    if not discovered and source.stats.get('warnings'):
                        state.update(status='partial', stage='discovery')
                        checkpoint(); return state
                atomic_json(snapshot, [w.model_dump(mode='json') for w in works])
                state.update(discovery_done=True, candidate_count=len(works))
                state['artifacts']['candidates.json'] = sha(snapshot); checkpoint()
            works = read_snapshot(snapshot)
            state['stage'] = 'selection'; checkpoint()
            eligible, history, topic_count = eligible_candidates(args.base_dir, works, settings)
            if args.facet:
                eligible = [w for w in eligible if set(facet_ids(w, settings.research)) & set(args.facet)]
            state.update(eligible_count=len(eligible), topic_count=topic_count)
            if args.rank:
                state['stage'] = 'ranking'; checkpoint()
                result = run_ranking(args, snapshot, output / 'ranking')
                state['ranking_status'] = result['status']
                ids = {paper_id(w) for w in eligible}
                chosen = [RankedWork.model_validate(row) for row in result.get('ranked', [])
                          if row.get('label') != 'ignore' and not row.get('extra', {}).get('feedback_read')]
                chosen = [w for w in chosen if paper_id(w) in ids]
                chosen, review = screen_aminer_delivery(chosen, settings.research)
                if result['status'] != 'complete':
                    atomic_json(output / 'selection.json', {'ranking_status': result['status'], 'review': review,
                        'partial_candidates': [w.model_dump(mode='json') for w in chosen]})
                    state.update(status='partial', selected_count=0)
                    checkpoint(); return state  # No partial evidence pack masquerades as a completed topic.
            else:
                feedback = FeedbackModel(load_feedback(args.base_dir, settings.research, history.state.get('feedback_entries', [])), settings.research)
                chosen = []
                for work in eligible:
                    entry = feedback.entry_for_work(work)
                    if entry and entry.rating == 'read':
                        continue
                    if entry and entry.rating != 'reset':
                        work = work.model_copy(update={'extra': {**work.extra, 'feedback_applicability': entry.applicability}})
                    chosen.append(work)
                chosen, review = screen_aminer_delivery(chosen, settings.research)
                if '未启用画像评分；按输入顺序选取，不能解释为推荐排名。' not in state['warnings']:
                    state['warnings'].append('未启用画像评分；按输入顺序选取，不能解释为推荐排名。')
            chosen = chosen[:args.max_papers]
            atomic_json(output / 'selection.json', {'ranked': args.rank, 'review': review,
                'selected': [w.model_dump(mode='json') for w in chosen], 'eligible': len(eligible)})
            state['selected_count'] = len(chosen)
            if not chosen:
                state.update(status='empty', stage='selection'); checkpoint(); return state
            if capture_inputs(args) != inputs:
                raise ValueError('Inputs changed during the run')
            state['stage'] = 'dossier'; checkpoint()
            mapping = json.loads(args.evidence_map.read_text('utf-8')) if args.evidence_map else {}
            aliases = {paper_id(w) for w in chosen} | {'aminer:' + str(w.extra.get('aminer_id', '')).casefold() for w in chosen}
            def selected(spec):
                value = spec.get('work_id', '')
                ident = 'doi:' + clean_doi(value) if clean_doi(value) else str(value).casefold()
                return ident in aliases
            original = mapping.get('documents', [])
            mapping['documents'] = [row for row in original if selected(row)]
            state['omitted_documents'] = len(original) - len(mapping['documents'])
            if state['omitted_documents']:
                state['warnings'].append(f"{state['omitted_documents']} 份本地文档对应的论文未入选，本次未用于专题证据。")
            if not mapping['documents']:
                state['warnings'].append('未提供入选论文的本地全文；只有题名/摘要线索，无法据此建立可靠引文脉络。')
            ledger = build_dossier(args.topic, chosen, settings.research, evidence_map=mapping,
                evidence_root=args.evidence_root or (args.evidence_map.parent if args.evidence_map else None), auto_citations=args.auto_citations)
            # Each export attempt gets its own directory, preserving interrupted work.
            attempt = output / ('attempt-' + uuid.uuid4().hex[:12])
            export_dossier(ledger, attempt)
            if (output / 'dossier').exists():
                raise ValueError('Unexpected existing dossier; preserve it and use a new run')
            attempt.rename(output / 'dossier')
            export_comparison(ledger, output / 'comparison.csv')
            state['dossier'] = True
            if capture_inputs(args) != inputs:
                raise ValueError('Inputs changed during the run')
            for path in [output / 'selection.json', output / 'comparison.csv', *(output / 'dossier').iterdir()]:
                state['artifacts'][path.relative_to(output).as_posix()] = sha(path)
            checkpoint()
            return finish()
        except Exception as exc:
            state.update(status='failed', error_type=type(exc).__name__)
            checkpoint()
            raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--topic', required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--cohort', default='combined')
    parser.add_argument('--discover', action='store_true')
    parser.add_argument('--facet', action='append', default=[])
    parser.add_argument('--rank', action='store_true')
    parser.add_argument('--cache-only', action='store_true')
    parser.add_argument('--max-new-texts', type=int, default=20)
    parser.add_argument('--deadline', type=int, default=180)
    parser.add_argument('--max-requests', type=int, default=12)
    parser.add_argument('--max-papers', type=int, default=10)
    parser.add_argument('--evidence-map', type=Path)
    parser.add_argument('--evidence-root', type=Path)
    parser.add_argument('--auto-citations', action='store_true')
    parser.add_argument('--archive', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.topic.strip() or len(args.topic) > 500 or not (args.snapshot or args.discover):
        parser.error('Provide a topic up to 500 characters and a snapshot or explicit discovery')
    if not 1 <= args.max_requests <= 24 or not 1 <= args.max_papers <= 20 or not 0 <= args.max_new_texts <= 200 or not 10 <= args.deadline <= 600:
        parser.error('Budgets out of range')
    if args.cache_only and not args.rank:
        parser.error('--cache-only requires --rank')
    return args


def main(argv=None):
    try:
        state = execute(parse_args(argv))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__, 'note': 'Check inputs and run manifest; use a new directory if inputs changed.'}))
        raise SystemExit(2) from None
    print(json.dumps({k: state.get(k) for k in ('status', 'stage', 'candidate_count', 'eligible_count', 'selected_count', 'ranking_status')}, ensure_ascii=False))
    raise SystemExit(0 if state['status'] in {'complete', 'empty'} else 2)


if __name__ == '__main__': main()
