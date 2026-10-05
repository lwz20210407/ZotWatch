"""Freeze same-run source cohorts for later read-only, human-labeled evaluation."""
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .research_dossier import paper_id
from .utils import atomic_json


def export_shadow(cohorts, api_summary, output_dir, *, window_days, baseline_cached_at=None):
    output = Path(output_dir)
    extra_fields = {'aminer_id', 'publication_year', 'date_precision', 'abstract_is_partial',
                    'abstract_source', 'external_ids', 'semantic_facets', 'aminer_facets'}
    snapshots = {}
    for name in ('baseline', 'aminer', 'combined'):
        snapshots[name] = []
        for work in cohorts.get(name, []):
            data = work.model_dump(mode='json')
            data['extra'] = {k: v for k, v in work.extra.items() if k in extra_fields}
            data['work_id'] = paper_id(work)
            snapshots[name].append(data)
    old_ids = {row['work_id'] for row in snapshots['baseline']}
    unique = {row['work_id']: row for row in snapshots['aminer'] if row['work_id'] not in old_ids}
    payload = {'schema_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(),
        'stage': 'topic_filtered_before_library_history_and_ranking', 'window_days': window_days,
        'baseline_cached_at': baseline_cached_at, 'cohorts': snapshots,
        'aminer_distinct_candidates': len(unique), 'api': api_summary,
        'note': 'Captured in one watch run. Provider caches/index times can differ; neither source order is a relevance ranking. No precision/recall claim without independent human labels.'}
    atomic_json(output / 'aminer-shadow.json', payload)
    with (output / 'aminer-review.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['work_id', 'title', 'doi', 'aminer_id', 'relevant', 'reason'])
        def safe(value):
            value = str(value or '')
            return "'" + value if value.lstrip().startswith(('=', '+', '-', '@')) or value.startswith(('\t', '\r')) else value
        for row in unique.values():
            writer.writerow([safe(v) for v in (row['work_id'], row['title'], row.get('doi'),
                                               row['extra'].get('aminer_id'), '', '')])
    return {'aminer_distinct_candidates': len(unique), 'human_labels': 0}


# --- Turning shadow mode into a decision -------------------------------------------
#
# Shadow mode exists to answer one question: does AMiner find relevant papers the
# existing sources miss? Until 2026-10-05 it could never answer it. The evidence above
# is overwritten every run and never committed, and the labelling CSV it writes was
# never filled in -- export_shadow hard-codes human_labels: 0. So the comparison was
# recomputed weekly and thrown away, and the switch to live had nothing to stand on.
#
# The three functions below close that loop without asking for any new habit. The
# AMiner-only papers are shown in the web report with the same useful / not-relevant
# buttons the owner already uses, those clicks arrive as ordinary feedback issues, and
# a per-week ledger lets the scoreboard count them across weeks.

USEFUL_RATINGS = frozenset({'direct', 'transferable', 'mechanism'})


def trial_candidates(cohorts, *, keep_unseen, limit=12):
    """AMiner-only papers worth putting in front of the owner this week.

    `keep_unseen` removes what the owner has already seen -- papers already in the
    library and papers already delivered -- since asking him to label those measures
    nothing. The shadow cohort is captured before both of those filters, so they must
    be applied here.
    """
    baseline_ids = {paper_id(work) for work in cohorts.get('baseline', [])}
    seen, novel = set(), []
    for work in cohorts.get('aminer', []):
        ident = paper_id(work)
        if ident in baseline_ids or ident in seen:
            continue
        seen.add(ident)
        novel.append(work)
    return list(keep_unseen(novel))[:max(0, int(limit))]


def persist_week(works, ledger_dir, week):
    """Record this week's AMiner-only papers so the scoreboard can count across weeks."""
    rows = [{'work_id': paper_id(work), 'title': work.title, 'doi': work.doi, 'url': work.url,
             'aminer_id': work.extra.get('aminer_id'), 'venue': work.venue,
             'year': work.extra.get('publication_year')} for work in works]
    path = Path(ledger_dir) / f'{week}.json'
    atomic_json(path, {'schema_version': 1, 'week': week, 'items': rows})
    return path


def scoreboard(ledger_dir, feedback_entries, *, min_labels=20, graduate_ratio=0.30):
    """Cumulative evidence for switching AMiner from shadow to live.

    min_labels and graduate_ratio are configurable engineering defaults, not
    statistical thresholds: they encode "do not decide on a handful of clicks" and "a
    third of the extra papers being worth reading justifies the extra noise". The
    verdict is advice for the owner, never an automatic switch.
    """
    weeks, items = [], {}
    ledger = Path(ledger_dir)
    for path in (sorted(ledger.glob('*.json')) if ledger.is_dir() else []):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        weeks.append(data.get('week') or path.stem)
        for row in data.get('items', []):
            items.setdefault(row.get('work_id'), row)

    # Match on every identity a paper can carry. paper_id() only emits doi:<x> for a
    # DOI that passes clean_doi's strict pattern; anything else becomes a snapshot hash,
    # which no feedback click can ever match. Keying on the raw DOI as well, normalised
    # the same way feedback normalises it, keeps those papers countable.
    def doi_key(value):
        doi = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', str(value or '').strip(), flags=re.I)
        return 'doi:' + doi.casefold().rstrip(' .;') if doi else None

    labels = {}
    if isinstance(feedback_entries, dict):  # FeedbackModel.entries is keyed by DOI/scope
        feedback_entries = feedback_entries.values()
    for entry in feedback_entries or []:
        rating = getattr(entry, 'rating', None)
        if getattr(entry, 'work_id', ''):
            labels[str(entry.work_id).casefold()] = rating
        if doi_key(getattr(entry, 'doi', '')):
            labels[doi_key(entry.doi)] = rating

    useful = irrelevant = 0
    for ident, row in items.items():
        keys = {str(ident).casefold()}
        if row.get('aminer_id'):
            keys.add('aminer:' + str(row['aminer_id']).casefold())
        if doi_key(row.get('doi')):
            keys.add(doi_key(row['doi']))
        rating = next((labels[k] for k in keys if k in labels), None)
        if rating in USEFUL_RATINGS:
            useful += 1
        elif rating == 'irrelevant':
            irrelevant += 1

    labelled = useful + irrelevant
    ratio = useful / labelled if labelled else None
    if labelled < min_labels:
        verdict = f'证据不足：已标注 {labelled} 篇，至少 {min_labels} 篇再判断'
    elif ratio >= graduate_ratio:
        verdict = f'建议转为 live：有用比例 {ratio:.0%}，达到 {graduate_ratio:.0%}'
    else:
        verdict = f'建议维持影子模式：有用比例 {ratio:.0%}，低于 {graduate_ratio:.0%}'
    return {'weeks': len(weeks), 'first_week': weeks[0] if weeks else None,
            'aminer_only_total': len(items), 'labelled': labelled, 'useful': useful,
            'irrelevant': irrelevant, 'useful_ratio': ratio, 'min_labels': min_labels,
            'graduate_ratio': graduate_ratio, 'verdict': verdict}
