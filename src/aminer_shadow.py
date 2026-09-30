"""Freeze same-run source cohorts for later read-only, human-labeled evaluation."""
import csv
import json
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
