"""Explicitly confirmed entity identities, bounded discovery and observable matches."""
import argparse
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from .utils import atomic_json

logger = logging.getLogger(__name__)
CONFIG = 'config/entity-tracking.json'
KINDS = {'person': 'A', 'organization': 'I', 'venue': 'S'}


def valid_issn(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{3}[\dX]', value):
        return False
    digits = value.replace('-', '')
    return (sum(int(digits[i]) * (8 - i) for i in range(7)) + (10 if digits[7] == 'X' else int(digits[7]))) % 11 == 0


def validate_registry(data):
    if not isinstance(data, dict) or data.get('schema_version') != 1 or not isinstance(data.get('entries'), list) or len(data['entries']) > 100:
        raise ValueError('Invalid entity registry')
    keys, bridges = set(), set()
    for row in data['entries']:
        if not isinstance(row, dict) or row.get('kind') not in KINDS:
            raise ValueError('Unknown entity kind')
        if not isinstance(row.get('aminer_id'), str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', row['aminer_id']):
            raise ValueError('Invalid AMiner identity')
        key = row['kind'] + ':' + row['aminer_id']
        if row.get('key') != key or key in keys:
            raise ValueError('Duplicate or inconsistent entity key')
        keys.add(key)
        if not isinstance(row.get('name'), str) or not row['name'].strip() or len(row['name']) > 300 or type(row.get('enabled')) is not bool:
            raise ValueError('Invalid entity name/status')
        if not isinstance(row.get('reason'), str) or not row['reason'].strip() or len(row['reason']) > 1000:
            raise ValueError('A confirmation reason is required')
        ids = row.get('openalex_ids', [])
        row.setdefault('openalex_ids', [])
        row.setdefault('issns', [])
        if not isinstance(row['issns'], list) or (row['issns'] and row['kind'] != 'venue') or any(not valid_issn(v) for v in row['issns']):
            raise ValueError('Invalid journal ISSN mapping')
        for value in row['issns']:
            if value in bridges:
                raise ValueError('Conflicting ISSN mapping')
            bridges.add(value)
        if not isinstance(ids, list) or len(ids) > 5:
            raise ValueError('Invalid provider mappings')
        for ident in ids:
            if not isinstance(ident, str) or not re.fullmatch(KINDS[row['kind']] + r'\d+', ident) or ident in bridges:
                raise ValueError('Invalid/conflicting OpenAlex identity')
            bridges.add(ident)
        if not isinstance(row.get('review_sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', row['review_sha256']):
            raise ValueError('Missing confirmation evidence')
    return data


def load_registry(base):
    path = Path(base) / CONFIG
    if not path.exists():
        return {'schema_version': 1, 'entries': []}
    if path.stat().st_size > 512_000:
        raise ValueError('Entity registry is too large')
    return validate_registry(json.loads(path.read_text('utf-8')))


def confirm(base, review_path, ident, *, reason, openalex_ids=(), issns=()):
    path = Path(review_path)
    if path.stat().st_size > 2_000_000:
        raise ValueError('Review is too large')
    raw = path.read_bytes(); review = json.loads(raw)
    if not isinstance(review, dict) or review.get('kind') not in KINDS or review.get('status') != 'needs_review':
        raise ValueError('A successful entity review is required')
    candidates = review.get('candidates')
    if not isinstance(candidates, list):
        raise ValueError('Invalid review candidates')
    matches = [r for r in candidates if isinstance(r, dict) and str(r.get('id') or r.get('org_id') or '') == ident]
    if len(matches) != 1:
        raise ValueError('Select exactly one unambiguous returned ID')
    candidate = matches[0]
    name = next((candidate.get(k) for k in ('name', 'org_name', 'name_en', 'name_zh') if isinstance(candidate.get(k), str) and candidate[k].strip()), '')
    key = review['kind'] + ':' + ident
    row = {'key': key, 'kind': review['kind'], 'aminer_id': ident, 'name': name, 'enabled': True,
        'openalex_ids': list(openalex_ids), 'issns': list(issns), 'reason': reason, 'review_sha256': hashlib.sha256(raw).hexdigest(),
        'confirmed_at': datetime.now(timezone.utc).isoformat()}
    from .research_archive import publication_lock
    with publication_lock(Path(base) / 'config'):
        registry = load_registry(base)
        if any(r['key'] == key for r in registry['entries']):
            raise ValueError('Entity already registered; use enable/disable rather than overwrite')
        registry['entries'].append(row)
        atomic_json(Path(base) / CONFIG, validate_registry(registry))
    return row


def set_enabled(base, key, enabled):
    from .research_archive import publication_lock
    with publication_lock(Path(base) / 'config'):
        registry = load_registry(base)
        matches = [r for r in registry['entries'] if r['key'] == key]
        if len(matches) != 1:
            raise ValueError('Unknown confirmed entity')
        matches[0]['enabled'] = enabled
        atomic_json(Path(base) / CONFIG, registry)


def mark_entities(work, entries):
    # IDs, never name similarity. Recommendation provenance is not authorship evidence.
    matches = []
    extra = {k: v for k, v in work.extra.items() if k != 'watched_entities'}
    strings = lambda values: {v for v in values if isinstance(v, str)} if isinstance(values, list) else set()
    scalar = lambda value: {value} if isinstance(value, str) and value else set()
    authorships = extra.get('openalex_authorships') or []
    authorships = [r for r in authorships if isinstance(r, dict)] if isinstance(authorships, list) else []
    identities = {
        'person': strings(extra.get('aminer_author_ids', [])),
        'organization': strings(extra.get('aminer_org_ids', [])),
        'venue': scalar(extra.get('aminer_venue_id')),
    }
    oa = {'person': set().union(*(scalar(r.get('author_id')) for r in authorships)),
          'organization': set().union(*(strings(r.get('institution_ids', [])) for r in authorships)),
          'venue': scalar(extra.get('openalex_source_id'))}
    for row in entries:
        if not row['enabled']:
            continue
        evidence = 'aminer_id' if row['aminer_id'] in identities[row['kind']] else 'openalex_id' if set(row['openalex_ids']) & oa[row['kind']] else ''
        if not evidence and row['kind'] == 'venue' and set(row.get('issns', [])) & strings(extra.get('issns', [])):
            evidence = 'issn'
        if evidence:
            matches.append({'key': row['key'], 'kind': row['kind'], 'name': row['name'], 'evidence': evidence})
    extra['watched_entities'] = matches
    return work.model_copy(update={'extra': extra})


def fetch_entities(session, settings, entries, since, cache_dir):
    """At most two confirmed OpenAlex mappings, one page each, in the existing budget."""
    from .author_watch import candidate_from_openalex
    from .source_paging import iter_works
    eligible = [r for r in entries if r['enabled'] and r['openalex_ids']]
    if not eligible or not settings.sources.openalex.enabled:
        return []
    cursor_path = Path(cache_dir) / 'rotation.json'
    fingerprint = hashlib.sha256(json.dumps(eligible, sort_keys=True).encode()).hexdigest()
    start = 0
    try:
        saved = json.loads(cursor_path.read_text('utf-8'))
        if saved.get('fingerprint') == fingerprint:
            start = int(saved['next']) % len(eligible)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    results = []
    fields = {'person': 'authorships.author.id', 'organization': 'authorships.institutions.id', 'venue': 'primary_location.source.id'}
    attempted = 0
    for offset in range(min(2, len(eligible))):
        row = eligible[(start + offset) % len(eligible)]; attempted += 1
        try:
            params = {'filter': f"{fields[row['kind']]}:{'|'.join(row['openalex_ids'])},from_publication_date:{since.date().isoformat()}",
                      'sort': 'publication_date:desc', 'per-page': 20, 'mailto': settings.sources.openalex.mailto}
            for item in iter_works(session, 'https://api.openalex.org/works', params, provider='openalex', max_pages=1,
                    interval_seconds=settings.sources.request_interval_seconds, logger=logger, context='Confirmed entity watch'):
                work = candidate_from_openalex(item)
                if work:
                    tagged = mark_entities(work, [row])
                    if tagged.extra['watched_entities']:
                        results.append(tagged)
        except Exception as exc:
            logger.warning('Entity discovery incomplete: %s', type(exc).__name__)
    try:
        atomic_json(cursor_path, {'fingerprint': fingerprint, 'next': (start + attempted) % len(eligible)})
    except OSError:
        logger.warning('Entity tracking cursor could not be saved')
    return results


def tracking_report(entries, candidates, selected, feedback=None):
    from .author_watch import work_key
    result = []
    for row in entries:
        key = row['key']
        counts = []
        for works in (candidates, selected):
            seen = {work_key(w) for w in works if any(r['key'] == key for r in w.extra.get('watched_entities', []))}
            counts.append(len(seen))
        result.append({'key': key, 'name': row['name'], 'kind': row['kind'], 'enabled': row['enabled'],
            'route': 'openalex_confirmed_mapping' if row['openalex_ids'] else 'aminer_author_recommendations_and_id_matching' if row['kind'] == 'person' else 'incoming_metadata_id_matching',
            'candidate_matches': counts[0], 'selected_for_report': counts[1],
            'note': 'Selection is not delivery. Zero matches do not prove no new publications.'})
        votes = {}
        if feedback:
            for work in candidates:
                if any(r['key'] == key for r in work.extra.get('watched_entities', [])):
                    entry = feedback.entry_for_work(work)
                    if entry and entry.rating != 'reset':
                        votes[work_key(work)] = entry.rating
        result[-1]['feedback'] = {rating: list(votes.values()).count(rating) for rating in sorted(set(votes.values()))}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['confirm', 'enable', 'disable', 'list'])
    parser.add_argument('--base-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--review', type=Path)
    parser.add_argument('--id')
    parser.add_argument('--key')
    parser.add_argument('--reason', default='')
    parser.add_argument('--openalex-id', action='append', default=[])
    parser.add_argument('--issn', action='append', default=[])
    args = parser.parse_args(argv)
    try:
        if args.command == 'confirm':
            if not args.review or not args.id or not args.reason.strip():
                parser.error('confirm requires --review, --id and --reason')
            result = confirm(args.base_dir, args.review, args.id, reason=args.reason, openalex_ids=args.openalex_id, issns=args.issn)
        elif args.command in {'enable', 'disable'}:
            if not args.key:
                parser.error('enable/disable requires --key')
            set_enabled(args.base_dir, args.key, args.command == 'enable')
            result = {'key': args.key, 'enabled': args.command == 'enable'}
        else:
            result = load_registry(args.base_dir)
    except (OSError, ValueError, KeyError, TypeError):
        parser.error('Invalid entity review/identity or inaccessible registry; no mapping was inferred from a name')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
