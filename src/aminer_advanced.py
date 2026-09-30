"""Opt-in AMiner research APIs with a durable per-run estimated spending cap."""
import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import requests

from .aminer_client import BASE_URL
from .research_archive import publication_lock
from .utils import atomic_json


@dataclass(frozen=True)
class Endpoint:
    method: str
    path: str
    fen: int
    fields: str
    required: str = ''


# Official catalog checked 2026-10-01. Prices are estimates, not billing receipts.
# Patent endpoints are intentionally absent, including scholar/institution patents.
ENDPOINTS = {
    'paper_info': Endpoint('POST', '/api/paper/info', 0, 'ids', 'ids'),
    'paper_search_pro': Endpoint('GET', '/api/paper/search/pro', 1, 'page size title keyword abstract author org venue order'),
    'paper_qa_search': Endpoint('POST', '/api/paper/qa/search', 5, 'use_topic query topic_high topic_middle topic_low title doi year sci_flag n_citation_flag size offset force_citation_sort force_year_sort author_terms org_terms author_id org_id venue_ids', 'use_topic'),
    'paper_qa_search_pro': Endpoint('POST', '/api/paper/qa/searchPro', 30, 'query query_type cursor authors author_ids organizations organization_ids venues venue_ids year_values year_from year_to languages language_preference has_chinese_title has_abstract min_citations max_citations all_terms any_terms exclude_terms search_in paper_ids exclude_paper_ids dois sort'),
    'paper_detail': Endpoint('GET', '/api/paper/detail', 1, 'id', 'id'),
    'paper_references': Endpoint('GET', '/api/paper/relation', 10, 'id', 'id'),
    'paper_keywords': Endpoint('GET', '/api/paper/list/citation/by/keywords', 10, 'page size keywords', 'page size keywords'),
    'paper_by_venue_year': Endpoint('GET', '/api/paper/platform/allpubs/more/detail/by/ts/org/venue', 20, 'year venue_id', 'year venue_id'),
    'person_detail': Endpoint('GET', '/api/person/detail', 100, 'id', 'id'),
    'person_figure': Endpoint('GET', '/api/person/figure', 50, 'id', 'id'),
    'person_papers': Endpoint('GET', '/api/person/paper/relation', 150, 'id', 'id'),
    'person_projects': Endpoint('GET', '/api/project/person/v3/open', 150, 'id', 'id'),
    'organization_detail': Endpoint('POST', '/api/organization/detail', 1, 'ids', 'ids'),
    'organization_people': Endpoint('GET', '/api/organization/person/relation', 50, 'org_id offset', 'org_id'),
    'organization_papers': Endpoint('GET', '/api/organization/paper/relation', 10, 'org_id offset', 'org_id offset'),
    'organization_disambiguate': Endpoint('POST', '/api/organization/na', 1, 'org', 'org'),
    'organization_disambiguate_pro': Endpoint('POST', '/api/organization/na/pro', 5, 'org', 'org'),
    'venue_detail': Endpoint('POST', '/api/venue/detail', 20, 'id', 'id'),
    'venue_papers': Endpoint('POST', '/api/venue/paper/relation', 10, 'id offset limit year', 'id'),
    'experiment_search': Endpoint('POST', '/api/v3/paper/search/experiment_data/SearchPro', 10, 'paper_id method dataset search_text size'),
}


class ResearchAPIError(RuntimeError):
    """Only a controlled category; never a remote message or credential-bearing URL."""


def validate_request(endpoint, params):
    if endpoint not in ENDPOINTS:
        raise ResearchAPIError('endpoint_not_allowed')
    spec = ENDPOINTS[endpoint]
    if not isinstance(params, dict) or set(params) - set(spec.fields.split()):
        raise ResearchAPIError('unsupported_parameters')
    if len(json.dumps(params, ensure_ascii=False)) > 32000:
        raise ResearchAPIError('oversized_parameters')
    if any(k not in params or params[k] is None or params[k] == '' or params[k] == [] for k in spec.required.split()):
        raise ResearchAPIError('missing_parameters')
    for key in ('page', 'offset', 'size', 'limit', 'year', 'year_from', 'year_to', 'min_citations', 'max_citations'):
        if key in params and (type(params[key]) is not int or params[key] < 0):
            raise ResearchAPIError('invalid_numeric_parameter')
    for key in ('size', 'limit'):
        if key in params and not 1 <= params[key] <= 100:
            raise ResearchAPIError('invalid_page_size')
    for key in ('id', 'org_id', 'venue_id', 'paper_id'):
        if key in params and (not isinstance(params[key], str) or len(params[key]) > 128 or not params[key].strip()):
            if not (endpoint == 'experiment_search' and key == 'paper_id' and params[key] == ''):
                raise ResearchAPIError('invalid_identity')
    if 'ids' in params and (not isinstance(params['ids'], list) or not 1 <= len(params['ids']) <= 100 or not all(isinstance(v, str) and 0 < len(v) <= 128 for v in params['ids'])):
        raise ResearchAPIError('invalid_identity_list')
    if endpoint == 'paper_search_pro' and not any(str(params.get(k, '')).strip() for k in ('title','keyword','abstract','author','org','venue')):
        raise ResearchAPIError('missing_search_terms')
    for key in ('query','title','keyword','abstract','author','org','venue','order','query_type','sort','search_in'):
        if key in params and not (endpoint == 'paper_qa_search' and key == 'title') and (not isinstance(params[key],str) or len(params[key]) > 2000):
            raise ResearchAPIError('invalid_text_parameter')
    if endpoint == 'paper_search_pro' and 'order' in params and params['order'] not in {'year','n_citation'}:
        raise ResearchAPIError('invalid_sort')
    if endpoint == 'paper_qa_search':
        if type(params.get('use_topic')) is not bool or (params.get('query') and params['use_topic'] is not True):
            raise ResearchAPIError('query_requires_topic_mode')
        if params.get('query') and any(params.get(k) for k in ('topic_high','topic_middle','topic_low')):
            raise ResearchAPIError('query_and_weighted_topics_conflict')
    if endpoint == 'paper_qa_search_pro':
        if len(params.get('query','')) > 500 or params.get('query_type','auto') not in {'auto','topic','keywords','title','identifier'} or params.get('sort','relevance') not in {'relevance','balanced','recent','citation'}:
            raise ResearchAPIError('invalid_semantic_query')
        if 'cursor' in params and (set(params) != {'cursor'} or not isinstance(params['cursor'], str) or not 16 <= len(params['cursor']) <= 256):
            raise ResearchAPIError('cursor_only_request_required')
        if 'year_values' in params and any(k in params for k in ('year_from','year_to')):
            raise ResearchAPIError('conflicting_year_filters')
        if params.get('year_from', 0) > params.get('year_to', 9999):
            raise ResearchAPIError('invalid_year_range')
        if not params:
            raise ResearchAPIError('missing_search_terms')
    if endpoint == 'experiment_search':
        for key in ('paper_id', 'method', 'dataset', 'search_text'):
            if key in params and not isinstance(params[key], str):
                raise ResearchAPIError('invalid_experiment_filter')
        params = {**{k: str(params.get(k, '')).strip() for k in ('paper_id','method','dataset','search_text')}, **({'size':params['size']} if 'size' in params else {})}
        if not any(params[k] for k in ('paper_id','method','dataset','search_text')):
            raise ResearchAPIError('missing_experiment_filter')
    return dict(params)


def experiment_rows(value, depth=0):
    if depth > 8:
        raise ResearchAPIError('invalid_experiment_response')
    if isinstance(value, list) and all(isinstance(v, dict) and ('paper_id' in v or 'experiment_name' in v) for v in value):
        return value
    if isinstance(value, dict):
        if 'paper_id' in value or 'experiment_name' in value:
            return [value]
        for key in ('results','data','items','experiments','records'):
            if key in value:
                return experiment_rows(value[key], depth + 1)
    raise ResearchAPIError('invalid_experiment_response')


class ResearchClient:
    def __init__(self, directory, *, budget_yuan='1.00', allow_paid=False, cache_only=False, token=None, session=None, timeout=25):
        amount = Decimal(str(budget_yuan)) * 100
        if not amount.is_finite() or amount != int(amount) or not 0 < amount <= 500:
            raise ValueError('Research budget must be 0.01..5.00 yuan')
        self.cap = int(amount)
        self.directory = Path(directory)
        self.allow_paid = allow_paid
        self.cache_only = cache_only
        self.token = (token if token is not None else os.getenv('AMINER_API_KEY', '')).removeprefix('Bearer ').strip()
        self.session = session or requests.Session()
        self.timeout = min(60, max(1, timeout))

    def _ledger(self):
        path = self.directory / 'spending.json'
        if not path.exists():
            if (self.directory/'responses').exists() and any((self.directory/'responses').iterdir()):
                raise ResearchAPIError('missing_budget_ledger')
            return {'schema_version':1, 'cap_fen':self.cap, 'reserved_fen':0, 'attempts':[]}
        data = json.loads(path.read_text('utf-8'))
        if not isinstance(data,dict) or data.get('schema_version') != 1 or data.get('cap_fen') != self.cap or not isinstance(data.get('attempts'), list):
            raise ResearchAPIError('budget_ledger_mismatch')
        if any(not isinstance(row,dict) or row.get('endpoint') not in ENDPOINTS or type(row.get('estimated_fen')) is not int or row['estimated_fen']!=ENDPOINTS[row['endpoint']].fen for row in data['attempts']):
            raise ResearchAPIError('invalid_budget_ledger')
        if (type(data.get('reserved_fen')) is not int or not 0 <= data['reserved_fen'] <= self.cap or
                data['reserved_fen'] != sum(row['estimated_fen'] for row in data['attempts'])):
            raise ResearchAPIError('invalid_budget_ledger')
        return data

    def summary(self):
        state = self._ledger()
        return {'budget_yuan':self.cap/100, 'estimated_reserved_yuan':state['reserved_fen']/100,
            'attempts':len(state['attempts']), 'actual_billed_yuan':None,
            'note':'Conservative published-price estimate for every attempted request, including failures. Not an invoice.'}

    def query(self, endpoint, params):
        params = validate_request(endpoint, params)
        if not self.allow_paid and not self.cache_only:
            raise ResearchAPIError('paid_calls_not_enabled')
        if not self.token and not self.cache_only:
            raise ResearchAPIError('credential_missing')
        spec = ENDPOINTS[endpoint]
        key = hashlib.sha256(json.dumps([1,endpoint,params], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with publication_lock(self.directory):
            ledger = self._ledger()
            if ledger.get('account_blocked') and not self.cache_only:
                raise ResearchAPIError('account_attention_required')
            cache = self.directory / 'responses' / (key + '.json')
            if cache.exists():
                cached = json.loads(cache.read_text('utf-8'))
                canonical = json.dumps(cached['result'], sort_keys=True, ensure_ascii=False)
                if cached.get('key') != key or hashlib.sha256(canonical.encode()).hexdigest() != cached.get('sha256'):
                    raise ResearchAPIError('corrupt_response_cache')
                return cached['result']
            if self.cache_only:
                raise ResearchAPIError('cache_miss')
            if ledger['reserved_fen'] + spec.fen > self.cap:
                raise ResearchAPIError('budget_exhausted')
            if ledger.get('cooldown_until', 0) > time.time():
                raise ResearchAPIError('rate_limit_cooldown')
            attempt = {'endpoint':endpoint, 'key':key, 'estimated_fen':spec.fen, 'status':'reserved'}
            ledger['attempts'].append(attempt); ledger['reserved_fen'] += spec.fen
            # Reserve before network I/O; a killed process never refunds a possibly billed call.
            atomic_json(self.directory/'spending.json',ledger)
            try:
                response = self.session.request(spec.method, BASE_URL+spec.path,
                    headers={'Authorization':self.token, 'X-Platform':'zotwatch','X-Skill-Name':'zotwatch-advanced-research','X-Skill-Version':'1'},
                    **({'params':params} if spec.method=='GET' else {'json':params}), timeout=self.timeout, allow_redirects=False)
                try:
                    body = response.json()
                except ValueError:
                    raise ResearchAPIError('invalid_json') from None
                code = str(body.get('code','')) if isinstance(body,dict) else ''
                if response.status_code == 429 or code == '40306':
                    ledger['cooldown_until'] = time.time()+60
                    raise ResearchAPIError('rate_limited')
                if response.status_code == 401 or code in {'401','40308'}:
                    ledger['account_blocked']=True
                    raise ResearchAPIError('credential_permission_or_balance')
                if response.status_code == 403 or code in {'40301','40302','40307'}:
                    raise ResearchAPIError('credential_permission_or_balance')
                if response.status_code != 200:
                    raise ResearchAPIError('http_failure')
                if isinstance(body,dict) and (body.get('success') is False or code not in {'','0','200'}):
                    raise ResearchAPIError('business_failure')
                if endpoint == 'experiment_search':
                    data = experiment_rows(body)
                elif isinstance(body,dict) and 'data' in body:
                    data = body['data']
                else:
                    raise ResearchAPIError('invalid_response_envelope')
                result = {'endpoint':endpoint,'data':data,'next_cursor':body.get('next_cursor') if isinstance(body,dict) else None,
                          'evidence_level':'aminer_metadata_or_extracted_records_not_verified_fulltext'}
                if isinstance(data,dict) and data.get('next_cursor'):
                    result['next_cursor'] = data['next_cursor']
                canonical = json.dumps(result, sort_keys=True, ensure_ascii=False)
                atomic_json(cache,{'key':key,'sha256':hashlib.sha256(canonical.encode()).hexdigest(),'result':result})
                attempt['status']='success'
                return result
            except requests.RequestException:
                attempt['status']='network_failure'
                raise ResearchAPIError('network_failure') from None
            except ResearchAPIError as exc:
                attempt['status']=str(exc)
                raise
            finally:
                atomic_json(self.directory/'spending.json',ledger)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['catalog','query'])
    parser.add_argument('--endpoint',choices=sorted(ENDPOINTS))
    parser.add_argument('--params-file',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--budget-yuan',default='1.00')
    parser.add_argument('--allow-paid',action='store_true')
    parser.add_argument('--cache-only',action='store_true',help='Read cached responses only; never contact an API')
    args=parser.parse_args(argv)
    if args.command=='catalog':
        print(json.dumps({k:{'method':v.method,'path':v.path,'estimated_yuan':v.fen/100,'fields':v.fields} for k,v in ENDPOINTS.items()},ensure_ascii=False,indent=2)); return
    if not args.endpoint or not args.params_file or not args.output_dir:
        parser.error('query requires --endpoint, --params-file and --output-dir')
    client=None
    try:
        client=ResearchClient(args.output_dir,budget_yuan=args.budget_yuan,allow_paid=args.allow_paid,cache_only=args.cache_only)
        result=client.query(args.endpoint,json.loads(args.params_file.read_text('utf-8')))
        print(json.dumps({'status':'success','result':result,'cost':client.summary()},ensure_ascii=False,indent=2))
    except (ResearchAPIError, OSError, ValueError) as exc:
        print(json.dumps({'status':'failed','category':str(exc) if isinstance(exc,ResearchAPIError) else type(exc).__name__}))
        raise SystemExit(2) from None


if __name__=='__main__': main()
