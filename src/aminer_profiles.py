"""On-demand AMiner entity dossiers; no automatic changes to tracking identities."""
import argparse
import html
import json
import re
from decimal import Decimal
from pathlib import Path

from .aminer_advanced import ResearchClient, ResearchAPIError
from .aminer_research import records, normalized
from .utils import atomic_json

KINDS={'person':('detail','figure','papers','projects'),
       'organization':('detail','people','papers'), 'venue':('detail','papers')}
DEFAULTS={'person':['detail','figure','papers'], 'organization':['detail','people','papers'], 'venue':['detail','papers']}


def request_for(kind, section, ident, offset=0, limit=10, year=None):
    if kind not in KINDS or section not in KINDS[kind]:
        raise ValueError('Unsupported profile section')
    endpoint=kind+'_'+section
    if kind=='organization' and section=='detail': params={'ids':[ident]}
    elif kind=='organization': params={'org_id':ident,'offset':offset}
    elif kind=='venue' and section=='papers': params={'id':ident,'offset':offset,'limit':limit,**({'year':year} if year else {})}
    else: params={'id':ident}
    return endpoint,params


def render(state, output):
    esc=lambda value:html.escape(str('' if value is None else value),quote=True)
    label={'detail':'基本资料','figure':'研究画像','papers':'论文列表','people':'学者列表','projects':'科研项目'}
    field_names={'name':'姓名/名称','name_en':'英文名称','name_zh':'中文名称','bio':'研究简介','bio_zh':'研究简介（中文）',
        'position':'职务','position_zh':'职务（中文）','orgs':'所属机构','org_zhs':'所属机构（中文）','edu':'教育经历','edu_zh':'教育经历（中文）',
        'honor':'荣誉','ai_domain':'研究领域','ai_interests':'研究兴趣','edus':'教育经历','works':'工作经历',
        'aliases':'别名','acronyms':'简称','details':'机构资料','issn':'ISSN','eissn':'电子 ISSN','type':'类型','alias':'别名'}
    def value_html(value):
        if isinstance(value,list):
            return '<ul>'+''.join('<li>'+value_html(v)+'</li>' for v in value)+'</ul>'
        if isinstance(value,dict):
            return '<dl>'+''.join(f'<dt>{esc(field_names.get(k,k))}</dt><dd>{value_html(v)}</dd>' for k,v in value.items() if k not in {'id','lower_alias'} and v not in (None,'',[]))+'</dl>'
        return '<span>'+esc(value)+'</span>'
    kind,ident=state['spec']['kind'],state['spec']['id']
    url={'person':'https://www.aminer.cn/profile/','venue':'https://www.aminer.cn/open/journal/detail/'}.get(kind)
    source_link=f'<a href="{esc(url+ident)}">查看来源实体 {esc(ident)}</a>' if url else f'机构 ID：{esc(ident)} · <a href="https://www.aminer.cn/open/docs">接口文档</a>'
    sections=[]
    candidates=[]
    detail=state['sections'].get('detail',{}).get('data',{})
    detail=detail[0] if isinstance(detail,list) and detail else detail
    display_name=next((detail.get(k) for k in ('name','name_en','name_zh') if isinstance(detail,dict) and detail.get(k)),ident)
    for section,result in state['sections'].items():
        data=result['data']
        if section=='papers':
            items=[]
            for row in records(data):
                paper=normalized(row,'entity_'+kind+'_papers')
                if not paper: continue
                # An explicit different scholar identity is conflicting provider evidence.
                if kind=='person' and row.get('author_id') and row['author_id']!=ident:
                    continue
                paper.extra['entity_relation']={'kind':kind,'aminer_id':ident,'evidence':'provider_relation_metadata'}
                candidates.append(paper.model_dump(mode='json'))
                items.append(f'<li><a href="{esc(paper.url)}">{esc(paper.title)}</a></li>')
            body='<ul>'+(''.join(items) or '<li>未返回可展示的论文身份。</li>')+'</ul>'
        else:
            body=value_html(data)
        sections.append(f'<section><h2>{label[section]}</h2>{body}</section>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AMiner 实体资料</title><style>body{max-width:1000px;margin:40px auto;padding:20px;font:16px/1.8 sans-serif;color:#24364b}section{border:1px solid #ddd;border-radius:12px;padding:20px;margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}li{margin:10px 0}a{color:#17627d}</style>'
    page+='<style>dt{font-weight:bold;margin-top:16px}dd{margin:4px 0 16px;overflow-wrap:anywhere}</style>'
    kind_label={'person':'学者资料','organization':'机构资料','venue':'期刊资料'}[kind]
    page+=f'<h1>{esc(display_name)} · {kind_label}</h1><p>{source_link}</p><p>资料来自 AMiner 数据库；不表示已逐条核对作者归属或机构任职，也不会自动写入你的追踪名单。列表可能被接口或页数上限截断。</p>'
    page+=f'<p>状态：{esc(state["status"])} · 本专题新增费用估算 ¥{state["incremental_estimated_yuan"]:.2f} · 共享预算累计估算 ¥{state["cost"]["estimated_reserved_yuan"]:.2f}</p><p><a href="papers-candidates.json">论文候选导出</a> · <a href="profile.json">原始结构化资料</a></p>'
    page+=''.join(sections)+'<h2>未完成项</h2><pre>'+esc(json.dumps(state['warnings'],ensure_ascii=False,indent=2))+'</pre></html>'
    (output/'index.html').write_text(page,'utf-8')
    atomic_json(output/'papers-candidates.json',candidates)


def run(args,client=None):
    output=args.output_dir.resolve()
    sections=args.section or DEFAULTS[args.kind]
    for section in sections: request_for(args.kind,section,args.id)
    spec={'kind':args.kind,'id':args.id,'sections':sections,'offset':args.offset,'limit':args.limit,'year':args.year}
    spec['api_directory']=str((args.api_run_dir or output/'api').resolve())
    spec['budget_yuan']=str(Decimal(args.budget_yuan).normalize())
    if output.exists() and any(output.iterdir()) and not args.resume: raise ValueError('Use a new directory or --resume')
    output.mkdir(parents=True,exist_ok=True)
    client=client or ResearchClient(args.api_run_dir or output/'api',budget_yuan=args.budget_yuan,allow_paid=args.allow_paid,cache_only=args.cache_only)
    from .research_archive import publication_lock
    with publication_lock(output):
        state=json.loads((output/'profile.json').read_text('utf-8')) if args.resume else {'spec':spec,'sections':{},'warnings':[],
            'initial_estimated_yuan':client.summary()['estimated_reserved_yuan'],'status':'running'}
        if state['spec']!=spec: raise ValueError('Profile request changed')
        if state.get('status')=='complete': return state
        state['warnings']=[]
        for section in sections:
            if section in state['sections']: continue
            endpoint,params=request_for(args.kind,section,args.id,args.offset,args.limit,args.year)
            try: state['sections'][section]=client.query(endpoint,params)
            except ResearchAPIError as exc: state['warnings'].append({'section':section,'reason':str(exc)})
            state['cost']=client.summary()
            state['incremental_estimated_yuan']=round(state['cost']['estimated_reserved_yuan']-state['initial_estimated_yuan'],2)
            atomic_json(output/'profile.json',state)
        state['status']='partial' if state['warnings'] else 'complete'
        atomic_json(output/'profile.json',state); render(state,output)
        return state


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=KINDS,required=True)
    parser.add_argument('--id',required=True)
    parser.add_argument('--section',action='append')
    parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--year',type=int)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--api-run-dir',type=Path)
    parser.add_argument('--budget-yuan',default='3.00')
    parser.add_argument('--allow-paid',action='store_true')
    parser.add_argument('--cache-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}',args.id) or not 0<=args.offset<=10000 or not 1<=args.limit<=100 or (args.year is not None and not 1500<=args.year<=2200): parser.error('Invalid entity/page bounds')
    return args


def main(argv=None):
    try:
        state=run(parse_args(argv)); print(json.dumps({'status':state['status'],'sections':list(state['sections']),'cost':state['cost']},ensure_ascii=False))
        raise SystemExit(0 if state['status']=='complete' else 2)
    except (ResearchAPIError,ValueError,OSError) as exc:
        print(json.dumps({'status':'failed','category':str(exc) if isinstance(exc,ResearchAPIError) else type(exc).__name__})); raise SystemExit(2) from None


if __name__=='__main__': main()
