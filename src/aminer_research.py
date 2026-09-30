"""Bounded on-demand advanced search, backward citation expansion and experiment cards."""
import argparse
import csv
import hashlib
import html
import json
import re
import time
from pathlib import Path

from .aminer_advanced import ResearchClient, ResearchAPIError
from .aminer_source import normalize_paper, clean_doi
from .citation_watch import merge_candidates
from .research_dossier import build_dossier, export_dossier
from .settings import load_settings
from .utils import atomic_json


def records(value):
    if isinstance(value,list):
        return [row for row in value if isinstance(row,dict)]
    if isinstance(value,dict):
        for key in ('items','papers','results','records','data'):
            if key in value:
                return records(value[key])
        if any(key in value for key in ('id','paper_id','_id')):
            return [value]
    return []


def normalized(row, route):
    copy=dict(row)
    copy['id']=row.get('id') or row.get('paper_id') or row.get('_id')
    paper=normalize_paper(copy,route=route)
    if paper and type(row.get('n_citation')) is int and row['n_citation'] >= 0:
        paper.metrics['cited_by']=row['n_citation']
    return paper


def references(value, source):
    rows=records(value)
    if any('cited' in row for row in rows):
        result=[]
        for row in rows:
            if (row.get('_id') or row.get('id'))==source and isinstance(row.get('cited'),list):
                result.extend(r for r in row['cited'] if isinstance(r,dict))
        return result
    return rows


def export_report(state, output):
    esc=lambda x:html.escape(str(x or ''),quote=True)
    papers=state['papers']
    links={key:'https://www.aminer.cn/pub/'+key for key in papers}
    cards=[]
    for key,paper in papers.items():
        outgoing=[e for e in state['edges'] if e['from']==key]
        refs=''.join(f'<li><a href="{esc(links[e["to"]])}">{esc(papers[e["to"]]["title"])}</a></li>' for e in outgoing)
        cards.append(f'<article><h2><a href="{esc(links[key])}">{esc(paper["title"])}</a></h2>'
            f'<p>年份：{esc(paper.get("extra",{}).get("publication_year","未知"))} · {esc(paper.get("venue"))}</p>'
            f'<p>{esc(paper.get("abstract") or "缺少原始摘要")}</p>'
            f'<details><summary>数据库记录的参考文献：{len(outgoing)} 条</summary><ul>{refs or "<li>本轮未扩展或未返回；不代表没有参考文献。</li>"}</ul></details></article>')
    experiments=[]
    for key,rows in state['experiments'].items():
        for row in rows:
            # Preserve original arrays/fields in JSON. Never flatten multiple methods.
            experiments.append(f'<article><h2>{esc(row.get("experiment_name") or "实验记录")}</h2>'
                f'<p>来源论文：{esc(papers[key]["title"])}</p><pre>{esc(json.dumps(row,ensure_ascii=False,indent=2))}</pre></article>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AMiner 专题深检索</title><style>body{max-width:1050px;margin:40px auto;padding:20px;font:16px/1.8 sans-serif;color:#243347}article{border:1px solid #ddd;border-radius:12px;padding:20px;margin:20px 0}h2{font-size:19px}pre{white-space:pre-wrap;overflow-wrap:anywhere}input{padding:10px;width:90%}a{color:#16627e}</style>'
    page+=f'<h1>{esc(state["spec"]["query"])}</h1><p>候选 {len(papers)} / 目标 {state["spec"]["target"]} 篇 · 数据库引文边 {len(state["edges"])} 条 · 实验记录 {sum(map(len,state["experiments"].values()))} 条</p>'
    page+='<p>候选池尚未人工筛选。引文关系来自数据库，不证明支持、反驳或方法继承；实验记录是平台结构化数据，尚未对照全文核验。</p>'
    page+=f'<p>状态：{esc(state["status"])} · 共享 API 预算累计估算：¥{state["cost"]["estimated_reserved_yuan"]:.2f} / ¥{state["cost"]["budget_yuan"]:.2f} · 本专题新增估算：¥{state.get("incremental_estimated_yuan",0):.2f}（实际账单未读取）</p>'
    page+='<p>'+('<a href="dossier/dossier.html">摘要证据比较</a> · ' if papers else '')+'<a href="papers.csv">候选表 CSV</a> · <a href="citation-graph.json">引文图数据 JSON</a> · <a href="experiments.json">原始实验记录 JSON</a></p>'
    page+='<label>筛选论文或实验 <input id="filter" type="search"></label><h2>候选文献与引用扩展</h2>'+''.join(cards)
    page+='<h2>实验记录</h2>'+(''.join(experiments) or '<p>本轮未获取实验记录；请补充原文或换用明确的实验检索条件。</p>')
    page+='<details><summary>覆盖与失败记录</summary><pre>'+esc(json.dumps(state['warnings'],ensure_ascii=False,indent=2))+'</pre></details>'
    page+='<script>document.getElementById("filter").addEventListener("input",function(){const q=this.value.toLowerCase();document.querySelectorAll("article").forEach(e=>e.hidden=!e.textContent.toLowerCase().includes(q));});</script></html>'
    (output/'index.html').write_text(page,'utf-8')
    atomic_json(output/'citation-graph.json',{'nodes':[{'id':key,'title':row['title'],'url':links[key]} for key,row in papers.items()],
        'edges':state['edges'],'evidence_level':'provider_metadata_not_local_citation_verification'})
    atomic_json(output/'experiments.json',{'results_by_paper':state['experiments'], 'evidence_level':'provider_structured_extraction_unverified'})
    with (output/'papers.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['aminer_id','title','doi','year','abstract_evidence','url'])
        for key,row in papers.items():
            values=[key,row['title'],row.get('doi'),row['extra'].get('publication_year'),
                'abstract_slice' if row['extra'].get('abstract_is_partial') else 'abstract' if row.get('abstract') else 'title_only',links[key]]
            writer.writerow(["'"+str(v) if str(v or '').lstrip().startswith(('=','+','-','@')) else str(v or '') for v in values])


def run(args, client=None):
    settings=load_settings(args.base_dir)
    output=args.output_dir.resolve()
    config_hash=hashlib.sha256((args.base_dir/'config/research.yaml').read_bytes()).hexdigest()
    spec={k:getattr(args,k) for k in ('query','mode','pages','page_size','target','year_from','year_to','depth','seeds_per_depth','details','experiments','seed_id')}
    spec['research_config_sha256']=config_hash
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Use a new output directory or explicit --resume')
    output.mkdir(parents=True,exist_ok=True)
    client=client or ResearchClient(args.api_run_dir or output/'api',budget_yuan=args.budget_yuan,allow_paid=args.allow_paid,cache_only=args.cache_only)
    from .research_archive import publication_lock
    with publication_lock(output):
        state=json.loads((output/'run.json').read_text('utf-8')) if args.resume else {
            'spec':spec,'papers':{},'edges':[],'experiments':{},'completed':[], 'warnings':[], 'status':'running',
            'initial_estimated_yuan':client.summary()['estimated_reserved_yuan']}
        if state['spec']!=spec:
            raise ValueError('Research input changed; start a new run')
        if state.get('status')=='complete':
            return state
        started=time.monotonic(); incomplete=False
        def save():
            state['cost']=client.summary()
            state['incremental_estimated_yuan']=round(state['cost']['estimated_reserved_yuan']-state['initial_estimated_yuan'],2)
            atomic_json(output/'run.json',state)
        def query(label, endpoint, params):
            nonlocal incomplete
            if time.monotonic()-started > args.deadline:
                incomplete=True; state['warnings'].append({'operation':label,'reason':'run_deadline'}); return None
            try:
                result=client.query(endpoint,params)
                if isinstance(result.get('data'),dict) and result['data'].get('warnings'):
                    state['warnings'].append({'operation':label,'provider_warnings':result['data']['warnings']})
                if label not in state['completed']: state['completed'].append(label)
                return result
            except ResearchAPIError as exc:
                incomplete=True; state['warnings'].append({'operation':label,'reason':str(exc)}); return None
        def add(row, route, *, force_seed=False):
            paper=normalized(row,route)
            if not paper: return None
            year=paper.extra.get('publication_year')
            if not force_seed and ((args.year_from is not None or args.year_to is not None) and
                    (not year or (args.year_from and year<args.year_from) or (args.year_to and year>args.year_to))):
                state.setdefault('rejected',{}).setdefault('year_missing_or_outside_range',0)
                state['rejected']['year_missing_or_outside_range']+=1
                return None
            ident=paper.extra['aminer_id']
            if ident in state['papers']:
                from .models import CandidateWork
                previous=CandidateWork.model_validate(state['papers'][ident])
                if previous.doi and paper.doi and clean_doi(previous.doi)!=clean_doi(paper.doi):
                    state['warnings'].append({'operation':'identity','reason':'conflicting_doi','id':ident}); return None
                paper=merge_candidates([previous,paper])[0]
            state['papers'][ident]=paper.model_dump(mode='json')
            return ident
        save()
        cursor=None; cursors=set()
        for page in range(args.pages):
            if len(state['papers'])>=args.target: break
            if args.mode=='pro':
                endpoint='paper_search_pro'; params={'title':args.query,'page':page,'size':args.page_size}
            else:
                endpoint='paper_qa_search_pro'
                params={'cursor':cursor} if cursor else {'query':args.query,'query_type':'auto','sort':'relevance',
                    **({'year_from':args.year_from} if args.year_from else {}),**({'year_to':args.year_to} if args.year_to else {})}
            result=query('search:'+str(page),endpoint,params)
            if result is None: break
            batch=records(result['data'])
            for row in batch:
                if len(state['papers'])<args.target: add(row,endpoint)
            save()
            if args.mode=='semantic':
                cursor=result.get('next_cursor')
                if not cursor: break
                if cursor in cursors:
                    incomplete=True; state['warnings'].append({'reason':'repeated_cursor'}); break
                cursors.add(cursor)
            elif len(batch)<args.page_size:
                break
        for ident in args.seed_id:
            result=query('seed:'+ident,'paper_detail',{'id':ident})
            if result:
                for row in records(result['data']):
                    if (row.get('id') or row.get('paper_id'))==ident: add(row,'paper_detail',force_seed=ident in args.seed_id)
        frontier=list(args.seed_id or state['papers'])[:args.seeds_per_depth]
        expanded=set()
        for level in range(args.depth):
            following=[]
            for source in frontier:
                if source not in state['papers'] or source in expanded: continue
                result=query('references:'+source,'paper_references',{'id':source})
                if result is None: continue
                expanded.add(source)
                cited=references(result['data'],source)
                if len(cited)>100:
                    state['warnings'].append({'operation':'references:'+source,'reason':'per_seed_reference_cap','returned':len(cited),'processed':100})
                for row in cited[:100]:
                    ident=row.get('id') or row.get('paper_id') or row.get('_id')
                    if len(state['papers'])>=args.target and ident not in state['papers']: continue
                    target=add(row,'paper_references')
                    if not target or target==source: continue
                    edge={'from':source,'to':target,'relation':'cites','evidence':'aminer_relation_metadata','depth':level+1}
                    if not any(e['from']==source and e['to']==target for e in state['edges']): state['edges'].append(edge)
                    if target not in expanded: following.append(target)
                save()
            frontier=list(dict.fromkeys(following))[:args.seeds_per_depth]
        for ident in list(state['papers'])[:args.details]:
            result=query('details:'+ident,'paper_detail',{'id':ident})
            if result:
                for row in records(result['data']):
                    if (row.get('id') or row.get('paper_id'))==ident: add(row,'paper_detail',force_seed=ident in args.seed_id)
            save()
        for ident in list(state['papers'])[:args.experiments]:
            result=query('experiments:'+ident,'experiment_search',{'paper_id':ident,'size':5})
            if result:
                state['experiments'][ident]=result['data']
            save()
        # Keep provider citation edges out of the locally verified citation ledger.
        from .models import CandidateWork
        selected=[CandidateWork.model_validate(row) for row in list(state['papers'].values())[:20]]
        if selected:
            ledger=build_dossier(args.query,selected,settings.research)
            export_dossier(ledger,output/'dossier')
        state['status']='partial' if incomplete else 'complete'
        state['target_reached']=len(state['papers'])>=args.target
        state['coverage_note']='Bounded candidate collection; reaching a limit or an empty result does not prove literature completeness.'
        save(); export_report(state,output)
        return state


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--query',required=True)
    parser.add_argument('--mode',choices=['pro','semantic'],default='semantic')
    parser.add_argument('--pages',type=int,default=1)
    parser.add_argument('--page-size',type=int,default=20)
    parser.add_argument('--target',type=int,default=50)
    parser.add_argument('--year-from',type=int)
    parser.add_argument('--year-to',type=int)
    parser.add_argument('--depth',type=int,default=1)
    parser.add_argument('--seeds-per-depth',type=int,default=2)
    parser.add_argument('--seed-id',action='append',default=[])
    parser.add_argument('--details',type=int,default=5)
    parser.add_argument('--experiments',type=int,default=0)
    parser.add_argument('--budget-yuan',default='1.00')
    parser.add_argument('--allow-paid',action='store_true')
    parser.add_argument('--cache-only',action='store_true')
    parser.add_argument('--api-run-dir',type=Path,help='Reuse one API cache and spending cap across research operations')
    parser.add_argument('--deadline',type=int,default=180)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if not args.query.strip() or len(args.query)>500 or not 1<=args.pages<=5 or not 1<=args.page_size<=100 or not 1<=args.target<=500:
        parser.error('Invalid query/search bounds')
    if not 0<=args.depth<=2 or not 1<=args.seeds_per_depth<=5 or not 0<=args.details<=20 or not 0<=args.experiments<=5 or not 10<=args.deadline<=600:
        parser.error('Invalid enrichment bounds')
    if len(args.seed_id)>5 or any(not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}',v) for v in args.seed_id): parser.error('Invalid seed IDs')
    if any(v is not None and not 1500<=v<=2200 for v in (args.year_from,args.year_to)) or (args.year_from and args.year_to and args.year_from>args.year_to): parser.error('Invalid year range')
    return args


def main(argv=None):
    try:
        state=run(parse_args(argv))
        print(json.dumps({'status':state['status'],'papers':len(state['papers']),'edges':len(state['edges']),'cost':state['cost']},ensure_ascii=False))
        raise SystemExit(0 if state['status']=='complete' else 2)
    except (ValueError,OSError,ResearchAPIError) as exc:
        print(json.dumps({'status':'failed','category':str(exc) if isinstance(exc,ResearchAPIError) else type(exc).__name__}))
        raise SystemExit(2) from None


if __name__=='__main__': main()
