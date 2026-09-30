import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.aminer_advanced import ResearchAPIError
from src.aminer_research import parse_args, run, references
from src.aminer_profiles import parse_args as profile_args, run as run_profile

ROOT=Path(__file__).resolve().parents[1]


class ResearchFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.output=Path(self.tmp.name)/'out'
        self.client=Mock()
        self.client.summary.return_value={'estimated_reserved_yuan':0,'budget_yuan':1}

    def args(self,*extra):
        return parse_args(['--base-dir',str(ROOT),'--query','ductile fracture','--mode','pro','--page-size','2',
            '--details','0','--experiments','0','--depth','1','--seeds-per-depth','1','--output-dir',str(self.output),*extra])

    def test_provider_reference_envelope_cannot_reverse_or_cross_source_edges(self):
        data=[{'_id':'source','cited':[{'_id':'target','title':'Target'}]}]
        self.assertEqual(references(data,'source')[0]['_id'],'target')
        self.assertEqual(references(data,'different'),[])

    def test_research_exports_metadata_graph_not_grounded_citation_claims(self):
        def query(endpoint,params):
            if endpoint=='paper_search_pro': return {'data':[{'id':'a','title':'Steel ductile fracture','year':2025}]}
            return {'data':[{'_id':'a','cited':[{'_id':'b','title':'Calibration of steel fracture'}]}]}
        self.client.query.side_effect=query
        state=run(self.args(),self.client)
        self.assertEqual(state['status'],'complete')
        self.assertEqual(state['edges'][0]['from'],'a'); self.assertEqual(state['edges'][0]['to'],'b')
        dossier=json.loads((self.output/'dossier/evidence-ledger.json').read_text('utf-8'))
        self.assertEqual(dossier['citations'],[])
        self.assertEqual(len(json.loads((self.output/'citation-graph.json').read_text())['edges']),1)
        before=self.client.query.call_count
        run(self.args('--resume'),self.client)
        self.assertEqual(self.client.query.call_count,before)

    def test_year_constraint_rejects_unknown_and_out_of_range(self):
        self.client.query.return_value={'data':[{'id':'a','title':'A valid year','year':2025},
            {'id':'b','title':'Old work','year':2010},{'id':'c','title':'Unknown year'}]}
        state=run(self.args('--year-from','2024','--year-to','2026','--depth','0'),self.client)
        self.assertEqual(list(state['papers']),['a'])

    def test_budget_failure_leaves_partial_outputs_not_a_successful_empty_review(self):
        self.client.query.side_effect=ResearchAPIError('budget_exhausted')
        state=run(self.args(),self.client)
        self.assertEqual(state['status'],'partial')
        self.assertTrue((self.output/'index.html').exists())
        self.assertNotIn('href="dossier/',(self.output/'index.html').read_text('utf-8'))

    def test_partial_then_resume_retains_original_request_contract(self):
        self.client.query.side_effect=ResearchAPIError('network_failure')
        run(self.args(),self.client)
        self.client.query.side_effect=None; self.client.query.return_value={'data':[]}
        self.assertEqual(run(self.args('--resume'),self.client)['status'],'complete')
        with self.assertRaises(ValueError): run(self.args('--resume','--target','20'),self.client)

    def test_cursor_loop_stops_with_visible_incomplete_status(self):
        self.client.query.return_value={'data':{'items':[{'paper_id':'a','title':'Steel fracture','year':2025}]},'next_cursor':'x'*20}
        state=run(self.args('--mode','semantic','--pages','3','--depth','0'),self.client)
        self.assertEqual(state['status'],'partial')
        self.assertEqual(self.client.query.call_count,2)
        self.assertEqual(self.client.query.call_args.args[1],{'cursor':'x'*20})

    def test_experiment_records_are_preserved_and_source_text_is_escaped(self):
        row={'paper_id':'a','experiment_name':'<img src=x onerror=bad()>','methods':[{'name':'A'},{'name':'B'}]}
        def query(endpoint,params):
            return {'data':[row]} if endpoint=='experiment_search' else {'data':[{'id':'a','title':'Steel fracture'}]}
        self.client.query.side_effect=query
        run(self.args('--depth','0','--experiments','1'),self.client)
        payload=json.loads((self.output/'experiments.json').read_text('utf-8'))
        self.assertEqual(payload['results_by_paper']['a'][0]['methods'],row['methods'])
        self.assertNotIn('<img src=x',(self.output/'index.html').read_text('utf-8'))

    def test_entity_dossier_exports_only_identity_consistent_paper_relations(self):
        def query(endpoint,params):
            if endpoint=='person_papers': return {'data':[{'id':'a','title':'Paper A','author_id':'p'}, {'id':'b','title':'Wrong author','author_id':'other'}]}
            return {'data':{'id':'p','name':'<script>bad</script>'}}
        self.client.query.side_effect=query
        args=profile_args(['--kind','person','--id','p','--output-dir',str(self.output)])
        state=run_profile(args,self.client)
        self.assertEqual(state['status'],'complete')
        self.assertEqual(len(json.loads((self.output/'papers-candidates.json').read_text('utf-8'))),1)
        self.assertNotIn('<script>bad',(self.output/'index.html').read_text('utf-8'))
        self.assertFalse((ROOT/'config/entity-tracking.json').samefile(self.output/'profile.json'))

    def test_entity_resume_retries_only_failed_sections(self):
        def query(endpoint,params):
            if endpoint=='venue_papers': raise ResearchAPIError('network_failure')
            return {'data':[]}
        self.client.query.side_effect=query
        args=profile_args(['--kind','venue','--id','v','--output-dir',str(self.output)])
        self.assertEqual(run_profile(args,self.client)['status'],'partial')
        self.client.query.reset_mock(); self.client.query.side_effect=None; self.client.query.return_value={'data':[]}
        args.resume=True
        self.assertEqual(run_profile(args,self.client)['status'],'complete')
        self.assertEqual(self.client.query.call_count,1)


if __name__=='__main__': unittest.main()
