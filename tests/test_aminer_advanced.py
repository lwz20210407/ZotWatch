import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import requests

from src.aminer_advanced import ResearchClient, ResearchAPIError, ENDPOINTS, validate_request, experiment_rows


def response(data, **fields):
    result=Mock(status_code=200)
    result.json.return_value={'success':True,'code':200,'data':data,**fields}
    return result


class AdvancedClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.session=Mock()

    def client(self, **kwargs):
        return ResearchClient(self.root,session=self.session,token='secret-token',**kwargs)

    def test_opt_in_and_patent_exclusion_before_network(self):
        with self.assertRaisesRegex(ResearchAPIError,'paid_calls_not_enabled'):
            self.client().query('paper_detail',{'id':'p'})
        for name in ('patent_detail','person_patents','organization_patents'):
            with self.assertRaisesRegex(ResearchAPIError,'endpoint_not_allowed'):
                self.client(allow_paid=True).query(name,{'id':'p'})
        self.session.request.assert_not_called()
        self.assertFalse(any('patent' in e.path for e in ENDPOINTS.values()))

    def test_budget_is_durable_counts_failed_attempts_and_never_retries(self):
        self.session.request.side_effect=requests.Timeout('secret-token in exception')
        first=self.client(allow_paid=True,budget_yuan='.01')
        with self.assertRaisesRegex(ResearchAPIError,'^network_failure$'):
            first.query('paper_detail',{'id':'p'})
        self.assertEqual(self.session.request.call_count,1)
        resumed=self.client(allow_paid=True,budget_yuan='.01')
        with self.assertRaisesRegex(ResearchAPIError,'budget_exhausted'):
            resumed.query('paper_detail',{'id':'p'})
        self.assertEqual(resumed.summary()['estimated_reserved_yuan'],.01)
        self.assertNotIn('secret-token',(self.root/'spending.json').read_text())

    def test_success_cache_avoids_new_charge_and_preserves_data(self):
        self.session.request.return_value=response([{'id':'p','abstract':'Original'}])
        first=self.client(allow_paid=True,budget_yuan='.01').query('paper_detail',{'id':'p'})
        second=self.client(allow_paid=True,budget_yuan='.01').query('paper_detail',{'id':'p'})
        self.assertEqual(first,second); self.assertEqual(self.session.request.call_count,1)
        self.assertFalse(self.session.request.call_args.kwargs['allow_redirects'])
        self.assertNotIn('secret-token',''.join(p.read_text() for p in self.root.rglob('*.json')))

    def test_cache_only_needs_no_token_and_never_spends_on_miss(self):
        self.session.request.return_value=response([{'id':'p'}])
        self.client(allow_paid=True).query('paper_detail',{'id':'p'})
        offline=ResearchClient(self.root,cache_only=True,token='',session=self.session)
        self.assertEqual(offline.query('paper_detail',{'id':'p'})['data'],[{'id':'p'}])
        before=(self.root/'spending.json').read_bytes()
        with self.assertRaisesRegex(ResearchAPIError,'cache_miss'): offline.query('paper_detail',{'id':'uncached'})
        self.assertEqual(self.session.request.call_count,1)
        self.assertEqual(before,(self.root/'spending.json').read_bytes())

    def test_cache_corruption_and_budget_change_fail_closed(self):
        self.session.request.return_value=response([])
        self.client(allow_paid=True).query('paper_detail',{'id':'p'})
        with self.assertRaisesRegex(ResearchAPIError,'budget_ledger_mismatch'):
            self.client(allow_paid=True,budget_yuan=2).query('paper_detail',{'id':'p'})
        path=next((self.root/'responses').glob('*.json')); data=json.loads(path.read_text()); data['result']['data']=['changed']
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ResearchAPIError,'corrupt_response_cache'):
            self.client(allow_paid=True).query('paper_detail',{'id':'p'})

    def test_invalid_request_never_reserves_cost(self):
        invalid=[('paper_qa_search_pro',{'query':'x','size':10}),
            ('paper_qa_search_pro',{'cursor':'a'*20,'query':'x'}),
            ('paper_qa_search',{'use_topic':False,'query':'x'}),
            ('paper_qa_search',{'use_topic':True,'query':'x','topic_high':'[["x"]]'}),
            ('paper_search_pro',{'page':0,'size':100}),
            ('experiment_search',{'experiment_name':'unsupported'}),
            ('experiment_search',{}), ('venue_papers',{'id':'p','limit':101})]
        for endpoint,params in invalid:
            with self.assertRaises(ResearchAPIError): self.client(allow_paid=True).query(endpoint,params)
        self.assertFalse((self.root/'spending.json').exists())

    def test_documented_list_parameters_are_accepted(self):
        """The catalogue types paper_qa_search year as []number and the ids as []string."""
        validate_request('paper_qa_search', {'use_topic': True, 'query': 'x', 'year': [2025, 2026],
                                             'author_id': ['a1'], 'org_id': ['o1', 'o2']})
        validate_request('paper_keywords', {'page': 0, 'size': 10, 'keywords': '["x"]'})
        for params in ({'use_topic': True, 'query': 'x', 'year': 2026},
                       {'use_topic': True, 'query': 'x', 'year': [-1]},
                       {'use_topic': True, 'query': 'x', 'author_id': 'a1'},
                       {'use_topic': True, 'query': 'x', 'org_id': ['']}):
            with self.assertRaises(ResearchAPIError, msg=params):
                validate_request('paper_qa_search', params)
        with self.assertRaises(ResearchAPIError):
            validate_request('paper_keywords', {'page': 0, 'size': 11, 'keywords': '["x"]'})

    def test_semantic_cursor_is_preserved_and_next_request_is_cursor_only(self):
        self.session.request.return_value=response({'papers':[{'paper_id':'p'}],'next_cursor':'a'*20})
        client=self.client(allow_paid=True)
        result=client.query('paper_qa_search_pro',{'query':'Which methods improve ductile fracture calibration?'})
        self.assertEqual(result['next_cursor'],'a'*20)
        client.query('paper_qa_search_pro',{'cursor':result['next_cursor']})
        self.assertEqual(self.session.request.call_args.kwargs['json'],{'cursor':'a'*20})
        self.assertEqual(client.summary()['estimated_reserved_yuan'],.6)

    def test_experiment_nested_envelopes_preserve_multiple_methods(self):
        row={'paper_id':'p','experiment_name':'Experiment','methods':[{'name':'A'},{'name':'B'}]}
        self.session.request.return_value=response({'results':{'records':[row]}})
        result=self.client(allow_paid=True).query('experiment_search',{'search_text':'  strain rate  '})
        self.assertEqual(result['data'],[row])
        self.assertEqual(self.session.request.call_args.kwargs['json'],{'paper_id':'','method':'','dataset':'','search_text':'strain rate'})
        with self.assertRaises(ResearchAPIError): experiment_rows({'unexpected':[]})
        with self.assertRaises(ResearchAPIError): experiment_rows([{'garbage':True}])

    def test_rate_limit_persists_and_auth_failure_is_sanitized(self):
        self.session.request.return_value=response(None,success=False,code=40306,msg='secret-token')
        client=self.client(allow_paid=True)
        with self.assertRaisesRegex(ResearchAPIError,'^rate_limited$'): client.query('paper_detail',{'id':'p'})
        with self.assertRaisesRegex(ResearchAPIError,'rate_limit_cooldown'): client.query('paper_detail',{'id':'other'})
        self.assertEqual(self.session.request.call_count,1)

    def test_budget_range_rejects_nonfinite_or_fractional_fen(self):
        for amount in ('NaN','Infinity','0','5.01','0.001','-1'):
            with self.assertRaises(ValueError): self.client(budget_yuan=amount)

    def test_every_allowed_endpoint_uses_its_declared_method_and_price(self):
        values={'id':'p','ids':['p'],'org_id':'i','venue_id':'v','org':'ETH Zurich',
                'offset':0,'page':0,'size':1,'year':2025,'keywords':'["fracture"]','use_topic':True}
        for name,spec in ENDPOINTS.items():
            with self.subTest(endpoint=name):
                params={key:values[key] for key in spec.required.split()}
                if name=='paper_search_pro': params['title']='ductile fracture'
                if name in {'paper_qa_search','paper_qa_search_pro'}: params['query']='How are fracture models calibrated?'
                if name=='experiment_search': params['paper_id']='p'
                session=Mock(); session.request.return_value=response([])
                client=ResearchClient(self.root/name,session=session,token='test',allow_paid=True,budget_yuan=5)
                self.assertEqual(client.query(name,params)['data'],[])
                self.assertEqual(session.request.call_args.args[0],spec.method)
                self.assertTrue(session.request.call_args.args[1].endswith(spec.path))
                self.assertEqual(client.summary()['estimated_reserved_yuan'],spec.fen/100)

    def test_missing_or_malformed_ledger_cannot_reset_spending(self):
        self.session.request.return_value=response([])
        client=self.client(allow_paid=True)
        client.query('paper_detail',{'id':'p'})
        ledger=self.root/'spending.json'
        ledger.unlink()
        with self.assertRaisesRegex(ResearchAPIError,'missing_budget_ledger'): client.query('paper_detail',{'id':'new'})
        ledger.write_text('[]')
        with self.assertRaises(ResearchAPIError): client.query('paper_detail',{'id':'new'})
        self.assertEqual(self.session.request.call_count,1)


if __name__=='__main__': unittest.main()
