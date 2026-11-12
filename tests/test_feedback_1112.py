"""November owner feedback: replay archived evidence, not just keywords in issue titles.

Fixtures originate in committed Oct 29 / Nov 5 / Nov 12 reports. This is a
regression set (not an independent precision/recall benchmark). No live APIs.
"""
import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

import numpy as np

from src.author_watch import author_news
from src.models import CandidateWork, RankedWork
from src.research_features import FeedbackEntry, FeedbackModel, feedback_delivery_candidates, load_feedback
from src.research_evidence import attach_evidence
from src.report_html import render_html
from src.score_rank import WorkRanker
from src.settings import load_settings
from src.topic_matching import research_priority, matching_priorities

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / 'tests/fixtures/feedback_20261112.json').read_text('utf-8'))


def candidate(case):
    return CandidateWork(source='openalex', identifier=case['doi'], doi=case['doi'],
                         title=case['title'], abstract=case['abstract'], published=datetime.now(timezone.utc))


def ranked(work, **updates):
    values = dict(**work.model_dump(), score=0.99, similarity=0.99, recency_score=1,
                  metric_score=0, author_bonus=0, venue_bonus=0, label='must_read')
    values.update(updates)
    return RankedWork(**values)


class NovemberFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(ROOT)

    def test_all_twelve_archived_papers_now_have_specific_demotions(self):
        self.assertEqual(len(CASES), 12)
        for row in CASES:
            with self.subTest(issue=row['issue']):
                name, factor = research_priority(candidate(row), self.settings.scoring)
                self.assertLessEqual(factor, 0.55)
                self.assertNotEqual(name, 'TC4核心研究')

    def test_demotions_run_through_actual_ranker_not_only_the_policy_helper(self):
        ranker = object.__new__(WorkRanker)
        ranker.settings = self.settings; ranker.profile = {}; ranker.journal_metrics = {}
        ranker.vectorizer = SimpleNamespace(text_separator='\n', encode=lambda texts: np.zeros((len(texts), 2)))
        ranker.index = SimpleNamespace(ntotal=5, search=lambda vectors, top_k: (np.full((len(vectors),top_k),0.95), None))
        result = ranker.rank([candidate(row) for row in CASES])
        self.assertTrue(all(w.label == 'ignore' for w in result))

    def test_each_owner_rejection_excludes_even_high_score_and_author_lane(self):
        model = FeedbackModel([FeedbackEntry(doi=r['doi'], rating='irrelevant', facets=r['facets']) for r in CASES], self.settings.research)
        works = [ranked(candidate(r), extra={'watched_authors':[{'name':'Fixture'}], 'referenced_by':[{'id':'W1'}],
                                            'semantic_facets':['calibration'], 'primary_problem':'unrelated_primary'}) for r in CASES]
        tagged = model.apply(works, self.settings.scoring.thresholds)
        self.assertTrue(all(w.extra['feedback_suppressed'] for w in tagged))
        self.assertTrue(all(w.label == 'ignore' for w in tagged))
        self.assertEqual(model.preferences, {})
        self.assertEqual(feedback_delivery_candidates(tagged), [])
        self.assertEqual(author_news(tagged, self.settings.author_watch), [])

    def test_issue_parser_accepts_owner_feedback_and_explicit_reset_restores(self):
        row=CASES[0]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data'; path.mkdir()
            def issue(owner, rating, updated):
                body='<!-- zotwatch-feedback-v1 -->\n```json\n'+json.dumps({'doi':row['doi'],'rating':rating,'facets':row['facets']})+'\n```'
                return {'number':row['issue'],'user':{'login':owner},'updated_at':updated,'body':body}
            (path/'feedback-issues.json').write_text(json.dumps([
                issue(self.settings.research.feedback_owner,'irrelevant','1'),
                issue('outsider','reset','2')]),encoding='utf-8')
            model=FeedbackModel(load_feedback(tmp,self.settings.research),self.settings.research)
            tagged=model.apply([ranked(candidate(row))],self.settings.scoring.thresholds)[0]
            self.assertTrue(tagged.extra['feedback_suppressed'])
            (path/'feedback-issues.json').write_text(json.dumps([
                issue(self.settings.research.feedback_owner,'reset','3')]),encoding='utf-8')
            reset=FeedbackModel(load_feedback(tmp,self.settings.research,[e.model_dump() for e in model.entries.values()]),self.settings.research)
            restored=reset.apply([tagged],self.settings.scoring.thresholds)[0]
            self.assertFalse(restored.extra['feedback_suppressed'])
            self.assertTrue(feedback_delivery_candidates([restored]))

    def test_global_hold_is_source_independent_but_scoped_dislike_stays_scoped(self):
        work=ranked(candidate(CASES[0]),extra={'primary_problem':'calibration'})
        for source in ['openalex','crossref','aminer']:
            model=FeedbackModel([FeedbackEntry(doi=work.doi,rating='later',applicability='hold')],self.settings.research)
            self.assertFalse(feedback_delivery_candidates(model.apply([work.model_copy(update={'source':source})],self.settings.scoring.thresholds)))
        scoped=FeedbackModel([FeedbackEntry(doi=work.doi,rating='irrelevant',scope='strain_rate',facets=['strain_rate'])],self.settings.research)
        self.assertTrue(feedback_delivery_candidates(scoped.apply([work],self.settings.scoring.thresholds)))
        alert=work.model_copy(update={'identifier':'update:fixture','doi':None,'source':'publication_update','extra':{'original_doi':work.doi}})
        self.assertTrue(feedback_delivery_candidates(scoped.apply([alert],self.settings.scoring.thresholds)))

    def test_transfer_cards_obey_material_requirement_and_doi_links_normalized(self):
        work=ranked(CandidateWork(source='openalex',identifier='W123',doi='https://doi.org/10.1234/Test',
                     title='LPBF 316L steel plasticity and ductile fracture',abstract='Build orientation controls ductile fracture in 316L steel.'))
        evidence=attach_evidence(work,self.settings.research)
        self.assertFalse(any('TC4' in card['topic'] for card in evidence.extra['transfer_cards']))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.html'; render_html([evidence],path)
            html=path.read_text('utf-8')
            self.assertNotIn('https://doi.org/https://doi.org/',html)
            self.assertIn('href="https://doi.org/10.1234/test"',html)

    def test_relevant_metal_mechanics_are_not_lost_to_new_rules(self):
        titles=[
            'Ductile fracture of LPBF Ti-6Al-4V under combined stress states',
            'Ductile fracture of coated Ti-6Al-4V under dynamic loading',
            'Steel LS-OPT constitutive model calibration using full-field strain measurements',
            'Inverse identification of post-necking hardening in aluminum alloys',
            'Damage regularization for mesh objective ductile fracture in steel',
            'Temperature dependent constitutive model of AZ31 magnesium alloy',
            'Processing map and inverse identification of metal plasticity parameters',
            'A fracture model for scratch-induced damage in steel',
            'A constitutive model for particle-reinforced aluminum matrix composites',
            'Microstructural insights into a stress-state-dependent ductile fracture criterion for steel',
            'Dynamic tensile plasticity of titanium alloys with SEM fractography',
        ]
        new_names={'工艺或组织设计主导','表面划痕与聚合物打印','增强复合材料工艺与靶板'}
        for title in titles:
            with self.subTest(title=title):
                work=CandidateWork(source='fixture',identifier=title,title=title)
                self.assertFalse(new_names.intersection(h['name'] for h in matching_priorities(work,self.settings.scoring)))
        for title in titles[:6]:
            self.assertEqual(research_priority(CandidateWork(source='fixture',identifier=title,title=title),self.settings.scoring)[1],1.0)

    def test_incidental_abstract_topic_cannot_demote_a_method_paper(self):
        work=CandidateWork(source='fixture',identifier='guard',title='Steel ductile fracture criterion calibration',
                           abstract='Processing maps, particle-reinforced composites and scratch-induced damage are discussed as possible applications.')
        self.assertEqual(research_priority(work,self.settings.scoring)[1],1.0)

    def test_pipeline_rejection_cannot_reenter_via_any_channel_or_delivery_history(self):
        import src.cli as cli
        from src.watch_history import WatchHistory, history_keys
        settings=self.settings.model_copy(deep=True)
        settings.translation.enabled=False
        settings.sources.aminer.enabled=False
        works=[candidate(row) for row in CASES[:3]]
        works[0].extra={'watched_authors':[{'name':'Fixture'}]}
        works[1].published=datetime.now(timezone.utc)-timedelta(days=365)
        works[1].extra={'referenced_by':[{'id':'W1','title':'Seed','url':'https://openalex.org/W1'}]}
        works[2].published=works[1].published
        works[2].extra={'semantic_facets':['other_metals']}
        good=CandidateWork(source='fixture',identifier='good',doi='10.1234/keep',
                           title='Steel inverse identification and ductile fracture calibration',published=datetime.now(timezone.utc))
        alert=ranked(good,identifier='publication_update:1',doi=None,title='Correction alert',
                     extra={'original_doi':works[0].doi},source='publication_update')
        with tempfile.TemporaryDirectory() as tmp, patch.object(cli,'ZoteroIngestor'), \
             patch.object(cli,'ProfileBuilder'), patch.object(cli,'CandidateFetcher') as fetch, \
             patch.object(cli,'WorkRanker') as ranker, patch.object(cli,'DedupeEngine') as dedupe, \
             patch.object(cli,'CitationDiscovery') as graph, patch.object(cli,'VersionMonitor') as monitor, \
             patch.object(cli,'enrich_ranked_works',side_effect=lambda works,*args,**kwargs: works):
            root=Path(tmp);(root/'config').mkdir()
            import yaml
            (root/'config/feedback.yaml').write_text(yaml.safe_dump({'entries':[
                {'doi':w.doi,'rating':'irrelevant'} for w in works]}),encoding='utf-8')
            fetch.return_value.fetch_all.return_value=[works[0],good]
            fetch.return_value._filter_by_topic.side_effect=lambda works:works
            dedupe.return_value.filter.side_effect=lambda works:list(works)
            ranker.return_value.rank.side_effect=lambda works:[ranked(w) for w in works]
            ranker.return_value.profile={'item_count':1,'problem_profiles':{}}
            graph.return_value.fetch.return_value=works[1:]
            graph.return_value.annotate.side_effect=lambda works:works
            graph.return_value.warnings=[]
            monitor.return_value.check.return_value=[alert];monitor.return_value.warnings=[]
            cli.run_watch(root,settings,Mock(),rss=True,report=True,top=20,push=False,defer_history=True)
            html=next((root/'reports').glob('report-*.html')).read_text('utf-8')
            feed=(root/'reports/feed.xml').read_text('utf-8')
            self.assertIn(good.title,html);self.assertIn('Correction alert',feed)
            for w in works:
                self.assertNotIn(w.title,html);self.assertNotIn(w.title,feed)
            pending=json.loads((root/'data/watch-state/pending.json').read_text('utf-8'))
            for w in works:
                self.assertFalse(history_keys(w).intersection(pending['sent']))
            self.assertFalse((root/'data/watch-state/state.json').exists())


if __name__ == '__main__': unittest.main()
