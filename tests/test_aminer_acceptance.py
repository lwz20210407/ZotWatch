import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from src.aminer_acceptance import evaluate_snapshot, profile_fingerprint
from src.candidate_vectors import CandidateVectors
from src.models import CandidateWork, RankedWork
from src.settings import load_settings
from src.storage import ProfileStorage

ROOT = Path(__file__).resolve().parents[1]


class AcceptanceTests(unittest.TestCase):
    def test_cache_only_partial_then_resume_without_modifying_inputs(self):
        settings = load_settings(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'repo'; output = Path(tmp) / 'result'; output.mkdir()
            storage = ProfileStorage(base / 'data/profile.sqlite'); storage.initialize(); storage.close()
            paper = CandidateWork(source='aminer', identifier='aminer:p', title='Steel ductile fracture', abstract='Steel ductile damage evolution under dynamic loading.', extra={'aminer_id':'p', 'publication_year':2020})
            snapshot = Path(tmp) / 'snapshot.json'; snapshot.write_text(json.dumps([paper.model_dump(mode='json')]), 'utf-8')
            encoder = Mock(text_separator='\n'); encoder.encode.return_value = np.asarray([[1, 1]], dtype=np.float32)
            cache = CandidateVectors(encoder, output / 'vector-cache', 'fixture', 2)
            def rank(rows):
                cache.encode([w.content_for_embedding('\n') for w in rows])
                return [RankedWork(**w.model_dump(), score=0.6, similarity=0.8, label='consider', recency_score=0, metric_score=0, author_bonus=0, venue_bonus=0) for w in rows]
            ranker = Mock(vectorizer=cache); ranker.rank.side_effect = rank
            before = profile_fingerprint(base)
            with patch('src.aminer_acceptance.load_settings', return_value=settings), patch('src.aminer_acceptance.WorkRanker', return_value=ranker):
                first = evaluate_snapshot(base, snapshot, 'aminer', output, 0, 60, True)
                self.assertEqual(first['status'], 'partial'); self.assertEqual(first['scored'], 0)
                encoder.encode.assert_not_called()
                second = evaluate_snapshot(base, snapshot, 'aminer', output, 1, 60)
                self.assertEqual(second['status'], 'complete'); self.assertEqual(second['scored'], 1)
                self.assertEqual(encoder.encode.call_count, 1)
                third = evaluate_snapshot(base, snapshot, 'aminer', output, 0, 60, True)
                self.assertEqual(third['scored'], 1); self.assertEqual(encoder.encode.call_count, 1)
            self.assertEqual(before, profile_fingerprint(base))


if __name__ == '__main__': unittest.main()
