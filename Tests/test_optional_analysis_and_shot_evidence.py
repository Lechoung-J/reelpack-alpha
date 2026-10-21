import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
from optional_backend_worker import classify
from asset_cli import shot_evidence_view
from shot_detect import detect_shots

class OptionalAnalysisAndShotEvidenceTests(unittest.TestCase):
    def test_genre_low_score_and_noise(self):
        r = classify(['Electronic music', 'Rock music', 'Speech', 'Noise'], [.07, .01, .9, .8])
        self.assertEqual([x['label'] for x in r['genre']['items']], ['Electronic music'])
        self.assertEqual(classify(['Electronic music'], [.07], genre_threshold=.1)['genre']['items'], [])
        self.assertTrue(r['top_tags'])
        self.assertTrue(r['description'])
        with self.assertRaises(ValueError):
            classify([], [], genre_threshold=-1)

    def test_sidecar_default_and_failure(self):
        with patch('shot_detect.detect_shots_isolated', return_value=({'status':'complete', 'backend':'transnetv2-pytorch-cpu'}, [])) as worker:
            r, _ = detect_shots('input', 'output', fallback=False)
            self.assertTrue(worker.call_args.args[0].endswith('.venv-transnet/bin/python'))
            self.assertEqual(r['requested_backend'], 'transnetv2')
            self.assertIsNone(r['fallback'])

    def test_evidence_anchors_no_media(self):
        frames = [dict(evidence_id=str(i), kind='image', timestamp=t, relative_path=f'evidence/{i}.jpg') for i,t in enumerate([0, .9, 1.9])]
        r = shot_evidence_view(dict(shots=dict(items=[dict(shot_id='shot_1',start=0,end=2,evidence_ids=['0','1','2'])]),evidence=frames))
        self.assertEqual(r['items'][0]['anchors']['middle']['timestamp'], .9)
        self.assertEqual(r['items'][0]['anchors']['tail']['relative_path'], 'evidence/2.jpg')
        self.assertEqual(r['items'][0]['content_description']['status'], 'not_run')
