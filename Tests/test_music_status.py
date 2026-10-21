"""Physical audio analysis and optional tags retain separate, truthful statuses."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
from optional_backends import analyze_music
from optional_backend_worker import classify, panns


class MusicStatusTests(unittest.TestCase):
    def descriptors(self, status='complete'):
        result = {
            'status': status, 'backend': 'librosa', 'model': 'physical-fixture',
            'params': {'sample_rate': 22050}, 'experimental': True,
            'tempo_bpm': 120, 'beat_times_seconds': [1.0, 1.5],
            'loudness': {'integrated_lufs': -14},
            'timbre': {'status': 'complete', 'backend': 'librosa_spectral'},
            'genre': {'status': 'not_run'}, 'mood': {'status': 'not_run'},
        }
        if status != 'complete':
            result['error'] = 'synthetic physical analysis failure'
        return result

    def run_semantic(self, descriptors, semantic):
        before = copy.deepcopy(descriptors)
        def runner(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, json.dumps(semantic), '')
        with patch('optional_backends.Path.is_file', return_value=True):
            result = analyze_music('synthetic.mp4', descriptors, runner)
        self.assertEqual(descriptors, before)
        self.assertEqual(result['descriptors'], {key: value for key, value in before.items()
                                                 if key not in ('genre', 'mood')})
        for key in ('status', 'backend', 'model', 'params', 'tempo_bpm',
                    'beat_times_seconds', 'loudness', 'timbre', 'experimental'):
            self.assertEqual(result[key], before[key])
        return result

    def complete_tags(self):
        return dict(status='complete', backend='panns', model='Cnn14',
                    params={'scope': 'whole_mix', 'scores_calibrated': False},
                    experimental=True, **classify(
                        ['Electronic music', 'Happy music', 'Speech'], [.8, .7, .6]))

    def test_tagging_success_preserves_physical_provenance(self):
        semantic = self.complete_tags()
        result = self.run_semantic(self.descriptors(), semantic)
        self.assertEqual(result['tagging']['status'], 'complete')
        self.assertEqual(result['tagging']['backend'], 'panns')
        self.assertEqual(result['tagging']['model'], 'Cnn14')
        self.assertEqual(result['tagging']['params'], semantic['params'])
        self.assertEqual(result['genre']['items'][0]['label'], 'Electronic music')
        self.assertEqual(result['mood']['items'][0]['label'], 'Happy music')
        self.assertEqual(result['events']['items'][0]['label'], 'Speech')

    def test_successful_tags_do_not_turn_failed_physical_analysis_complete(self):
        result = self.run_semantic(self.descriptors('failed'), self.complete_tags())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error'], 'synthetic physical analysis failure')
        self.assertEqual(result['tagging']['status'], 'complete')
        self.assertTrue(result['genre']['items'])

    def test_tagging_failure_or_unavailability_never_relabels_complete_descriptors(self):
        for status in ('failed', 'not_run', 'partial'):
            with self.subTest(tagging_status=status):
                semantic = {'status': status, 'reason': 'synthetic optional backend unavailable'}
                result = self.run_semantic(self.descriptors(), semantic)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['tagging']['status'], status)
                self.assertEqual(result['tagging']['reason'], semantic['reason'])
                self.assertNotIn('reason', result)
                for field in ('genre', 'mood', 'events'):
                    self.assertEqual(result[field]['status'], status)
                    self.assertEqual(result[field]['items'], [])
                    self.assertEqual(result[field]['reason'], semantic['reason'])

    def test_missing_environment_is_not_run_and_never_spawns_worker(self):
        with patch('optional_backends.Path.is_file', return_value=False), \
             patch('optional_backends.subprocess.run') as worker:
            result = analyze_music('synthetic.mp4', self.descriptors())
        worker.assert_not_called()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['tagging']['status'], 'not_run')
        self.assertIn('optional python unavailable', result['tagging']['reason'])
        self.assertEqual(result['tempo_bpm'], 120)

    def test_worker_failure_is_independent_of_physical_analysis(self):
        def failed_worker(*args, **kwargs):
            raise subprocess.CalledProcessError(1, 'synthetic-worker', stderr='synthetic broken checkpoint')
        with patch('optional_backends.Path.is_file', return_value=True):
            result = analyze_music('synthetic.mp4', self.descriptors(), failed_worker)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['tagging']['status'], 'failed')
        self.assertIn('synthetic broken checkpoint', result['tagging']['reason'])
        self.assertEqual(result['timbre']['status'], 'complete')
        self.assertEqual(result['genre']['status'], 'failed')

    def test_missing_or_incomplete_weights_stop_before_import_decode_or_download(self):
        cases = (([False], 0), ([True], 1), ([True, False], 300000000))
        for files, size in cases:
            with self.subTest(files=files, checkpoint_bytes=size), \
                 patch('optional_backend_worker.Path.home', return_value=Path('/synthetic/unread-home')), \
                 patch.dict('os.environ', {'REELPACK_PANNS_CHECKPOINT': '/synthetic/checkpoint.pth'}), \
                 patch('optional_backend_worker.Path.is_file', side_effect=files), \
                 patch('optional_backend_worker.Path.stat', return_value=SimpleNamespace(st_size=size)), \
                 patch('optional_backend_worker.Path.open') as opened, \
                 patch('builtins.__import__', wraps=__import__) as imports:
                result = panns('synthetic.mp4')
            opened.assert_not_called()
            self.assertFalse(any(call.args[0] in ('torch', 'panns_inference', 'audio_analyze')
                                 for call in imports.call_args_list))
            self.assertEqual(result['status'], 'not_run')
            self.assertIn('no download attempted', result['reason'])
            self.assertEqual(result['timeline']['status'], 'not_run')
            self.assertEqual(result['timeline']['items'], [])
            self.assertIsNone(result['trend'])
            adapted = self.run_semantic(self.descriptors(), result)
            self.assertEqual(adapted['status'], 'complete')
            self.assertEqual(adapted['tagging']['status'], 'not_run')
            self.assertEqual(adapted['genre']['status'], 'not_run')


if __name__ == '__main__':
    unittest.main()
