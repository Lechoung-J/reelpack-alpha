import copy
from contextlib import nullcontext
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
from audio_analyze import analyze
from audio_service import analyze_audio, ensure_audio_board, _render_in_audio_environment
from optional_backend_worker import panns
from optional_backends import analyze_music
from test_audio_board import measured_fixture


class AudioServiceTests(unittest.TestCase):
    def test_long_step_heartbeat_stops_when_work_finishes(self):
        import threading
        from audio_service import _waiting
        received = []
        event = threading.Event()
        def progress(stage, percent, message):
            received.append((stage, percent, message)); event.set()
        with _waiting(progress, 'audio_tagging', 44, 'local work', interval=.01):
            self.assertTrue(event.wait(.5))
        count = len(received)
        self.assertGreater(count, 0)
        self.assertIn('已等待', received[0][2])
        self.assertIn('尚未返回结果', received[0][2])
        self.assertEqual(received[0][:2], ('audio_tagging', 44))


    def test_missing_model_cost_notice_precedes_any_physical_work(self):
        events = []
        def physical(*args):
            events.append('physical'); return measured_fixture()
        def progress(stage, percent, message):
            if '327 MB' in message: events.append('cost_notice')
        with patch('audio_service.Path.is_file', return_value=False), \
             patch('audio_service._physical_analysis', side_effect=physical), \
             patch('optional_backends.analyze_music', side_effect=lambda media, music, **kw: music):
            result = analyze_audio('not-read.mp4', [], None, backend='panns', progress=progress)
        self.assertEqual(events[:2], ['cost_notice', 'physical'])
        self.assertEqual(result['tempo_bpm'], 120)

    def run_fixture(self, music, *, package=None, backend='librosa', progress=None):
        def runner(command, **kwargs):
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(kwargs['timeout'], 900)
            self.assertEqual(kwargs['env']['MPLCONFIGDIR'], '/synthetic/runtime-cache/mpl')
            return subprocess.CompletedProcess(command, 0, json.dumps(music), '')
        environment = {'REELPACK_AUDIO_PYTHON': sys.executable, 'MPLCONFIGDIR': '/synthetic/runtime-cache/mpl'}
        with patch('audio_service._runtime_environment', return_value=environment), patch('audio_service.subprocess.run', side_effect=runner):
            return analyze_audio('synthetic media.mp4', [{'start': 0}, {'start': .51}, {'start': 1.1}, {'start': 1.8}],
                                 package, backend=backend, progress=progress)

    def test_shared_service_board_metrics_and_progress(self):
        fixture = measured_fixture()
        before = copy.deepcopy(fixture)
        progress = []
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_fixture(fixture, package=directory, progress=lambda *args: progress.append(args))
            self.assertEqual(result['audio_board']['status'], 'complete', result['audio_board'])
            self.assertEqual(result['audio_board']['metrics']['aligned_cut_count'], 3)
            self.assertEqual(result['cut_alignment']['duration_seconds'], 2.)
            self.assertTrue(result['cut_alignment']['includes_media_start'])
        self.assertEqual(fixture, before)
        self.assertEqual(result['tempo_bpm'], 120.)
        self.assertEqual(result['beat_times_seconds'], before['beat_times_seconds'])
        self.assertEqual([x[0] for x in progress], ['audio_decode', 'audio_physical', 'audio_tagging', 'audio_board', 'audio_complete'])
        self.assertEqual([x[1] for x in progress], sorted(x[1] for x in progress))

    def test_optional_no_package_and_empty_beats(self):
        result = self.run_fixture(measured_fixture(beats=()))
        self.assertEqual(result['audio_board']['status'], 'not_run')
        self.assertEqual(result['audio_board']['reason'], 'no package target')
        self.assertEqual(result['cut_alignment']['cut_count'], 4)
        self.assertEqual(result['cut_alignment']['status'], 'not_measured')
        self.assertIsNone(result['cut_alignment']['aligned_ratio'])
        self.assertIsNone(result['cut_alignment']['aligned_cut_count'])

    def test_missing_worker_and_invalid_response_fail_without_throw(self):
        with patch('audio_service._runtime_environment', return_value={'REELPACK_AUDIO_PYTHON': '/no/local/python'}), patch('audio_service.subprocess.run') as worker:
            result = analyze_audio('x', [{'start': 0}], None)
            worker.assert_not_called()
            self.assertEqual(result['status'], 'not_run')
            self.assertTrue(result['reason'])
            self.assertTrue(result['impact'])
            self.assertTrue(result['remedy'])
            self.assertEqual(result['timbre']['status'], 'not_run')
            self.assertIn('audio python unavailable', result['timbre']['reason'])
            self.assertEqual(result['genre']['status'], 'not_run')
            self.assertTrue(result['genre']['remedy'])
            self.assertIsNone(result['cut_alignment']['aligned_ratio'])
        for output in ('', '[]', '{bad', '{"status":"invalid"}'):
            with self.subTest(output=output), patch('audio_service._runtime_environment', return_value={'REELPACK_AUDIO_PYTHON': sys.executable}), patch('audio_service.subprocess.run', return_value=subprocess.CompletedProcess('x', 0, output, '')):
                result = analyze_audio('x', [], None)
                self.assertEqual(result['status'], 'failed')
                self.assertIsNone(result['cut_alignment']['aligned_cut_count'])
        fixture = measured_fixture()
        fixture['beat_times_seconds'] = 5
        result = self.run_fixture(fixture)
        self.assertEqual(result['cut_alignment']['status'], 'not_measured')
        self.assertEqual(result['status'], 'partial')

    def test_physical_worker_timeout_is_failed_and_null_alignment(self):
        with patch('audio_service._runtime_environment', return_value={'REELPACK_AUDIO_PYTHON': sys.executable}), patch('audio_service.subprocess.run', side_effect=subprocess.TimeoutExpired('audio worker', 900)):
            result = analyze_audio('x', [{'start': 0}], None)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('TimeoutExpired', result['reason'])
        self.assertEqual(result['timbre']['status'], 'failed')
        self.assertIn('TimeoutExpired', result['timbre']['reason'])
        self.assertIsNone(result['cut_alignment']['aligned_cut_count'])

    def test_board_failure_does_not_drop_tempo_or_beats(self):
        with tempfile.TemporaryDirectory() as directory, patch('audio_service.render_from_json', side_effect=RuntimeError('mock drawing failure')):
            result = self.run_fixture(measured_fixture(), package=directory)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['tempo_bpm'], 120.)
        self.assertEqual(result['beat_count'], 4)
        self.assertEqual(result['audio_board']['status'], 'not_run')
        self.assertIn('mock drawing failure', result['audio_board']['reason'])

    def test_previous_board_is_not_current_evidence_after_audio_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = ensure_audio_board(measured_fixture(), directory)
            image = Path(directory) / valid['path']
            before = image.read_bytes()
            with patch('audio_service.render_from_json') as renderer:
                result = self.run_fixture({'status': 'failed', 'error': 'synthetic decode failure'}, package=directory)
                renderer.assert_not_called()
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['audio_board']['status'], 'not_run')
            self.assertIn('previous metrics are not evidence', result['audio_board']['reason'])
            self.assertIsNone(result['cut_alignment']['aligned_ratio'])
            self.assertEqual(image.read_bytes(), before)

    def test_standalone_states_have_reason_impact_remedy(self):
        fixture = measured_fixture()
        fixture['timbre'] = {'status': 'failed', 'reason': 'synthetic spectral failure'}
        fixture['genre'] = {'status': 'not_run'}
        fixture['mood'] = {'status': 'not_run'}
        result = self.run_fixture(fixture)
        for name in ('genre', 'mood', 'timbre', 'audio_board'):
            for field in ('reason', 'impact', 'remedy'):
                self.assertTrue(result[name][field])

    def test_panns_missing_preserves_physical_and_exposes_cost(self):
        def runner(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, json.dumps({'status': 'not_run', 'reason': 'missing explicit weights'}), '')
        with patch('optional_backends._runtime_environment', return_value={'REELPACK_MUSIC_PYTHON': sys.executable}):
            result = analyze_music('x', measured_fixture(), runner=runner)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['tempo_bpm'], 120.)
        self.assertEqual(result['tagging']['status'], 'not_run')
        self.assertEqual(result['tagging']['cost']['approximate_checkpoint_mb'], 327)
        self.assertFalse(result['tagging']['cost']['automatic_download'])
        for key in ('genre', 'mood', 'events'):
            self.assertTrue(result[key]['reason'])
            self.assertTrue(result[key]['impact'])
            self.assertTrue(result[key]['remedy'])

    def test_safe_audio_render_fallback_timeout_and_environment(self):
        captured = []
        def runner(command, **kwargs):
            captured.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, json.dumps({'status': 'complete'}), '')
        environment = {'REELPACK_AUDIO_PYTHON': sys.executable, 'MPLCONFIGDIR': '/synthetic/cache/mpl'}
        with patch('audio_service._runtime_environment', return_value=environment), patch('audio_service.subprocess.run', side_effect=runner):
            self.assertEqual(_render_in_audio_environment(Path('/with spaces ;/metrics.json'), Path('/with spaces ;/board.png'))['status'], 'complete')
        command, kwargs = captured[0]
        self.assertIn('/with spaces ;/metrics.json', command)
        self.assertIn('/with spaces ;/board.png', command)
        self.assertNotIn('shell', kwargs)
        self.assertEqual(kwargs['timeout'], 60)
        self.assertEqual(kwargs['env'], environment)
        with tempfile.TemporaryDirectory() as directory, patch('audio_service.render_from_json', side_effect=ModuleNotFoundError('Pillow')), patch('audio_service._render_in_audio_environment', side_effect=subprocess.TimeoutExpired('render', 60)):
            result = ensure_audio_board(measured_fixture(), directory)
            self.assertEqual(result['status'], 'not_run')
            self.assertIn('TimeoutExpired', result['reason'])

    def test_source_hash_and_single_decode(self):
        y = np.zeros(100)
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / 'synthetic media.wav'
            media.write_bytes(b'fixture source bytes')
            with patch('audio_analyze.extract_audio', return_value=(y, 8000)) as decoder, patch('audio_analyze.analyze_array', return_value={'status': 'complete'}) as array:
                self.assertEqual(analyze(media)['status'], 'complete')
                decoder.assert_called_once_with(media)
                self.assertEqual(array.call_args.kwargs['source_hash'], hashlib.sha256(media.read_bytes()).hexdigest())

    def test_explicit_labels_override_home_and_no_download(self):
        labels = ['Speech', 'Music', 'Electronic music'] + [f'Label {i}' for i in range(524)]
        scores = np.zeros((1, 527)); scores[0, 2] = .8
        test = self
        class Model:
            def __init__(self, **kwargs):
                config = sys.modules['panns_inference.config']
                test.assertEqual(config.labels, labels)
                with test.assertRaisesRegex(RuntimeError, 'automatic download disabled'):
                    os.system('wget must-never-run')
            def inference(self, audio):
                return scores, None
        fake_torch = SimpleNamespace(set_num_threads=lambda count: None, no_grad=nullcontext)
        fake_panns = SimpleNamespace(AudioTagging=Model)
        with tempfile.TemporaryDirectory() as directory:
            labels_path = Path(directory) / 'custom labels.csv'
            with labels_path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['index', 'mid', 'display_name'])
                writer.writerows((i, f'/m/id{i}', label) for i, label in enumerate(labels))
            with patch.dict(os.environ, {'REELPACK_PANNS_LABELS': str(labels_path), 'REELPACK_PANNS_CHECKPOINT': '/synthetic/explicit.pth'}), patch('optional_backend_worker.Path.home', return_value=Path('/never-read-home')), patch('optional_backend_worker.Path.is_file', return_value=True), patch('optional_backend_worker.Path.stat', return_value=SimpleNamespace(st_size=327000000)), patch.dict(sys.modules, {'torch': fake_torch, 'panns_inference': fake_panns}), patch('audio_analyze.extract_audio', return_value=(np.zeros(100), 32000)), patch('optional_backend_worker._music_timeline', return_value={'status': 'not_run', 'method': None}), patch('optional_backend_worker.os.system') as download:
                result = panns('synthetic local media.mp4')
                download.assert_not_called()
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['genre']['items'][0]['label'], 'Electronic music')
            self.assertEqual(result['params']['labels_path'], str(labels_path))
            self.assertFalse(result['params']['automatic_download'])


if __name__ == '__main__':
    unittest.main()
