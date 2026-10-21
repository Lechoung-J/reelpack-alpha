import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
from optional_backends import invoke, analyze_music, recognize_frames
from optional_backend_worker import classify, rapidocr


class OptionalBackendTests(unittest.TestCase):
    def runner(self, payload):
        return lambda *a, **k: subprocess.CompletedProcess(a[0], 0, json.dumps(payload), '')

    def test_missing_environment(self):
        with patch.dict('os.environ', REELPACK_MUSIC_PYTHON='/nonexistent/python'):
            self.assertEqual(invoke('panns', ['x'])['status'], 'not_run')

    def test_process_errors(self):
        with patch('optional_backends.Path.is_file', return_value=True):
            for error in (subprocess.TimeoutExpired('x', 1), subprocess.CalledProcessError(1, 'x'), ValueError('bad')):
                with self.subTest(error=error):
                    def runner(*a, **k):
                        raise error
                    self.assertEqual(invoke('panns', ['x'], runner)['status'], 'failed')
            self.assertEqual(invoke('panns', ['x'], self.runner([]))['status'], 'failed')

    def test_preserve_descriptors_and_classification(self):
        semantic = dict(status='complete', **classify(['Electronic music', 'Angry music', 'Speech', 'Music'], [.7, .5, .8, .9]))
        with patch('optional_backends.Path.is_file', return_value=True):
            result = analyze_music('x', {'status': 'complete', 'tempo_bpm': 120, 'timbre': {'status': 'complete'}}, self.runner(semantic))
        self.assertEqual(result['tempo_bpm'], 120)
        self.assertEqual(result['genre']['items'][0]['label'], 'Electronic music')
        self.assertEqual(result['events']['items'][0]['label'], 'Speech')
        self.assertEqual(len(result['events']['items']), 1)
        self.assertIn('AudioSet', result['description'])

    def test_failure_preserves_physical_results(self):
        with patch('optional_backends.Path.is_file', return_value=False):
            result = analyze_music('x', {'tempo_bpm': 100, 'genre': {'status': 'complete'}})
        self.assertEqual(result['genre']['status'], 'not_run')
        self.assertEqual(result['tempo_bpm'], 100)

    def test_manifest_and_empty_ocr(self):
        def runner(command, **kwargs):
            frames = json.loads(Path(command[-1]).read_text())
            self.assertEqual(frames[0]['evidence_ids'], ['frame1'])
            self.assertEqual(Path(frames[0]['path']).read_bytes(), b'image')
            return self.runner(dict(status='complete', items=[]))(command)
        with patch('optional_backends.Path.is_file', return_value=True):
            self.assertEqual(recognize_frames([(1.5, b'image')], {1.5: ['frame1']}, runner)['status'], 'complete')

    def test_ocr_boxes(self):
        import tempfile
        import types
        fake = types.SimpleNamespace(RapidOCR=lambda: lambda path: ([[[[1, 2], [5, 2], [5, 7], [1, 7]], '字幕', .9]], None))
        with tempfile.TemporaryDirectory() as td, patch.dict(sys.modules, rapidocr_onnxruntime=fake):
            path = Path(td) / 'manifest.json'
            path.write_text(json.dumps([dict(path='x', timestamp=2, evidence_ids=['e'])]))
            item = rapidocr(path)['items'][0]
        self.assertEqual(item['region']['width'], 4)
        self.assertEqual(item['timestamp'], 2)
        self.assertEqual(item['evidence_ids'], ['e'])

    def test_unclassified_music_fallback_hint(self):
        semantic = dict(status='complete', **classify(['Music', 'Speech', 'Scary music'], [.75, .2, .04]))
        with patch('optional_backends.Path.is_file', return_value=True):
            result = analyze_music('x', {'status': 'complete'}, self.runner(semantic))
        self.assertTrue(result['unclassified'])
        self.assertIn('Music 0.75', result['fallback_hint'])
        self.assertIn('top_tags', result['fallback_hint'])

    def test_framewise_aggregation_by_shots(self):
        from optional_backend_worker import _aggregate_framewise
        import numpy as np
        labels = ['Electronic music', 'Angry music', 'Speech']
        fw = np.array([[.8, .1, .2], [.7, .2, .1], [.1, .8, .2], [.1, .7, .1]])
        result = _aggregate_framewise(labels, fw, 32000, [{'start': 0, 'end': 2}, {'start': 2, 'end': 4}], 4)
        self.assertEqual(len(result['items']), 2)
        self.assertEqual(result['items'][0]['genre'][0]['label'], 'Electronic music')
        self.assertEqual(result['items'][1]['mood'][0]['label'], 'Angry music')


class MusicTimelineRegressionTests(unittest.TestCase):
    def test_novelty_peaks_respect_minimum_distance(self):
        import numpy as np
        from optional_backend_worker import _novelty_peaks
        features = np.zeros((8, 2)); features[3:] = 4; features[4] = 8
        peaks = _novelty_peaks(features, np.arange(8, dtype=float), threshold=.1, min_distance=2)
        self.assertGreaterEqual(len(peaks), 1)
        self.assertTrue(all(b[0] - a[0] >= 2 for a, b in zip(peaks, peaks[1:])))

    def test_two_layer_energy_is_audio_rms(self):
        import numpy as np
        from optional_backend_worker import _build_two_layer_timeline
        audio = np.ones(100) * .5
        scores = np.tile([[.8, .1]], (4, 1))
        result = _build_two_layer_timeline(audio, 10, ['Electronic music', 'Angry music'], scores, 10, [], bucket_seconds=5)
        self.assertEqual(len(result['overview']['items']), 2)
        self.assertAlmostEqual(result['overview']['items'][0]['energy'], .5)
        self.assertEqual(result['overview']['items'][0]['energy_type'], 'audio_rms')

    def test_sed_chunks_and_padded_tail(self):
        import numpy as np
        import types
        from optional_backend_worker import _music_timeline
        calls = []
        class SED:
            model = types.SimpleNamespace(sample_rate=10)
            def __init__(self, **kwargs):
                pass
            def inference(self, audio):
                calls.append(audio.shape[1])
                return np.array([[[.8, .1]] * audio.shape[1]])
        checkpoint = unittest.mock.Mock()
        checkpoint.is_file.return_value = True
        checkpoint.stat.return_value.st_size = 300000000
        with patch.dict(sys.modules, {'panns_inference': types.SimpleNamespace(SoundEventDetection=SED)}):
            result = _music_timeline(np.zeros(105), 10, ['Electronic music', 'Angry music'],
                                     [{'start': 10, 'end': 10.5}], None, checkpoint)
        self.assertEqual(calls, [100, 10])
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['method'], 'framewise')
        self.assertAlmostEqual(result['items'][0]['genre'][0]['score'], .8)
        self.assertEqual(result['items'][0]['end'], 10.5)

    def test_overlap_weighted_sliding_windows(self):
        import numpy as np
        from optional_backend_worker import _sliding_timeline
        model = unittest.mock.Mock()
        model.inference.side_effect = [(np.array([[x, 0.]]), None) for x in [.9, .3, .0]]
        result = _sliding_timeline(model, np.zeros(60), 10, ['Electronic music', 'Angry music'],
                                   [{'start': 0, 'end': 2}, {'start': 2, 'end': 6}])
        self.assertEqual(model.inference.call_count, 3)
        self.assertAlmostEqual(result['items'][0]['genre'][0]['score'], .9)
        self.assertAlmostEqual(result['items'][1]['genre'][0]['score'], (.9 * 3 + .3 * 4) / 9)
        self.assertEqual(result['params']['aggregation'], 'sliding_window_clipwise')

    def test_missing_sed_falls_back_without_import(self):
        import numpy as np
        from optional_backend_worker import _music_timeline
        checkpoint = unittest.mock.Mock()
        checkpoint.is_file.return_value = False
        model = unittest.mock.Mock()
        model.inference.return_value = (np.array([[.8]]), None)
        result = _music_timeline(np.zeros(30), 10, ['Electronic music'], [], model, checkpoint)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['method'], 'sliding_window_clipwise')
        self.assertIn('no download attempted', result['fallback_reason'])
        model.inference.side_effect = RuntimeError('unavailable')
        result = _music_timeline(np.zeros(30), 10, ['Electronic music'], [], model, checkpoint)
        self.assertEqual(result['status'], 'not_run')
        self.assertIn('unavailable', result['reason'])

    def test_sed_error_is_visible_in_fallback(self):
        import numpy as np
        import types
        from optional_backend_worker import _music_timeline
        checkpoint = unittest.mock.Mock()
        checkpoint.is_file.return_value = True
        checkpoint.stat.return_value.st_size = 300000000
        model = unittest.mock.Mock()
        model.inference.return_value = (np.array([[.8]]), None)
        with patch.dict(sys.modules, {'panns_inference': types.SimpleNamespace(
                SoundEventDetection=unittest.mock.Mock(side_effect=RuntimeError('broken checkpoint')))}):
            result = _music_timeline(np.zeros(30), 10, ['Electronic music'], [], model, checkpoint)
        self.assertEqual(result['status'], 'complete')
        self.assertIn('broken checkpoint', result['fallback_reason'])


class MusicStructureQualityTests(unittest.TestCase):
    def features(self, column, columns=8, at=10):
        import numpy as np
        x = np.zeros((30, columns))
        x[at:, column] = 1
        return x

    def test_physical_types_and_normalization(self):
        import numpy as np
        from optional_backend_worker import _novelty_peaks
        for column, kind in [(0, 'energy'), (1, 'timbre'), (2, 'timbre'), (3, 'rhythm'), (4, 'timbre')]:
            with self.subTest(kind=kind, column=column):
                peaks = _novelty_peaks(self.features(column), np.arange(30), duration=30, details=True)
                self.assertEqual(len(peaks), 1)
                self.assertEqual(peaks[0]['type'], kind)
                self.assertTrue(0 <= peaks[0]['magnitude'] <= 1)
                self.assertGreater(peaks[0]['magnitude_raw'], 0)
                self.assertAlmostEqual(sum(peaks[0]['contributions'].values()), 1)

    def test_label_dimension_does_not_amplify_novelty(self):
        import numpy as np
        from optional_backend_worker import _novelty_peaks
        small = self.features(0, columns=5)
        small[10:, 4] = 1
        large = np.column_stack((small[:, :4], np.tile(small[:, 4:5], (1, 527))))
        a, b = [_novelty_peaks(x, np.arange(30), duration=30, details=True)[0] for x in (small, large)]
        self.assertAlmostEqual(a['novelty'], b['novelty'])
        for key in a['contributions']:
            self.assertAlmostEqual(a['contributions'][key], b['contributions'][key])
        self.assertGreater(b['magnitude_raw'], a['magnitude_raw'])

    def test_edges_flat_and_short_audio(self):
        import numpy as np
        from optional_backend_worker import _novelty_peaks, _build_two_layer_timeline
        x = np.zeros((30, 5)); x[1:, 0] = 1; x[10:, 0] = 2; x[28:, 0] = 3
        peaks = _novelty_peaks(x, np.arange(30) * .997, duration=30, details=True)
        self.assertEqual([round(p['time'], 2) for p in peaks], [9.97])
        for duration in (2, 30):
            result = _build_two_layer_timeline(np.zeros(duration * 10), 10, ['Music'],
                                              np.ones((duration, 1)), duration, [])
            self.assertEqual(result['turning_points'], [])
            self.assertTrue(result['no_significant_turns'])
            self.assertEqual(len(result['structure']['segments']), 1)

    def build(self, labels, scores):
        import numpy as np
        from optional_backend_worker import _build_two_layer_timeline
        with patch('optional_backend_worker._audio_features', return_value=self.features(0)):
            return _build_two_layer_timeline(np.zeros(300), 10, labels, np.asarray(scores), 30, [])

    def test_from_to_only_for_changed_nonempty_same_category(self):
        cases = [(['Music'], [[.8]] * 30, 'energy', None, None),
                 (['Rock music'], [[.8]] * 30, 'energy', None, None),
                 (['Rock music'], [[0.]] * 10 + [[.8]] * 20, 'energy', None, None),
                 (['Rock music', 'Sad music'], [[.8, 0.]] * 10 + [[0., .8]] * 20, 'energy', None, None),
                 (['Rock music', 'Jazz'], [[.8, 0.]] * 10 + [[0., .8]] * 20, 'semantic', 'Rock music', 'Jazz'),
                 (['Sad music', 'Happy music'], [[.8, 0.]] * 10 + [[0., .8]] * 20, 'semantic', 'Sad music', 'Happy music')]
        for labels, scores, kind, old, new in cases:
            with self.subTest(labels=labels, kind=kind):
                result = self.build(labels, scores)
                point = result['turning_points'][0]
                self.assertEqual((point['type'], point['from'], point['to']), (kind, old, new))
                self.assertEqual(point['physical_type'], 'energy')
                for key in ('status', 'backend', 'method', 'params', 'experimental'):
                    self.assertIn(key, point)
                self.assertGreaterEqual(result['structure']['segments'][0]['end'], 4)
                json.dumps(result, allow_nan=False)

    def test_structure_configuration(self):
        import numpy as np
        from optional_backend_worker import _build_two_layer_timeline
        args = (np.zeros(300), 10, ['Music'], np.ones((30, 1)), 30, [])
        with patch('optional_backend_worker._audio_features', return_value=self.features(0, at=5)):
            self.assertEqual(len(_build_two_layer_timeline(*args, min_segment_seconds=4)['turning_points']), 1)
            self.assertEqual(_build_two_layer_timeline(*args, min_segment_seconds=6)['turning_points'], [])
            self.assertEqual(_build_two_layer_timeline(*args, min_novelty=100)['turning_points'], [])
        for kw in ({'window_seconds': 0}, {'min_segment_seconds': -1}, {'normalization_percentile': 101}):
            with self.assertRaises(ValueError):
                _build_two_layer_timeline(*args, **kw)

    def test_actual_source_intervals_preserve_label_change(self):
        import numpy as np
        from optional_backend_worker import _build_two_layer_timeline
        # Nonuniform source grid: first score covers ten seconds, not half the clip.
        with patch('optional_backend_worker._audio_features', return_value=self.features(0)):
            result = _build_two_layer_timeline(np.zeros(300), 10, ['Rock music', 'Jazz'],
                                              np.array([[.8, 0.], [0., .8]]), 30, [],
                                              frame_intervals=[(0, 10), (10, 30)])
        point = result['turning_points'][0]
        self.assertEqual((point['from'], point['to']), ('Rock music', 'Jazz'))
        self.assertAlmostEqual(result['structure']['segments'][1]['genre'][0]['score'], .8)
