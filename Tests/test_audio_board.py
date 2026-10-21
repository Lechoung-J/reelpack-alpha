import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
from audio_analyze import build_board_data
from audio_alignment import align_cuts_to_beats
from audio_board import (IMAGE_PATH, METRICS_PATH, deterministic_json, histogram,
                         make_board_payload, render_from_json, validate_payload)
from audio_service import ensure_audio_board


def measured_fixture(beats=(0., .5, 1., 1.5), cuts=(0., .51, 1.1, 1.8)):
    sr = 8000
    t = np.arange(sr * 2) / sr
    board = build_board_data(.4 * np.sin(2 * np.pi * 440 * t), sr, source_hash='a' * 64,
                             max_times=96, max_freqs=64)
    board['cut_times_seconds'] = list(cuts)
    return {'status': 'complete', 'tempo_bpm': 120., 'beat_count': len(beats),
            'beat_times_seconds': list(beats), 'onset_times_seconds': [.01, .49, 1.2],
            'sample_rate': sr, 'duration_seconds': 2., 'board_data': board,
            'cut_alignment': align_cuts_to_beats(cuts, beats),
            'genre': {'status': 'complete', 'items': [{'label': 'NeverBoardGenre'}]},
            'mood': {'status': 'complete', 'items': [{'label': 'NeverBoardMood'}]}}


class BoardDSPTests(unittest.TestCase):
    def test_bounded_waveform_and_real_tone_hz(self):
        sr = 8000
        y = .5 * np.sin(2 * np.pi * 440 * np.arange(sr * 40) / sr)
        data = build_board_data(y, sr)
        self.assertLessEqual(len(data['waveform']['times_seconds']), 600)
        spectrum = data['spectrogram']
        self.assertLessEqual(len(spectrum['values_db']), 600)
        self.assertEqual(len(spectrum['values_db'][0]), 128)
        average = np.mean(spectrum['values_db'][2:-2], axis=0)
        peak = spectrum['frequencies_hz'][int(np.argmax(average))]
        self.assertAlmostEqual(peak, 440, delta=20)
        self.assertEqual(spectrum['params']['frequency_scale'], 'log')
        self.assertEqual(spectrum['params']['excluded_below_hz'], 20.)
        self.assertEqual(spectrum['units']['frequency'], 'Hz')
        self.assertLess(len(deterministic_json(data)), 1_500_000)
        self.assertNotIn('samples', data)
        self.assertEqual(data['duration_seconds'], 40.)

    def test_frequency_rebin_keeps_linear_and_log_tone_locations(self):
        sr = 8000
        t = np.arange(sr * 2) / sr
        for scale in ('linear', 'log'):
            with self.subTest(scale=scale):
                data = build_board_data(np.sin(2 * np.pi * 1700 * t), sr, frequency_scale=scale)
                spec = data['spectrogram']
                peak = spec['frequencies_hz'][int(np.argmax(np.mean(spec['values_db'][2:-2], axis=0)))]
                self.assertAlmostEqual(peak, 1700, delta=60)
                edges = np.asarray(spec['frequency_edges_hz'])
                if scale == 'linear':
                    np.testing.assert_allclose(np.diff(edges), np.diff(edges)[0])
                else:
                    np.testing.assert_allclose(edges[1:] / edges[:-1], edges[1] / edges[0])

    def test_time_rebin_averages_power_before_db(self):
        sr = 8000
        t = np.arange(sr * 2) / sr
        y = np.sin(2 * np.pi * 500 * t)
        y[sr:] *= .1
        fine = build_board_data(y, sr, max_times=2)['spectrogram']
        coarse = build_board_data(y, sr, max_times=1)['spectrogram']
        index = int(np.argmax(coarse['values_db'][0]))
        values = np.asarray(fine['values_db'])[:, index]
        weights = np.asarray(fine['frame_counts'])
        expected = 10 * np.log10(np.average(10 ** (values / 10), weights=weights))
        self.assertAlmostEqual(coarse['values_db'][0][index], expected, delta=.001)
        self.assertGreater(coarse['values_db'][0][index], np.mean(values) + 3)

    def test_rms_db_mean_has_explicit_units_and_silence_finite(self):
        data = build_board_data(np.full(10000, .5), 10000)
        expected = 20 * math.log10(.5)
        self.assertAlmostEqual(data['waveform']['rms_dbfs_mean'], expected)
        self.assertAlmostEqual(data['waveform']['rms_dbfs_global'], expected)
        self.assertEqual(data['waveform']['units']['rms_db'], 'dBFS')
        silent = build_board_data(np.zeros(1), 8000)
        self.assertEqual(silent['waveform']['rms_dbfs_mean'], -120.)
        self.assertTrue(all(x == -80. for x in silent['spectrogram']['values_db'][0]))
        json.dumps(silent, allow_nan=False)


class AudioBoardTests(unittest.TestCase):
    def test_metadata_metrics_reconcile_with_music(self):
        music = measured_fixture()
        payload = make_board_payload(music)
        self.assertEqual(payload['metrics']['tempo_bpm'], 120.)
        self.assertEqual(payload['metrics']['beat_count'], len(music['beat_times_seconds']))
        self.assertEqual(payload['metrics']['cut_count'], 4)
        self.assertEqual(payload['metrics']['aligned_cut_count'], 3)
        self.assertEqual(payload['metrics']['aligned_ratio'], .75)
        self.assertEqual(payload['metrics']['duration_seconds'], 2.)
        self.assertTrue(payload['metrics']['includes_media_start'])
        self.assertEqual(payload['board_data']['source_sha256'], 'a' * 64)
        serialized = deterministic_json(payload)
        self.assertNotIn('NeverBoardGenre', serialized)
        self.assertNotIn('NeverBoardMood', serialized)
        self.assertIn('no genre', serialized.lower())

    def test_persisted_json_replay_and_verified_cache(self):
        music = measured_fixture()
        before = copy.deepcopy(music)
        with tempfile.TemporaryDirectory(prefix='audio board ') as directory:
            root = Path(directory)
            first = ensure_audio_board(music, root)
            self.assertEqual(first['status'], 'complete', first)
            self.assertEqual(music, before)
            metadata = (root / METRICS_PATH).read_bytes()
            png = (root / IMAGE_PATH).read_bytes()
            self.assertEqual(first['sha256'], hashlib.sha256(png).hexdigest())
            music['audio_board'] = first
            with patch('audio_service.render_from_json') as renderer:
                self.assertEqual(ensure_audio_board(music, root), first)
                renderer.assert_not_called()
            with patch('audio_analyze.extract_audio') as decoder:
                replay = ensure_audio_board({}, root, force=True)
                decoder.assert_not_called()
            self.assertEqual(replay['sha256'], first['sha256'])
            self.assertEqual((root / METRICS_PATH).read_bytes(), metadata)
            self.assertEqual((root / IMAGE_PATH).read_bytes(), png)
            alternate = root / 'replay.png'
            self.assertEqual(render_from_json(root / METRICS_PATH, alternate)['sha256'], first['sha256'])
            (root / IMAGE_PATH).write_bytes(b'corrupted cached PNG')
            restored = ensure_audio_board(music, root)
            self.assertEqual(restored['sha256'], first['sha256'])

    def test_no_beats_panel_has_no_histogram_and_null_alignment(self):
        music = measured_fixture(beats=())
        with tempfile.TemporaryDirectory() as directory, patch('audio_board.histogram') as hist:
            result = ensure_audio_board(music, directory)
            hist.assert_not_called()
            self.assertEqual(result['status'], 'complete', result)
            self.assertEqual(result['metrics']['alignment_status'], 'not_measured')
            self.assertIsNone(result['metrics']['aligned_cut_count'])
            self.assertIsNone(result['metrics']['aligned_ratio'])

    def test_histogram_includes_all_offsets_outside_tolerance(self):
        result = histogram([-.7, -.15, 0., .15, 1.2])
        self.assertEqual(sum(result['counts']), 5)
        self.assertEqual(result['extent_seconds'], 1.2)
        self.assertLessEqual(result['edges_seconds'][0], -.7)
        self.assertGreaterEqual(result['edges_seconds'][-1], 1.2)

    def test_renderer_failure_does_not_mutate_physical_result(self):
        music = measured_fixture()
        before = copy.deepcopy(music)
        with tempfile.TemporaryDirectory() as directory, patch('audio_service.render_from_json', side_effect=RuntimeError('mock image failure')):
            result = ensure_audio_board(music, directory)
            self.assertEqual(result['status'], 'not_run')
            self.assertIn('mock image failure', result['reason'])
            self.assertTrue(result['impact'])
            self.assertTrue(result['remedy'])
            self.assertEqual(result['metadata_path'], METRICS_PATH)
        self.assertEqual(music, before)

    def test_invalid_bound_or_package_symlink_is_not_run(self):
        bad = measured_fixture()
        bad['board_data']['spectrogram']['values_db'] *= 601
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(ensure_audio_board(bad, directory)['status'], 'not_run')
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            (Path(directory) / 'evidence').symlink_to(outside, target_is_directory=True)
            result = ensure_audio_board(measured_fixture(), directory)
            self.assertEqual(result['status'], 'not_run')
            self.assertIn('escapes', result['reason'])
            self.assertFalse((Path(outside) / 'audio_board.png').exists())

    def test_render_cli_reads_only_metrics_with_space_paths(self):
        with tempfile.TemporaryDirectory(prefix='audio board ; ') as directory:
            metadata = Path(directory) / 'metrics input.json'
            image = Path(directory) / 'image output.png'
            metadata.write_text(deterministic_json(make_board_payload(measured_fixture())))
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve().parents[1] / 'Core/audio_board.py'),
                                     '--metrics', str(metadata), '--output', str(image)],
                                    text=True, capture_output=True, timeout=60, check=True)
            response = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual(response['status'], 'complete', result.stdout)
            self.assertTrue(image.is_file())


if __name__ == '__main__':
    unittest.main()
