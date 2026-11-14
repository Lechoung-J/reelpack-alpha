"""Synthetic speaker-clustering checks; no downloaded models or network."""
import importlib.util
import io
import json
import math
import shutil
from collections import Counter
from contextlib import redirect_stdout
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Core'))
import speaker_cluster

_VOICE_EVAL_SPEC = importlib.util.spec_from_file_location(
    'voice_eval', ROOT / 'Tools' / 'voice_eval.py')
voice_eval = importlib.util.module_from_spec(_VOICE_EVAL_SPEC)
_VOICE_EVAL_SPEC.loader.exec_module(voice_eval)

AUDIO_DEPENDENCIES = ('numpy', 'scipy', 'librosa', 'sklearn', 'soundfile')
HAS_AUDIO = all(importlib.util.find_spec(name) for name in AUDIO_DEPENDENCIES)
HAS_FFMPEG = shutil.which('ffmpeg') is not None


class SpeakerClusterContractTests(unittest.TestCase):
    """Failure shapes hold even without the audio interpreter."""

    def test_missing_file_fails_without_raising(self):
        result = speaker_cluster.cluster_speakers('/nonexistent/reelpack-audio.wav')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('reason', result)
        self.assertEqual(result['windows'], [])
        self.assertEqual(result['clusters'], {})

    def test_invalid_options_fail_without_raising(self):
        for kwargs in ({'k': 0}, {'k': '2'}, {'max_seconds': 0}, {'max_seconds': float('nan')}):
            with self.subTest(**kwargs):
                result = speaker_cluster.cluster_speakers('/nonexistent/reelpack-audio.wav', **kwargs)
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['reason'])
                self.assertNotIn('Traceback', result['reason'])

    sample_rate = 16000

    def _patched_pipeline(self, *, windows, **overrides):
        """Patch the heavy pipeline so contract tests need no audio backend."""
        patches = {
            'decode_audio': mock.Mock(
                return_value=([0.0] * (self.sample_rate * 3), self.sample_rate)),
            'detect_windows': mock.Mock(return_value=(windows, None)),
            '_window_features': mock.Mock(return_value=(windows, object())),
            '_standardize': mock.Mock(return_value=object()),
        }
        patches.update(overrides)
        return mock.patch.multiple(speaker_cluster, **patches)

    def test_candidate_clustering_error_fails_at_top_level(self):
        windows = [(0.0, 1.0), (1.4, 2.4), (2.8, 3.8), (4.2, 5.2)]
        features = [(0.0, 1.0), (0.2, 0.9), (0.8, 0.1), (0.9, 0.0)]
        # An error inside _choose_clusters must not degrade to a
        # single-cluster "complete" result.
        candidate = mock.Mock(side_effect=RuntimeError('injected kmeans fault'))
        with self._patched_pipeline(
                windows=windows,
                _standardize=mock.Mock(return_value=features),
                _candidate=candidate):
            result = speaker_cluster.cluster_speakers('unused.wav')
        candidate.assert_called_once()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('injected kmeans fault', result['reason'])
        self.assertEqual(result['windows'], [])
        self.assertEqual(result['clusters'], {})

    def test_fewer_than_three_windows_marks_low_window_count(self):
        windows = [(0.0, 1.0), (1.5, 2.5)]
        # Auto mode must not even attempt a split below three windows.
        choose = mock.Mock(side_effect=AssertionError('unused'))
        with self._patched_pipeline(windows=windows, _choose_clusters=choose):
            result = speaker_cluster.cluster_speakers('unused.wav')
        choose.assert_not_called()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['window_count'], 2)
        self.assertEqual(len(result['clusters']), 1)
        self.assertTrue(result['confidence']['low_window_count'])
        self.assertIn('样本过短，未做区分', result['reason'])


class VoiceEvalCliTests(unittest.TestCase):
    """The CLI accepts both report targets; --out-dir writes into a directory."""

    def test_out_dir_writes_report_without_conflicting_with_json_out(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            out_dir = base / 'nested' / 'report'
            json_out = base / 'exact.json'
            with redirect_stdout(io.StringIO()):
                status = voice_eval.main([
                    'missing-a.wav', 'missing-b.wav',
                    '--out-dir', str(out_dir), '--json-out', str(json_out)])
            self.assertEqual(status, 1)  # missing inputs are a failed report, not a crash
            written = out_dir / 'voice_eval.json'
            self.assertTrue(written.is_file())
            self.assertTrue(json_out.is_file())
            report = json.loads(written.read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(json.loads(json_out.read_text(encoding='utf-8'))['status'],
                             'failed')


@unittest.skipUnless(HAS_AUDIO and HAS_FFMPEG,
                     'requires .venv-audio (numpy/librosa/scikit-learn/soundfile) and ffmpeg')
class SpeakerClusterSyntheticTests(unittest.TestCase):
    sample_rate = 16000

    def harmonic_voice(self, f0, seconds, *, seed=0,
                       formants=((500.0, 1.0), (1500.0, 0.45)), wobble=True):
        import numpy as np
        rng = np.random.default_rng(seed)
        times = np.arange(int(round(seconds * self.sample_rate))) / self.sample_rate
        signal = np.zeros_like(times)
        for harmonic in range(1, 60):
            frequency = f0 * harmonic
            if frequency >= self.sample_rate / 2 - 200:
                break
            weight = sum(gain * math.exp(-0.5 * ((frequency - center) / 220.0) ** 2)
                         for center, gain in formants)
            if weight <= 1e-4:
                weight = 1.0 / harmonic ** 2
            signal += weight * np.sin(2 * math.pi * frequency * times + 0.2 * harmonic)
        if wobble:
            signal *= 0.75 + 0.25 * np.sin(2 * math.pi * 2.3 * times + seed)
        signal += 0.002 * rng.standard_normal(len(times))
        signal /= max(1e-9, float(np.max(np.abs(signal))))
        return (0.25 * signal).astype(np.float32)

    def write_sequence(self, path, blocks, *, gap=0.4, lead=0.3):
        import numpy as np
        import soundfile as sf
        parts = [np.zeros(int(lead * self.sample_rate), dtype=np.float32)]
        intervals = []
        cursor = len(parts[0])
        gap_samples = np.zeros(int(gap * self.sample_rate), dtype=np.float32)
        for block in blocks:
            start = cursor / self.sample_rate
            parts.append(block)
            cursor += len(block)
            intervals.append((start, cursor / self.sample_rate))
            parts.append(gap_samples)
            cursor += len(gap_samples)
        sf.write(str(path), np.concatenate(parts), self.sample_rate, subtype='PCM_16')
        return intervals

    @staticmethod
    def overlap(first_start, first_end, second_start, second_end):
        return max(0.0, min(first_end, second_end) - max(first_start, second_start))

    def truth_labels(self, windows, intervals):
        labels = []
        for window in windows:
            best, best_overlap = None, 0.0
            for (start, end), label in intervals:
                overlap = self.overlap(window['start'], window['end'], start, end)
                if overlap > best_overlap:
                    best, best_overlap = label, overlap
            labels.append(best)
        return labels

    @staticmethod
    def purity(predicted, truth):
        pairs = [(prediction, label) for prediction, label in zip(predicted, truth)
                 if label is not None]
        groups = {}
        for prediction, label in pairs:
            groups.setdefault(prediction, Counter())[label] += 1
        return sum(max(counts.values()) for counts in groups.values()) / len(pairs)

    def interleaved_utterances(self):
        """Four utterances per speaker, each with its own envelope phase/noise."""
        speaker_a = [self.harmonic_voice(110.0, 1.6, seed=seed) for seed in (1, 3, 5, 7)]
        speaker_b = [self.harmonic_voice(220.0, 1.6, seed=seed,
                                         formants=((900.0, 1.0), (2300.0, 0.45)))
                     for seed in (2, 4, 6, 8)]
        blocks = []
        for first, second in zip(speaker_a, speaker_b):
            blocks.extend([first, second])
        return blocks

    def test_two_different_f0_sources_split_into_two_clusters(self):
        truth = [0, 1, 0, 1, 0, 1, 0, 1]
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'two-speakers.wav'
            intervals = self.write_sequence(audio, self.interleaved_utterances())
            result = speaker_cluster.cluster_speakers(str(audio))
        self.assertEqual(result['status'], 'complete', result.get('reason'))
        self.assertEqual(result['window_count'], 8, result['windows'])
        self.assertEqual(len(result['clusters']), 2, result['confidence'])
        labels = self.truth_labels(result['windows'], [(span, label)
                                                       for span, label in zip(intervals, truth)])
        self.assertGreaterEqual(self.purity([w['cluster'] for w in result['windows']], labels), 0.95)
        self.assertGreaterEqual(result['confidence']['silhouette'], 0.55)

    def test_copied_single_source_is_not_split(self):
        block = self.harmonic_voice(140.0, 2.5, seed=7, wobble=False)
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'one-speaker.wav'
            intervals = self.write_sequence(audio, [block] * 8)
            result = speaker_cluster.cluster_speakers(str(audio))
            intervals = [(span, 0) for span in intervals]
        self.assertEqual(result['status'], 'complete', result.get('reason'))
        self.assertEqual(result['window_count'], 8, result['windows'])
        self.assertEqual(len(result['clusters']), 1, result['confidence'])
        self.assertTrue(all(window['cluster'] == 0 for window in result['windows']))
        self.assertTrue(all(label == 0 for label in
                            self.truth_labels(result['windows'], intervals)))

    def test_fixed_k_is_respected(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'fixed-k.wav'
            self.write_sequence(audio, self.interleaved_utterances())
            result = speaker_cluster.cluster_speakers(str(audio), k=2)
        self.assertEqual(result['status'], 'complete', result.get('reason'))
        self.assertEqual(len(result['clusters']), 2)
        self.assertEqual(result['confidence']['method'], 'fixed_k')

    def test_silent_audio_is_not_run(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'silence.wav'
            sf.write(str(audio), np.zeros(self.sample_rate * 2, dtype=np.float32),
                     self.sample_rate, subtype='PCM_16')
            result = speaker_cluster.cluster_speakers(str(audio))
        self.assertEqual(result['status'], 'not_run')
        self.assertIn('silent', result['reason'])


if __name__ == '__main__':
    unittest.main()
