"""Synthetic CAM++ speaker-clustering checks; no download and no network.

The end-to-end tests need the ``sherpa-onnx`` wheel, the audio dependencies
(numpy/scipy/scikit-learn), ffmpeg and a local CAM++ ONNX model. Point
``REELPACK_SPEAKER_MODEL`` at the model, or place the reelpack sample under
``~/Downloads/reelpack-samples/models/``; otherwise those tests are skipped.
"""
import importlib.util
import math
import os
import shutil
import sys
import tempfile
import unittest
import wave
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Core'))
import speaker_campp

AUDIO_DEPENDENCIES = ('numpy', 'scipy', 'sklearn', 'sherpa_onnx')
HAS_AUDIO = all(importlib.util.find_spec(name) for name in AUDIO_DEPENDENCIES)
HAS_FFMPEG = shutil.which('ffmpeg') is not None

DEFAULT_MODEL = (Path.home() / 'Downloads' / 'reelpack-samples' / 'models'
                 / '3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx')
_ENV_MODEL = os.environ.get('REELPACK_SPEAKER_MODEL')
MODEL = Path(_ENV_MODEL).expanduser() if _ENV_MODEL else DEFAULT_MODEL
HAS_MODEL = MODEL.is_file()


class SpeakerCamppContractTests(unittest.TestCase):
    """Failure shapes hold even without sherpa-onnx or the model file."""

    def test_missing_model_is_not_run(self):
        # An explicit path wins over the environment; both are absent here.
        with mock.patch.dict(os.environ, {'REELPACK_SPEAKER_MODEL': ''}):
            result = speaker_campp.cluster_speakers(
                '/nonexistent/reelpack-audio.wav',
                model_path='/nonexistent/campp.onnx')
        self.assertEqual(result['status'], 'not_run')
        self.assertEqual(result['reason'], 'model_missing')
        self.assertEqual(result['windows'], [])
        self.assertEqual(result['clusters'], {})
        self.assertEqual(result['backend'], speaker_campp.BACKEND)

    def test_model_path_takes_precedence_over_environment(self):
        # The environment points at an existing file, but the explicit path is
        # missing, so the call must stop at model_missing rather than fall back.
        with tempfile.NamedTemporaryFile(suffix='.onnx') as environment_model:
            with mock.patch.dict(os.environ,
                                 {'REELPACK_SPEAKER_MODEL': environment_model.name}):
                result = speaker_campp.cluster_speakers(
                    '/nonexistent/reelpack-audio.wav',
                    model_path='/nonexistent/campp.onnx')
        self.assertEqual(result['status'], 'not_run')
        self.assertEqual(result['reason'], 'model_missing')

    def test_invalid_options_fail_without_raising(self):
        for kwargs in ({'k': 0}, {'k': '2'}, {'k': True},
                       {'max_seconds': 0}, {'max_seconds': float('nan')}):
            with self.subTest(**kwargs):
                result = speaker_campp.cluster_speakers(
                    '/nonexistent/reelpack-audio.wav', **kwargs)
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['reason'])
                self.assertNotIn('Traceback', result['reason'])

    def test_injected_clustering_error_fails_at_top_level(self):
        windows = [(0.0, 1.5), (1.5, 3.0), (3.0, 4.5), (4.5, 6.0)]
        segments = [None] * len(windows)
        features = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        candidate = mock.Mock(side_effect=RuntimeError('injected kmeans fault'))
        with mock.patch.multiple(
                speaker_campp,
                resolve_model=mock.Mock(return_value=Path('/fake/campp.onnx')),
                decode_audio=mock.Mock(return_value=([0.0] * 96000, 16000)),
                detect_windows=mock.Mock(return_value=(windows, segments, None)),
                embed_windows=mock.Mock(return_value=(windows, features)),
                _candidate=candidate):
            result = speaker_campp.cluster_speakers('unused.wav')
        candidate.assert_called_once()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('injected kmeans fault', result['reason'])
        self.assertEqual(result['windows'], [])
        self.assertEqual(result['clusters'], {})

    def test_missing_audio_fails_without_raising(self):
        with mock.patch.object(speaker_campp, 'resolve_model',
                               return_value=Path('/fake/campp.onnx')):
            result = speaker_campp.cluster_speakers('/nonexistent/reelpack-audio.wav')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('FileNotFoundError', result['reason'])


@unittest.skipUnless(HAS_AUDIO and HAS_FFMPEG and HAS_MODEL,
                     'requires sherpa-onnx + audio deps + ffmpeg + a local CAM++ model')
class SpeakerCamppSyntheticTests(unittest.TestCase):
    sample_rate = 16000

    @classmethod
    def setUpClass(cls):
        if not HAS_MODEL:
            raise unittest.SkipTest('local CAM++ model not found')

    def speech_voice(self, f0, seconds, *, scale=1.0, seed=0, syllable_hz=3.0):
        """Rough source-filter speech: glottal pulses through moving vowel formants.

        CAM++ is trained on speech; stationary tones collapse to one embedding,
        while this syllable-modulated formant signal preserves speaker cues.
        """
        import numpy as np
        from scipy.signal import lfilter

        def formant_filter(source, formants):
            filtered = np.zeros_like(source)
            for center, bandwidth in formants:
                radius = math.exp(-math.pi * bandwidth / self.sample_rate)
                theta = 2 * math.pi * center / self.sample_rate
                filtered += lfilter([1 - radius],
                                    [1, -2 * radius * math.cos(theta), radius ** 2],
                                    source)
            return filtered

        rng = np.random.default_rng(seed)
        count = int(round(seconds * self.sample_rate))
        times = np.arange(count) / self.sample_rate
        phase = np.cumsum(f0 * (1 + 0.02 * rng.standard_normal(count))) / self.sample_rate
        pulses = np.diff(np.floor(phase), prepend=0.0)
        source = pulses * 2.0 - 0.02 * rng.standard_normal(count)
        vowels = [(730, 1090, 2440), (270, 2290, 3010), (530, 1840, 2480),
                  (570, 840, 2410), (300, 870, 2240)]
        signal = np.zeros(count)
        for index in range(int(seconds / 0.25) + 1):
            start = int(index * 0.25 * self.sample_rate)
            end = min(count, start + int(0.25 * self.sample_rate))
            if start >= count:
                break
            signal[start:end] = formant_filter(
                source[start:end], [(frequency * scale, 80.0)
                                    for frequency in vowels[index % len(vowels)]])
        envelope = 0.5 + 0.5 * np.abs(np.sin(2 * math.pi * syllable_hz * times + seed))
        pauses = np.ones(count)
        for start in range(0, count, int(1.2 * self.sample_rate)):
            pauses[start:min(count, start + int(0.15 * self.sample_rate))] = 0.0
        envelope *= pauses
        ramp_length = int(0.3 * self.sample_rate)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, ramp_length))
        envelope[:ramp_length] *= ramp
        envelope[-ramp_length:] *= ramp[::-1]
        signal *= envelope
        signal += 0.001 * rng.standard_normal(count)
        signal /= max(1e-9, float(np.max(np.abs(signal))))
        return (0.3 * signal).astype(np.float32)

    def write_sequence(self, path, blocks, gap):
        import numpy as np
        parts = []
        for index, block in enumerate(blocks):
            parts.append(block)
            if index < len(blocks) - 1:
                parts.append(np.zeros(int(gap * self.sample_rate), dtype=np.float32))
        audio = np.concatenate(parts)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype('<i2')
        with wave.open(str(path), 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(pcm.tobytes())

    @staticmethod
    def overlap(first_start, first_end, second_start, second_end):
        return max(0.0, min(first_end, second_end) - max(first_start, second_start))

    def purity(self, windows, intervals):
        assignments = []
        for window in windows:
            best, best_overlap = None, 0.0
            for start, end, label in intervals:
                overlap = self.overlap(window['start'], window['end'], start, end)
                if overlap > best_overlap:
                    best, best_overlap = label, overlap
            assignments.append((window['cluster'], best))
        groups = {}
        for prediction, label in assignments:
            groups.setdefault(prediction, Counter())[label] += 1
        return (sum(max(counts.values()) for counts in groups.values())
                / len(assignments))

    def test_two_synthetic_voices_split_into_two_clusters(self):
        speaker_a = self.speech_voice(95.0, 4.5, scale=0.90, seed=1, syllable_hz=3.0)
        speaker_b = self.speech_voice(230.0, 4.5, scale=1.25, seed=2, syllable_hz=4.0)
        blocks, intervals, cursor = [], [], 0.0
        for _ in range(4):
            blocks.append(speaker_a)
            intervals.append((cursor, cursor + 4.5, 'A'))
            cursor += 6.0
            blocks.append(speaker_b)
            intervals.append((cursor, cursor + 4.5, 'B'))
            cursor += 6.0
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'two-speakers.wav'
            self.write_sequence(audio, blocks, 1.5)
            result = speaker_campp.cluster_speakers(str(audio), model_path=MODEL)
        self.assertEqual(result['status'], 'complete', result.get('reason'))
        self.assertEqual(result['backend'], speaker_campp.BACKEND)
        self.assertEqual(len(result['clusters']), 2, result['confidence'])
        self.assertGreaterEqual(result['confidence']['silhouette'], 0.35)
        self.assertGreaterEqual(self.purity(result['windows'], intervals), 0.95)

    def test_alternating_speakers_with_long_silence_stay_two(self):
        # Two voices alternate, then the recording ends in room tone that is
        # above the absolute RMS floor (as real recordings do). The tail used
        # to be embedded as an extra cluster, so whole-file auto-k returned
        # three clusters; on files where it wins the k=2 split instead it
        # merges the two speakers into one.
        import numpy as np
        speaker_a = self.speech_voice(95.0, 4.5, scale=0.90, seed=1, syllable_hz=3.0)
        speaker_b = self.speech_voice(230.0, 4.5, scale=1.25, seed=2, syllable_hz=4.0)
        blocks, intervals, cursor = [], [], 0.0
        for _ in range(4):
            blocks.append(speaker_a)
            intervals.append((cursor, cursor + 4.5, 'A'))
            cursor += 6.0
            blocks.append(speaker_b)
            intervals.append((cursor, cursor + 4.5, 'B'))
            cursor += 6.0
        room_tone = (0.01 * np.random.default_rng(11).standard_normal(
            int(24.0 * self.sample_rate))).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'alternating-with-silence.wav'
            self.write_sequence(audio, blocks + [room_tone], 1.5)
            result = speaker_campp.cluster_speakers(str(audio), model_path=MODEL)
        self.assertEqual(result['status'], 'complete', result.get('reason'))
        self.assertEqual(len(result['clusters']), 2, result['confidence'])
        self.assertEqual(result['confidence']['selected_k'], 2)
        self.assertGreaterEqual(self.purity(result['windows'], intervals), 0.95)

    def test_single_continuous_voice_stays_one_cluster(self):
        # k=1 remains legal: a lone continuous speaker has no centroid gap and
        # must not be split just because k=2 is preferred when it is admissible.
        block = self.speech_voice(120.0, 24.0, seed=5)
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'one-speaker.wav'
            self.write_sequence(audio, [block], 0.0)
            result = speaker_campp.cluster_speakers(str(audio), model_path=MODEL)
        self.assertEqual(result['status'], 'complete', result.get('reason'))
        self.assertEqual(len(result['clusters']), 1, result['confidence'])
        self.assertEqual(result['confidence']['selected_k'], 1)

    def test_copied_segment_is_not_split(self):
        # A 0.75 s loop aligns with the hop, so every 1.5 s window sees the
        # same content and the backend must not invent a split from copies.
        block = self.speech_voice(140.0, 0.75, seed=3)
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'one-source.wav'
            self.write_sequence(audio, [block] * 32, 0.0)
            result = speaker_campp.cluster_speakers(str(audio), model_path=MODEL)
        self.assertEqual(result['status'], 'complete', result.get('reason'))
        self.assertEqual(result['window_count'], 30, result['windows'])
        self.assertEqual(len(result['clusters']), 1, result['confidence'])
        self.assertTrue(all(window['cluster'] == 0 for window in result['windows']))


if __name__ == '__main__':
    unittest.main()
