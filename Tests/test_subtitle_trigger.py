"""Subtitle-band trigger contracts plus the ad/talk/podcast pipeline wiring."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))

import core
from asset_contract import validate_analysis
from ocr_conflicts import find_conflicts
from ocr_regions import image_size
from subtitle_trigger import (band_signature, band_signatures, changed, count_changes,
                              count_signature_changes, densify, duplicate_text_timestamps,
                              extract_band_frames, normalize_text, signature_change_pairs,
                              signature_changed, signature_difference, subtitle_band_times,
                              triggered_times, triggered_times_from_signatures)


def fake_signature(value):
    """Deterministic 64x16 signature stand-in for pipeline wiring tests."""
    digest = hashlib.sha256(str(value).encode('utf-8')).digest()
    return (digest * 64)[:1024]


try:
    import cv2
    import numpy as np
except ImportError:  # Media fixtures below are skipped without OpenCV.
    cv2 = None
    np = None


class SubtitleTextTests(unittest.TestCase):
    def test_normalize_text_folds_width_case_whitespace_and_punctuation(self):
        self.assertEqual(normalize_text('  ＡＢＣ，１２３！ '), 'abc123')
        self.assertEqual(normalize_text('Hello, WORLD!'), 'helloworld')
        self.assertEqual(normalize_text('第一句：字幕。'), '第一句字幕')
        self.assertEqual(normalize_text(None), '')

    def test_repeated_text_does_not_trigger(self):
        self.assertFalse(changed('我们今天讨论镜头语言', '我们今天讨论镜头语言。'))
        self.assertFalse(changed('  Hello World ', 'hello，world！'))
        self.assertFalse(changed('', ''))

    def test_a_different_sentence_triggers(self):
        self.assertTrue(changed('今天我们讨论镜头语言', '明天我们去公园散步'))
        self.assertTrue(changed('', '第一句字幕出现'))

    def test_edit_distance_ratio_respects_threshold(self):
        self.assertTrue(changed('abcdefghij', 'abcdefgxyz', threshold=0.15))
        self.assertFalse(changed('abcdefghij', 'abcdefgxyz', threshold=0.9))
        with self.assertRaises(ValueError):
            changed('a', 'b', threshold=-0.1)

    def test_densify_is_symmetric_and_includes_the_change_point(self):
        self.assertEqual(densify(10.0), [9.0, 9.5, 10.0, 10.5, 11.0])
        self.assertEqual(densify(10.0, window=2.0, step=1.0), [8.0, 9.0, 10.0, 11.0, 12.0])
        self.assertEqual(len(densify(10.0)), 5)
        with self.assertRaises(ValueError):
            densify(10.0, step=0)

    def test_triggered_times_adds_both_sides_of_a_change(self):
        observations = [(0.0, '第一句字幕'), (30.0, '第一句字幕'), (60.0, '完全不同的第二句')]
        self.assertEqual(triggered_times(observations), [59.0, 59.5, 60.0, 60.5, 61.0])
        self.assertEqual(triggered_times(observations, existing=[60.0]),
                         [59.0, 59.5, 60.5, 61.0])
        self.assertEqual(triggered_times(observations, duration=60.0),
                         [59.0, 59.5])

    def test_same_observations_schedule_nothing(self):
        self.assertEqual(triggered_times([(0, '同句'), (30, '同句。'), (60, '  同句 ')]), [])

    def test_change_count_matches_the_triggered_transitions(self):
        slow = [(t, '同一句字幕') for t in (0.0, 30.0, 60.0, 90.0)]
        fast = [(t, text) for t, text in zip((0.0, 30.0, 60.0, 90.0),
                                            ('春眠不觉晓处处闻啼鸟', '白日依山尽黄河入海流',
                                             '床前明月光疑是地上霜', '欲穷千里目更上一层楼'))]
        self.assertEqual(count_changes(slow), 0)
        self.assertEqual(count_changes(fast), 3)
        self.assertEqual(count_changes(fast), len(triggered_times(fast)) // 5)


class BandSignatureTriggerTests(unittest.TestCase):
    """Image signatures drive the trigger; text is only a dedup hint."""

    @staticmethod
    def signature(*changes):
        image = bytearray(1024)
        for index, value in changes:
            image[index] = value
        return bytes(image)

    def test_identical_band_images_do_not_trigger(self):
        baseline = self.signature()
        self.assertFalse(signature_changed(baseline, baseline))
        self.assertEqual(signature_change_pairs([(0.0, baseline), (30.0, baseline)]), [])
        self.assertEqual(count_signature_changes([(0.0, baseline), (30.0, baseline)]), 0)

    def test_a_few_changed_pixels_trigger_below_the_mean_threshold(self):
        # One glyph-sized change: three pixels move by 255.  Mean absolute
        # difference stays far below 2/255, but the Hamming-like branch fires.
        before = self.signature()
        after = self.signature((10, 255), (11, 255), (12, 255))
        difference = signature_difference(before, after)
        self.assertLess(difference['mean_abs_diff'], 2.0 / 255.0)
        self.assertEqual(difference['changed_pixels'], 3)
        self.assertTrue(signature_changed(before, after))

    def test_mean_difference_triggers_when_every_pixel_moves_slightly(self):
        before = self.signature()
        after = bytes(min(255, value + 3) for value in before)
        difference = signature_difference(before, after)
        self.assertGreater(difference['mean_abs_diff'], 2.0 / 255.0)
        self.assertEqual(difference['changed_pixels'], 0)
        self.assertTrue(signature_changed(before, after))
        self.assertFalse(signature_changed(before, after, mean_abs_threshold=0.5))

    def test_pixel_delta_below_the_threshold_is_ignored(self):
        before = self.signature()
        after = self.signature((0, 8), (1, 8), (2, 8))
        self.assertEqual(signature_difference(before, after)['changed_pixels'], 0)
        self.assertFalse(signature_changed(before, after))

    def test_signature_changes_schedule_densified_windows(self):
        before, after = self.signature(), self.signature((5, 255), (6, 255), (7, 255))
        signatures = [(0.0, before), (30.0, before), (60.0, after)]
        self.assertEqual(signature_change_pairs(signatures), [(30.0, 60.0)])
        self.assertEqual(triggered_times_from_signatures(signatures),
                         [59.0, 59.5, 60.0, 60.5, 61.0])
        self.assertEqual(triggered_times_from_signatures(signatures, existing=[60.0]),
                         [59.0, 59.5, 60.5, 61.0])
        self.assertEqual(triggered_times_from_signatures(signatures, duration=60.0),
                         [59.0, 59.5])
        with self.assertRaises(ValueError):
            signature_changed(before, after, changed_pixel_count=-1)

    def test_text_only_dedupes_identical_sentences_after_an_image_change(self):
        before, after = self.signature(), self.signature((5, 255), (6, 255), (7, 255))
        signatures = [(0.0, before), (10.0, after)]
        self.assertEqual(signature_change_pairs(signatures), [(0.0, 10.0)])
        # Same sentence with different punctuation is one record.
        self.assertEqual(duplicate_text_timestamps(signatures, {0.0: '同一句字幕', 10.0: '同一句字幕。'}),
                         {10.0})
        # A one-character subtitle change must not be deduplicated.
        self.assertEqual(duplicate_text_timestamps(signatures, {0.0: 'slow line 0', 10.0: 'slow line 3'}),
                         set())
        self.assertFalse(changed('slow line 0', 'slow line 3'))
        # Unreadable text must never be treated as a duplicate.
        self.assertEqual(duplicate_text_timestamps(signatures, {0.0: '', 10.0: ''}), set())
        skipped = duplicate_text_timestamps(signatures, {0.0: '同一句', 10.0: '同一句。'})
        self.assertEqual(triggered_times_from_signatures(signatures, skip_timestamps=skipped), [])

    def test_band_signature_needs_image_data(self):
        with self.assertRaises(ValueError):
            band_signature(b'')
        with self.assertRaises(ValueError):
            band_signature(b'not an image')
        with self.assertRaises(ValueError):
            band_signature(b'x', size=(0, 16))


class BandFrameDropTests(unittest.TestCase):
    """A timed-out sampling point is reported, never silently skipped."""

    @staticmethod
    def run_for(command, **kwargs):
        timestamp = command[command.index('-ss') + 1]
        if timestamp == '1.0':
            raise subprocess.TimeoutExpired(command, kwargs.get('timeout', 60))
        return SimpleNamespace(returncode=0, stdout=b'\xff\xd8jpeg', stderr=b'')

    def test_timeout_is_reported_as_a_dropped_sampling_point(self):
        dropped = []
        with patch('subtitle_trigger.subprocess.run', side_effect=self.run_for):
            frames = extract_band_frames('media.mp4', [0.0, 1.0, 2.0], dropped=dropped)
        self.assertEqual([timestamp for timestamp, _ in frames], [0.0, 2.0])
        self.assertEqual(dropped, [{'timestamp': 1.0, 'reason': 'ffmpeg_timeout'}])

    def test_decode_failure_is_reported_with_its_own_reason(self):
        dropped = []
        with patch('subtitle_trigger.subprocess.run',
                   return_value=SimpleNamespace(returncode=1, stdout=b'', stderr=b'decode error')):
            with self.assertRaises(RuntimeError):
                extract_band_frames('media.mp4', [0.5, 1.5], dropped=dropped)
        self.assertEqual(dropped, [{'timestamp': 0.5, 'reason': 'ffmpeg_failed'},
                                   {'timestamp': 1.5, 'reason': 'ffmpeg_failed'}])

    def test_all_failed_pass_still_carries_dropped_metadata(self):
        with patch('subtitle_trigger.subprocess.run', side_effect=self.run_for):
            with self.assertRaises(RuntimeError) as caught:
                extract_band_frames('media.mp4', [1.0])
        self.assertEqual(caught.exception.attempted, 1)
        self.assertEqual(caught.exception.dropped, [{'timestamp': 1.0, 'reason': 'ffmpeg_timeout'}])

    def test_conflict_marked_without_rewriting_either_text(self):
        transcript = [{'start': 0, 'end': 4, 'text': '今天我们讨论摄影画面的构图'}]
        ocr = [{'start': 1, 'end': 2, 'text': '完全不同的一段画面文字内容'}]
        conflicts = find_conflicts(ocr, transcript)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(set(conflicts[0]), {'start', 'end', 'ocr_text', 'transcript_text', 'similarity'})
        self.assertLess(conflicts[0]['similarity'], 0.6)
        self.assertEqual(conflicts[0]['ocr_text'], ocr[0]['text'])
        self.assertEqual(conflicts[0]['transcript_text'], transcript[0]['text'])
        self.assertEqual(ocr[0]['text'], '完全不同的一段画面文字内容')
        self.assertEqual(transcript[0]['text'], '今天我们讨论摄影画面的构图')

    def test_short_or_similar_or_non_overlapping_pairs_are_not_marked(self):
        transcript = [{'start': 0, 'end': 4, 'text': '今天我们讨论摄影画面的构图'}]
        cases = [
            ([{'start': 1, 'end': 2, 'text': '不同文字'}], 'short'),
            ([{'start': 1, 'end': 2, 'text': '今天我们讨论摄影画面的构图'}], 'similar'),
            ([{'start': 9, 'end': 10, 'text': '完全不同的一段画面文字内容'}], 'apart'),
        ]
        for ocr, label in cases:
            with self.subTest(label=label):
                self.assertEqual(find_conflicts(ocr, transcript), [])

    def test_point_ocr_observation_inside_a_segment_overlaps(self):
        transcript = [{'start': 0, 'end': 4, 'text': '今天我们讨论摄影画面的构图'}]
        ocr = [{'timestamp': 2.5, 'text': '完全不同的一段画面文字内容'}]
        self.assertEqual(len(find_conflicts(ocr, transcript)), 1)


@unittest.skipUnless(cv2 is not None, 'cv2 unavailable')
class SubtitleBandMediaTests(unittest.TestCase):
    def two_tone_video(self, root, seconds=2, size=(320, 180), fps=5):
        path = Path(root) / 'two-tone.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        try:
            for _ in range(seconds * fps):
                frame = np.zeros((size[1], size[0], 3), np.uint8)
                frame[:int(size[1] * 0.85)] = (0, 0, 255)   # red top (BGR)
                frame[int(size[1] * 0.85):] = (255, 0, 0)   # blue bottom band
                writer.write(frame)
        finally:
            writer.release()
        return path

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg unavailable')
    def test_band_frames_decode_only_the_bottom_fifteen_percent(self):
        with tempfile.TemporaryDirectory() as td:
            video = self.two_tone_video(td)
            frames = extract_band_frames(video, [0.2, 1.2], band=0.15)
            self.assertEqual([timestamp for timestamp, _ in frames], [0.2, 1.2])
            for _, image in frames:
                width, height = image_size(image)
                self.assertEqual(width, 320)
                self.assertLessEqual(height, 28)
                decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
                blue, green, red = decoded.mean(axis=(0, 1))
                self.assertGreater(blue, 200)
                self.assertLess(red, 60)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg unavailable')
    def test_band_times_cover_a_long_clip_at_the_requested_interval(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / 'long.mp4'
            import subprocess
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                            'color=c=black:s=160x90:r=2:d=95', '-c:v', 'mpeg4', '-y', str(video)],
                           check=True, capture_output=True)
            self.assertEqual(subtitle_band_times(video, interval=30.0, duration=95),
                             [0.0, 30.0, 60.0, 90.0])
            self.assertEqual(subtitle_band_times(video, interval=60.0, duration=95),
                             [0.0, 60.0, 94.5])
            self.assertEqual(subtitle_band_times(video, interval=30.0, duration=10),
                             [0.0, 9.5])

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            subtitle_band_times('missing.mp4', interval=0, duration=10)
        with self.assertRaises(ValueError):
            subtitle_band_times('missing.mp4', band=0, duration=10)
        with self.assertRaises(ValueError):
            extract_band_frames('missing.mp4', [0], band=-0.1)


@unittest.skipUnless(cv2 is not None, 'cv2 unavailable')
@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg unavailable')
class SubtitleBandImageMediaTests(unittest.TestCase):
    """Real ffmpeg decoding: visual changes trigger, static frames do not."""

    WIDTH, HEIGHT, FPS = 640, 360, 5

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.slow = cls.make_video(root / 'slow.mp4', 30, 10,
                                  ['SLOW LINE 0', 'SLOW LINE 1', 'SLOW LINE 2'])
        cls.fast = cls.make_video(root / 'fast.mp4', 30, 2,
                                  [f'FAST LINE {index}' for index in range(15)])
        cls.static = cls.make_video(root / 'static.mp4', 30, None, ['STATIC TEXT'])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def make_video(cls, path, seconds, change_every, lines):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), cls.FPS,
                                 (cls.WIDTH, cls.HEIGHT))
        try:
            band_top = int(cls.HEIGHT * 0.85)
            for index in range(seconds * cls.FPS):
                frame = np.full((cls.HEIGHT, cls.WIDTH, 3), 30, np.uint8)
                frame[band_top:] = (245, 245, 245)
                block = index // (change_every * cls.FPS) if change_every else 0
                text = lines[block % len(lines)]
                (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.putText(frame, text, ((cls.WIDTH - width) // 2,
                                          band_top + (cls.HEIGHT - band_top + height) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2, cv2.LINE_AA)
                writer.write(frame)
        finally:
            writer.release()
        return path

    def signatures(self, video, seconds=30, interval=1.0):
        times = subtitle_band_times(video, interval=interval, band=0.15, duration=seconds)
        frames = extract_band_frames(video, times, band=0.15, max_width=0)
        self.assertEqual([timestamp for timestamp, _ in frames], times)
        return times, band_signatures(frames, size=(64, 16))

    def test_slow_one_character_subtitle_changes_are_all_triggered(self):
        times, signatures = self.signatures(self.slow)
        changes = 2  # three 10-second subtitle blocks
        triggered = count_signature_changes(signatures)
        self.assertEqual(triggered, changes)
        self.assertGreaterEqual(triggered, changes - 1)
        # Text edit distance alone misses this one-character change.
        self.assertFalse(changed('SLOW LINE 0', 'SLOW LINE 1'))
        densified = triggered_times_from_signatures(signatures, duration=30, existing=times)
        # 9.0/11.0/19.0/21.0 are already coarse samples and are not repeated.
        self.assertEqual(densified, [9.5, 10.5, 19.5, 20.5])

    def test_sparse_coarse_scan_still_catches_one_character_changes(self):
        # The 60s/talk case samples every 30s; with a 30s clip and a
        # 10s interval both transitions still produce a trigger.
        _, signatures = self.signatures(self.slow, interval=10.0)
        self.assertEqual(count_signature_changes(signatures), 2)

    def test_reported_60s_talk_case_now_triggers(self):
        # A 60s clip, subtitles every 10s, and talk's 30s coarse
        # interval -> coarse scans [0, 30, 59.5].  Text edit distance says
        # "SLOW LINE 0" -> "SLOW LINE 3" is ~0.08 (< 0.15) and triggers 0;
        # band-image signatures trigger at both transitions.
        video = self.make_video(Path(self.tmp.name) / 'slow60.mp4', 60, 10,
                                [f'SLOW LINE {index}' for index in range(6)])
        times = subtitle_band_times(video, interval=30.0, band=0.15, duration=60)
        self.assertEqual(times, [0.0, 30.0, 59.5])
        frames = extract_band_frames(video, times, band=0.15, max_width=0)
        self.assertEqual(count_signature_changes(band_signatures(frames, size=(64, 16))), 2)
        self.assertFalse(changed('SLOW LINE 0', 'SLOW LINE 3'))

    def test_fast_subtitle_cycles_trigger_more_than_slow(self):
        _, slow_signatures = self.signatures(self.slow)
        _, fast_signatures = self.signatures(self.fast)
        slow = count_signature_changes(slow_signatures)
        fast = count_signature_changes(fast_signatures)
        self.assertGreaterEqual(fast, 12)  # 2s -> 30s yields 14 transitions
        self.assertGreater(fast, slow * 3)

    def test_a_static_band_triggers_nothing(self):
        times, signatures = self.signatures(self.static)
        self.assertEqual(count_signature_changes(signatures), 0)
        self.assertEqual(triggered_times_from_signatures(signatures, duration=30, existing=times), [])


from test_core_local import LocalPipelineTests  # noqa: E402  (reuse mocked pipeline fixture)


class TriggeredProfilePipelineTests(LocalPipelineTests):
    """ad/talk/podcast run two OCR passes; densified frames sit on both sides."""

    def run_profile(self, profile, duration=300.0, texts=None, signature_values=None):
        texts = texts or {0.0: '第一句字幕内容', 30.0: '第一句字幕内容', 60.0: '完全不同的第二句'}
        signature_values = signature_values or texts
        band_calls, ocr_calls = [], []

        def band(media, times, **kwargs):
            band_calls.append([float(timestamp) for timestamp in times])
            return [(float(timestamp), b'band frame') for timestamp in times]

        def signatures(frames, **kwargs):
            return [(float(timestamp),
                     fake_signature(signature_values.get(round(float(timestamp), 6), '')))
                    for timestamp, _ in frames]

        def recognize(frames, refs):
            ocr_calls.append([float(timestamp) for timestamp, _ in frames])
            items = [{'timestamp': timestamp, 'start': timestamp, 'end': timestamp,
                      'text': texts.get(round(float(timestamp), 6), ''),
                      'status': 'complete', 'evidence_ids': refs.get(timestamp, [])}
                     for timestamp, _ in frames]
            return {'status': 'complete', 'backend': 'tesseract', 'model': 'stub',
                    'params': {}, 'items': items, 'errors': []}

        arguments = list(self.arguments) + ['--profile', profile]
        with patch.object(core, '_probe_video_metadata', return_value={
                'duration_seconds': duration, 'width': 1920, 'height': 1080, 'fps': 24.0}), \
             patch.object(core, 'subtitle_band_times', return_value=[0.0, 30.0, 60.0]), \
             patch.object(core, 'extract_band_frames', side_effect=band), \
             patch.object(core, 'band_signatures', side_effect=signatures), \
             patch('ocr_tesseract.recognize_frames', side_effect=recognize):
            core.main(arguments)
        analysis = json.loads(Path(self.completion()['analysis']).read_text())
        return analysis, band_calls, ocr_calls

    def test_talk_profile_ocrs_only_coarse_and_densified_timestamps(self):
        analysis, band_calls, ocr_calls = self.run_profile('talk')
        self.assertEqual(band_calls, [[0.0, 30.0, 60.0], [59.0, 59.5, 60.5, 61.0]])
        self.assertEqual(ocr_calls, band_calls)
        sampling = analysis['ocr']['sampling']
        self.assertEqual(sampling['strategy'], 'subtitle_trigger')
        self.assertEqual(sampling['coarse_interval'], 30)
        self.assertEqual(sampling['band'], 0.15)
        self.assertEqual(sampling['densified'], 4)
        self.assertEqual(sampling['executed_frame_count'], 7)
        # Seven OCR frames instead of per-second OCR over the same clip.
        self.assertLess(sampling['executed_frame_count'], int(300 / 1))
        self.assertEqual(analysis['capabilities']['ocr']['sampling'], sampling)
        # Tunable trigger thresholds are recorded for future adjustment.
        self.assertEqual(sampling['params']['trigger'], 'subtitle_band_image')
        self.assertEqual(sampling['params']['signature_size'], [64, 16])
        self.assertEqual(sampling['params']['mean_abs_diff_threshold'], 2.0 / 255.0)
        self.assertGreater(sampling['params']['changed_pixel_count_threshold'], 0)
        self.assertEqual(sampling['params']['text_comparison'], 'dedupe_only')

    def test_text_never_triggers_but_dedupes_identical_sentences(self):
        # Images differ at every coarse point (e.g. a moving overlay) while OCR
        # keeps reading one sentence: image changes count, text suppresses the
        # duplicate densified pass.
        analysis, band_calls, _ = self.run_profile(
            'talk', texts={0.0: '同一句字幕', 30.0: '同一句字幕。', 60.0: '  同一句字幕 '},
            signature_values={0.0: 'image-a', 30.0: 'image-b', 60.0: 'image-c'})
        sampling = analysis['ocr']['sampling']
        self.assertEqual(sampling['triggered_changes'], 2)
        self.assertEqual(sampling['text_deduplicated_changes'], 2)
        self.assertEqual(sampling['densified'], 0)
        self.assertEqual(sampling['densified_frames'], 0)
        self.assertEqual(band_calls, [[0.0, 30.0, 60.0]])
        self.assertFalse(validate_analysis(analysis))

    def run_schedule(self, schedule, times, duration=300.0):
        """Run the talk pipeline over explicit coarse times and per-index texts."""
        texts = {round(float(timestamp), 6): schedule(index)
                 for index, timestamp in enumerate(times)}

        def band(media, selected, **kwargs):
            return [(float(timestamp), b'band frame') for timestamp in selected]

        def signatures(frames, **kwargs):
            return [(float(timestamp), fake_signature(texts.get(round(float(timestamp), 6), '')))
                    for timestamp, _ in frames]

        def recognize(frames, refs):
            items = [{'timestamp': timestamp, 'start': timestamp, 'end': timestamp,
                      'text': texts.get(round(float(timestamp), 6), ''),
                      'status': 'complete', 'evidence_ids': refs.get(timestamp, [])}
                     for timestamp, _ in frames]
            return {'status': 'complete', 'backend': 'tesseract', 'model': 'stub',
                    'params': {}, 'items': items, 'errors': []}

        self.messages.clear()
        with patch.object(core, '_probe_video_metadata', return_value={
                'duration_seconds': duration, 'width': 1920, 'height': 1080, 'fps': 24.0}), \
             patch.object(core, 'subtitle_band_times', return_value=list(times)), \
             patch.object(core, 'extract_band_frames', side_effect=band), \
             patch.object(core, 'band_signatures', side_effect=signatures), \
             patch('ocr_tesseract.recognize_frames', side_effect=recognize):
            core.main(list(self.arguments) + ['--profile', 'talk', '--force'])
        analysis = json.loads(Path(self.completion()['analysis']).read_text())
        self.assertFalse(validate_analysis(analysis))
        return analysis['ocr']['sampling']

    def test_change_speed_controls_trigger_count_and_frame_totals(self):
        times = [float(t) for t in range(0, 300, 30)]
        lines = ['春眠不觉晓处处闻啼鸟', '白日依山尽黄河入海流', '床前明月光疑是地上霜',
                 '欲穷千里目更上一层楼', '千山鸟飞绝万径人踪灭']
        slow = self.run_schedule(lambda index: '同一句字幕', times)
        medium = self.run_schedule(lambda index: lines[index // 2], times)
        fast = self.run_schedule(lambda index: lines[index % len(lines)], times)
        self.assertEqual(slow['triggered_changes'], 0)
        self.assertEqual(medium['triggered_changes'], 4)
        self.assertEqual(fast['triggered_changes'], 9)
        self.assertLess(slow['triggered_changes'], medium['triggered_changes'])
        self.assertLess(medium['triggered_changes'], fast['triggered_changes'])
        for sampling in (slow, medium, fast):
            self.assertEqual(sampling['coarse_scans'], 10)
            self.assertEqual(sampling['total_ocr_frames'],
                             sampling['coarse_scans'] + sampling['densified_frames']
                             - sampling['dropped_count'])
            self.assertEqual(sampling['dropped_count'], len(sampling['dropped']))
        self.assertEqual(slow['densified_frames'], 0)

    def test_dropped_sampling_points_are_reported_and_keep_totals_consistent(self):
        def band(media, selected, **kwargs):
            dropped = kwargs.get('dropped')
            frames = []
            for timestamp in selected:
                if round(float(timestamp), 6) == 59.5:
                    dropped.append({'timestamp': 59.5, 'reason': 'ffmpeg_timeout'})
                else:
                    frames.append((float(timestamp), b'band frame'))
            return frames

        texts = {0.0: '第一句字幕内容', 30.0: '第一句字幕内容', 60.0: '完全不同的第二句'}

        def signatures(frames, **kwargs):
            return [(float(timestamp), fake_signature(texts.get(round(float(timestamp), 6), '')))
                    for timestamp, _ in frames]

        def recognize(frames, refs):
            items = [{'timestamp': timestamp, 'start': timestamp, 'end': timestamp,
                      'text': texts.get(round(float(timestamp), 6), ''),
                      'status': 'complete', 'evidence_ids': refs.get(timestamp, [])}
                     for timestamp, _ in frames]
            return {'status': 'complete', 'backend': 'tesseract', 'model': 'stub',
                    'params': {}, 'items': items, 'errors': []}

        with patch.object(core, '_probe_video_metadata', return_value={
                'duration_seconds': 300.0, 'width': 1920, 'height': 1080, 'fps': 24.0}), \
             patch.object(core, 'subtitle_band_times', return_value=[0.0, 30.0, 60.0]), \
             patch.object(core, 'extract_band_frames', side_effect=band), \
             patch.object(core, 'band_signatures', side_effect=signatures), \
             patch('ocr_tesseract.recognize_frames', side_effect=recognize):
            core.main(list(self.arguments) + ['--profile', 'talk'])
        analysis = json.loads(Path(self.completion()['analysis']).read_text())
        sampling = analysis['ocr']['sampling']
        self.assertEqual(sampling['dropped'], [{'timestamp': 59.5, 'reason': 'ffmpeg_timeout'}])
        self.assertEqual(sampling['dropped_count'], 1)
        self.assertEqual(sampling['coarse_scans'], 3)
        self.assertEqual(sampling['densified_frames'], 4)
        self.assertEqual(sampling['total_ocr_frames'], 6)
        self.assertEqual(sampling['total_ocr_frames'],
                         sampling['coarse_scans'] + sampling['densified_frames']
                         - sampling['dropped_count'])
        self.assertFalse(validate_analysis(analysis))

    def test_podcast_profile_uses_sixty_second_coarse_interval(self):
        analysis, _, _ = self.run_profile('podcast')
        self.assertEqual(analysis['ocr']['sampling']['coarse_interval'], 60)

    def test_ad_profile_runs_triggered_scan_at_ten_second_interval(self):
        analysis, band_calls, ocr_calls = self.run_profile('ad')
        sampling = analysis['ocr']['sampling']
        self.assertEqual(sampling['strategy'], 'subtitle_trigger')
        self.assertEqual(sampling['profile'], 'ad')
        self.assertEqual(sampling['coarse_interval'], 10)
        self.assertEqual(band_calls, [[0.0, 30.0, 60.0], [59.0, 59.5, 60.5, 61.0]])
        self.assertEqual(ocr_calls, band_calls)
        self.assertEqual(sampling['params']['densify_window'], 1.0)
        self.assertEqual(sampling['params']['densify_step'], 0.5)

    def test_profile_coarse_scans_scale_with_interval(self):
        # 60 s video with no subtitle changes: ad samples [0..50] plus the 59.5
        # tail (7), talk samples [0, 30] plus 59.5 (3), podcast [0] plus 59.5 (2).
        expected = {'ad': (10, 7), 'talk': (30, 3), 'podcast': (60, 2)}
        for profile, (interval, scans) in expected.items():
            with self.subTest(profile=profile):
                self.messages.clear()
                band_calls = []

                def band(media, selected, **kwargs):
                    band_calls.append([float(timestamp) for timestamp in selected])
                    return [(float(timestamp), b'band frame') for timestamp in selected]

                def signatures(frames, **kwargs):
                    # Identical band images never trigger a densified pass.
                    return [(float(timestamp), fake_signature('same'))
                            for timestamp, _ in frames]

                def recognize(frames, refs):
                    return {'status': 'complete', 'backend': 'tesseract', 'model': 'stub',
                            'params': {}, 'items': [], 'errors': []}

                with patch.object(core, '_probe_video_metadata', return_value={
                        'duration_seconds': 60.0, 'width': 1920, 'height': 1080, 'fps': 24.0}), \
                     patch.object(core, 'extract_band_frames', side_effect=band), \
                     patch.object(core, 'band_signatures', side_effect=signatures), \
                     patch('ocr_tesseract.recognize_frames', side_effect=recognize):
                    core.main(list(self.arguments) + ['--profile', profile, '--force'])
                sampling = json.loads(
                    Path(self.completion()['analysis']).read_text())['ocr']['sampling']
                self.assertEqual(sampling['strategy'], 'subtitle_trigger')
                self.assertEqual(sampling['coarse_interval'], interval)
                self.assertEqual(sampling['coarse_scans'], scans)
                self.assertEqual(sampling['coarse_scans'], len(band_calls[0]))
                self.assertEqual(band_calls[0],
                                 subtitle_band_times(None, interval=interval, band=0.15,
                                                     duration=60.0))
                self.assertEqual(sampling['total_ocr_frames'], scans)
                self.assertEqual(sampling['params']['densify_window'], 1.0)
                self.assertEqual(sampling['params']['densify_step'], 0.5)

    def test_auto_profile_keeps_legacy_sample_shape(self):
        core.main(self.arguments)
        analysis = json.loads(Path(self.completion()['analysis']).read_text())
        sampling = analysis['ocr']['sampling']
        self.assertTrue(sampling['independent_sampling'])
        self.assertEqual(sampling['executed_timestamps'], [2.5])
        self.assertNotIn('profile', sampling)


if __name__ == '__main__':
    unittest.main()
