"""Original-media OCR extraction contracts with synthetic pixels and mock decoders."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
import core


class OCRFrameTests(unittest.TestCase):
    source = '/synthetic/original.mp4'

    def capture(self, readable=True, fps=24, count=240, frame=None):
        cap = Mock()
        cap.isOpened.return_value = readable
        cap.get.side_effect = lambda prop: {cv2.CAP_PROP_FPS: fps,
                                           cv2.CAP_PROP_FRAME_COUNT: count}[prop]
        cap.read.return_value = (True, frame) if readable else (False, None)
        return cap

    def test_small_sources_are_not_upscaled_and_large_sources_respect_cap(self):
        for width, height, maximum, expected in (
            (640, 360, 1920, (640, 360)),
            (2560, 1440, 1920, (1920, 1080)),
            (2560, 1440, 0, (2560, 1440)),
            (1279, 719, 960, (960, 540)),
        ):
            with self.subTest(source_size=(width, height), max_width=maximum):
                image = np.full((height, width, 3), 100, dtype=np.uint8)
                cap = self.capture(count=48, frame=image)
                with patch.object(core.cv2, 'VideoCapture', return_value=cap) as opened, \
                     patch.object(core.cv2, 'resize', wraps=cv2.resize) as resized, \
                     patch.object(core.cv2, 'imencode', wraps=cv2.imencode) as encoded, \
                     patch.object(core, '_compute_phash', return_value=np.zeros(64)), \
                     patch.object(core.subprocess, 'run', side_effect=AssertionError('unexpected fallback')):
                    frames = core.extract_frames(self.source, interval=2, max_width=maximum)
                opened.assert_called_once_with(self.source)
                cap.release.assert_called_once()
                self.assertEqual([timestamp for timestamp, _ in frames], [1.0])
                decoded = cv2.imdecode(np.frombuffer(frames[0][1], dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual((decoded.shape[1], decoded.shape[0]), expected)
                if maximum and width > maximum:
                    resized.assert_called_once()
                    self.assertEqual(resized.call_args.args[1], expected)
                    self.assertEqual(resized.call_args.kwargs['interpolation'], cv2.INTER_AREA)
                else:
                    resized.assert_not_called()
                self.assertEqual(encoded.call_args.args[0], '.jpg')
                self.assertEqual(encoded.call_args.args[2], [cv2.IMWRITE_JPEG_QUALITY, 95])

    def run_timestamps(self, readable, read_results=None):
        image = np.zeros((18, 32, 3), dtype=np.uint8)
        cap = self.capture(readable=readable, frame=image)
        if read_results is not None:
            cap.read.side_effect = read_results
        hashes = [np.full(64, index) for index in range(3)]
        with patch.object(core.cv2, 'VideoCapture', return_value=cap), \
             patch.object(core, '_probe_video_metadata', return_value={'duration_seconds': 10}) as probe, \
             patch.object(core, '_compute_phash', side_effect=hashes), \
             patch.object(core.cv2, 'imdecode', return_value=image), \
             patch.object(core.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, b'png', b'')) as fallback:
            frames = core.extract_frames(self.source, interval=2, max_frames=3)
        cap.release.assert_called_once()
        if readable:
            probe.assert_not_called()
            self.assertEqual([call.args[1] for call in cap.set.call_args_list], [40, 120, 200])
        else:
            probe.assert_called_once_with(self.source)
            cap.read.assert_not_called()
            cap.set.assert_not_called()
        for call in fallback.call_args_list:
            command = call.args[0]
            self.assertEqual(command[command.index('-i') + 1], self.source)
            self.assertEqual(command[command.index('-frames:v') + 1], '1')
            self.assertEqual(command[command.index('-c:v') + 1], 'png')
            self.assertNotIn('-vf', command)
            self.assertTrue(call.kwargs['check'])
        sought = [float(call.args[0][call.args[0].index('-ss') + 1]) for call in fallback.call_args_list]
        return [timestamp for timestamp, _ in frames], sought

    def test_ffmpeg_fallback_and_opencv_use_identical_half_interval_timestamps(self):
        direct, direct_fallback = self.run_timestamps(True)
        fallback, sought = self.run_timestamps(False)
        self.assertEqual(direct_fallback, [])
        self.assertEqual(fallback, direct)
        self.assertEqual(sought, direct)
        self.assertEqual(len(direct), 3)
        for actual, expected in zip(direct, (10 / 6, 5.0, 50 / 6)):
            self.assertAlmostEqual(actual, expected)

    def test_single_opencv_read_failure_falls_back_at_that_same_timestamp(self):
        image = np.zeros((18, 32, 3), dtype=np.uint8)
        timestamps, sought = self.run_timestamps(True, [(True, image), (False, None), (True, image)])
        self.assertEqual(sought, [timestamps[1]])
        self.assertEqual(timestamps[1], 5.0)

    def test_duplicate_frames_keep_first_sampling_time(self):
        image = np.zeros((18, 32, 3), dtype=np.uint8)
        cap = self.capture(frame=image)
        with patch.object(core.cv2, 'VideoCapture', return_value=cap), \
             patch.object(core, '_compute_phash', return_value=np.zeros(64)), \
             patch.object(core.subprocess, 'run') as fallback:
            frames = core.extract_frames(self.source, interval=2, max_frames=3)
        self.assertEqual(len(frames), 1)
        self.assertAlmostEqual(frames[0][0], 10 / 6)
        self.assertEqual(cap.read.call_count, 3)
        cap.release.assert_called_once()
        fallback.assert_not_called()

    def test_invalid_sampling_parameters_do_not_open_media(self):
        with patch.object(core.cv2, 'VideoCapture') as opened:
            for kwargs in ({'interval': 0}, {'max_frames': 0}, {'max_width': -1}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    core.extract_frames(self.source, **kwargs)
        opened.assert_not_called()

    def test_unknown_duration_and_decode_failures_release_capture(self):
        for duration, output, error in (
            (0, None, '无法读取原片时长'),
            (10, None, '无法解码原片 OCR 帧'),
        ):
            with self.subTest(duration=duration):
                cap = self.capture(readable=False)
                with patch.object(core.cv2, 'VideoCapture', return_value=cap), \
                     patch.object(core, '_probe_video_metadata', return_value={'duration_seconds': duration}), \
                     patch.object(core.cv2, 'imdecode', return_value=output), \
                     patch.object(core.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, b'png', b'')):
                    with self.assertRaisesRegex(RuntimeError, error):
                        core.extract_frames(self.source)
                cap.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
