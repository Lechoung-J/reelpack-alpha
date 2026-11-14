"""Global frame board and content_hints engine contracts.

The board tests synthesize a 30s colour-bar clip with the local ffmpeg; no
model, network or downloaded fixture is involved.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))

import core
from asset_store import AssetStore
from global_board import build_global_board


FFMPEG = shutil.which('ffmpeg')
FFPROBE = shutil.which('ffprobe')


@unittest.skipUnless(FFMPEG and FFPROBE, 'ffmpeg and ffprobe are required')
class GlobalBoardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.video = Path(cls.temp.name) / 'bars.mp4'
        subprocess.run(
            [FFMPEG, '-nostdin', '-v', 'error', '-f', 'lavfi',
             '-i', 'smptebars=size=320x240:rate=10', '-t', '30',
             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', str(cls.video)],
            check=True, capture_output=True, text=True, timeout=120,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_sixteen_uniform_frames_build_one_labelled_board(self):
        out = Path(self.temp.name) / 'board' / 'frame-board.jpg'
        result = build_global_board(self.video, out, frames=16, grid_cols=4)
        self.assertEqual(result['status'], 'complete', result.get('reason'))
        self.assertEqual(result['grid']['cols'], 4)
        self.assertEqual(result['grid']['rows'], 4)
        self.assertEqual(result['grid']['count'], 16)
        self.assertEqual(len(result['frames']), 16)
        self.assertEqual([frame['index'] for frame in result['frames']], list(range(16)))
        timestamps = [frame['timestamp'] for frame in result['frames']]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertGreater(timestamps[0], 0)
        self.assertLess(timestamps[-1], 30)
        path = Path(result['path'])
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 0)
        from PIL import Image
        with Image.open(path) as image:
            self.assertEqual(image.size, (4 * 320, 4 * (180 + 18)))

    def test_frame_count_is_clamped_to_twelve_through_twenty_four(self):
        low = build_global_board(self.video, Path(self.temp.name) / 'low.jpg', frames=4)
        high = build_global_board(self.video, Path(self.temp.name) / 'high.jpg', frames=99)
        self.assertEqual(low['status'], 'complete', low.get('reason'))
        self.assertEqual(high['status'], 'complete', high.get('reason'))
        self.assertEqual(low['grid']['count'], 12)
        self.assertEqual(high['grid']['count'], 24)

    def test_missing_media_returns_failed_without_raising(self):
        out = Path(self.temp.name) / 'missing-board.jpg'
        result = build_global_board(Path(self.temp.name) / 'missing.mp4', out)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('不存在', result['reason'])
        self.assertEqual(result['frames'], [])
        self.assertIsNone(result['path'])
        self.assertFalse(out.exists())

    def test_undecodable_media_returns_failed_without_raising(self):
        broken = Path(self.temp.name) / 'broken.mp4'
        broken.write_bytes(b'not a video')
        result = build_global_board(broken, Path(self.temp.name) / 'broken-board.jpg')
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['reason'])
        self.assertEqual(result['frames'], [])


class ContentHintsTests(unittest.TestCase):
    def test_synthetic_sample_produces_three_plausible_ratios(self):
        shots = {'status': 'complete', 'items': [
            {'shot_id': f'shot_{index}', 'start': index * 50, 'end': index * 50 + 40}
            for index in range(6)
        ]}
        segments = [{'start': 0, 'end': 60, 'text': 'a'}, {'start': 120, 'end': 180, 'text': 'b'}]
        ocr_items = [
            {'timestamp': 1, 'text': 'A'}, {'timestamp': 2, 'text': ' A '},
            {'timestamp': 3, 'text': 'B'}, {'timestamp': 4, 'text': 'B'},
            {'timestamp': 5, 'text': 'C'},
        ]
        hints = core.compute_content_hints(
            shots=shots, segments=segments, ocr_items=ocr_items,
            duration_seconds=300, frame_board='assets/frame-board.jpg')
        self.assertAlmostEqual(hints['shots_per_min'], 1.2, places=4)
        self.assertAlmostEqual(hints['speech_ratio'], 0.4, places=4)
        self.assertAlmostEqual(hints['ocr_change_rate'], 0.4, places=4)
        self.assertEqual(hints['frame_board'], 'assets/frame-board.jpg')
        self.assertEqual(hints['duration_seconds'], 300)

    def test_overlapping_segments_are_not_double_counted(self):
        hints = core.compute_content_hints(
            segments=[{'start': 0, 'end': 30}, {'start': 20, 'end': 40}], duration_seconds=100)
        self.assertAlmostEqual(hints['speech_ratio'], 0.4, places=4)

    def test_missing_inputs_stay_null_instead_of_zero(self):
        hints = core.compute_content_hints(
            shots={'status': 'not_run', 'items': []}, segments=[], ocr_items=[],
            duration_seconds=None, frame_board=None)
        for key in ('shots_per_min', 'speech_ratio', 'ocr_change_rate', 'frame_board', 'duration_seconds'):
            self.assertIsNone(hints[key], key)

    def test_blank_ocr_text_is_null_and_single_entry_is_zero_change(self):
        blank = core.compute_content_hints(ocr_items=[{'timestamp': 1, 'text': '   '}], duration_seconds=10)
        self.assertIsNone(blank['ocr_change_rate'])
        single = core.compute_content_hints(ocr_items=[{'timestamp': 1, 'text': 'only'}], duration_seconds=10)
        self.assertEqual(single['ocr_change_rate'], 0.0)

    def test_invalid_duration_keeps_duration_dependent_fields_null(self):
        hints = core.compute_content_hints(
            shots={'status': 'complete', 'items': [{'shot_id': 's', 'start': 0, 'end': 1}]},
            segments=[{'start': 0, 'end': 1}], duration_seconds='not-a-number')
        self.assertIsNone(hints['shots_per_min'])
        self.assertIsNone(hints['speech_ratio'])
        self.assertIsNone(hints['duration_seconds'])

    def test_pipeline_analysis_records_hints_and_existing_board(self):
        with tempfile.TemporaryDirectory() as td:
            store = AssetStore(Path(td) / 'library')
            source = Path(td) / 'clip.mp4'
            source.write_bytes(b'synthetic fixture')
            archive = store.import_file(source)
            board = Path(archive['package_dir']) / 'assets' / 'frame-board.jpg'
            board.parent.mkdir(parents=True)
            board.write_bytes(b'fake board bytes')
            args = SimpleNamespace(model='large', ocr_backend='tesseract', vision_model='mock')
            with patch.object(core, '_probe_video_metadata', return_value={'duration_seconds': 300}):
                analysis = core._save_pipeline_analysis(
                    store, archive, str(source), str(source),
                    [{'start': 0, 'end': 60, 'text': 'a'}], 'a', 'a', '', '', args,
                    shots={'status': 'complete', 'items': [
                        {'shot_id': 'shot_0001', 'start': 0, 'end': 10}]})
            hints = analysis['content_hints']
            self.assertEqual(hints['frame_board'], 'assets/frame-board.jpg')
            self.assertAlmostEqual(hints['shots_per_min'], 0.2, places=4)
            self.assertAlmostEqual(hints['speech_ratio'], 0.2, places=4)
            self.assertIsNone(hints['ocr_change_rate'])
            self.assertEqual(hints['duration_seconds'], 300)
            saved = json.loads((Path(archive['package_dir']) / 'analysis.json').read_text())
            self.assertEqual(saved['content_hints'], hints)


if __name__ == '__main__':
    unittest.main()
