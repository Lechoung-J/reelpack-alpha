import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
import core
from asset_contract import empty_analysis, validate_analysis
from asset_store import AssetStore
from shot_detect import detect_shots_isolated, detect_shots


class OCRPipelineAndShotWorkerTests(unittest.TestCase):
    def test_native_ocr_frame_error_propagates(self):
        payload = {'frames': [{'text': '', 'error': "Failed CVPixelBuffer 420f"}]}
        result = SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr='IOSurface failure')
        with patch.object(core.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(RuntimeError, '420f'):
                core.describe_frames_macos([(1., b'jpeg')], 'helper')

    def test_native_ocr_malformed_and_count_mismatch(self):
        for output in ['not JSON', '{"frames": []}', '{"frames": [null]}']:
            result = SimpleNamespace(returncode=0, stdout=output, stderr='diagnostic')
            with patch.object(core.subprocess, 'run', return_value=result):
                with self.assertRaises(RuntimeError):
                    core.describe_frames_macos([(0, b'jpeg')], 'helper')

    def test_native_ocr_multiline_and_precise_time(self):
        result = SimpleNamespace(returncode=0, stdout=json.dumps({'frames': [{'text': '第一行\n第二行', 'error': None}]}), stderr='')
        with patch.object(core.subprocess, 'run', return_value=result):
            self.assertEqual(core.describe_frames_macos([(1.234, b'jpeg')], 'helper'), [(1.234, '第一行\n第二行')])

    def test_optional_ocr_nonblocking(self):
        for exc in [RuntimeError('420f'), FileNotFoundError('helper'), subprocess.TimeoutExpired('helper', 300)]:
            with patch.object(core, 'describe_frames_macos', side_effect=exc):
                items, error = core.optional_macos_ocr([], 'helper')
                self.assertEqual(items, [])
                self.assertIn(type(exc).__name__, error)

    def save_fixture(self, root, segments=None, **overrides):
        source = Path(root) / 'source.mp4'; source.write_bytes(b'fixture')
        store = AssetStore(Path(root) / 'library'); archive = store.import_file(source)
        args = SimpleNamespace(model='large', language='ja', vision_model='gemma4:12b',
                               ocr_backend='macos', skip_summary=False, summary_backend='ollama', **overrides)
        segments = segments or [{'start': 0, 'end': 1, 'text': 'speech'}]
        with patch.object(core, '_probe_video_metadata', return_value={'duration_seconds': 2}):
            core._save_pipeline_analysis(store, archive, str(source), str(source),
                segments, 'speech', 'speech', '', '', args,
                shots={'status': 'complete', 'backend': 'real-test-backend', 'items': []}, music={'status': 'complete'})
        return store.get_asset(archive['asset_id'])['analysis']

    def test_failed_ocr_and_summary_persist_without_losing_other_stages(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.save_fixture(td, ocr_error='Vision internal 420f allocation failed')
            self.assertEqual(a['ocr']['status'], 'failed')
            self.assertIn('420f', a['ocr']['reason'])
            self.assertEqual(a['capabilities']['ocr']['status'], 'failed')
            self.assertEqual(a['summary']['status'], 'failed')
            self.assertEqual(a['summary']['text'], '')
            self.assertIn('sandbox', a['summary']['reason'])
            self.assertEqual(a['music']['status'], 'complete')
            self.assertEqual(a['transcript']['raw_segments'][0]['text'], 'speech')

    def test_successful_empty_ocr_is_not_failure(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.save_fixture(td, ocr_attempted=True, ocr_observations=[])
            self.assertEqual(a['ocr']['status'], 'complete')
            self.assertEqual(a['ocr']['items'], [])

    def test_structured_ocr_preserves_multiline_precision(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.save_fixture(td, ocr_attempted=True, ocr_observations=[
                {'timestamp': 1.234, 'text': 'one\ntwo', 'evidence_ids': []}])
            self.assertEqual(a['ocr']['items'][0]['text'], 'one\ntwo')
            self.assertEqual(a['ocr']['items'][0]['start'], 1.234)

    def test_ocr_items_carry_pixel_and_normalized_regions(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.save_fixture(td, ocr_attempted=True,
                ocr_frame_sizes={1.5: (100, 50)},
                ocr_observations=[
                    {'timestamp': 1.5, 'text': '字幕甲',
                     'region': {'x': 10, 'y': 5, 'width': 40, 'height': 10, 'units': 'pixels'}},
                    {'timestamp': 1.5, 'text': '字幕乙'},
                ])
            first, second = a['ocr']['items']
            self.assertEqual((first['region']['x'], first['region']['y'],
                              first['region']['w'], first['region']['h']), (10, 5, 40, 10))
            self.assertEqual(first['region']['width'], 40)
            self.assertEqual(first['region_norm'], {'x': 0.1, 'y': 0.1, 'w': 0.4, 'h': 0.2})
            self.assertIsNone(second['region'])
            self.assertIsNone(second['region_norm'])
            self.assertFalse(validate_analysis(a))

    def test_ocr_declares_per_backend_region_support(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.save_fixture(td, ocr_attempted=True,
                ocr_observations=[{'timestamp': 1.0, 'text': '无坐标字幕'}])
            # Null region means "backend does not report coordinates", not failure.
            self.assertIsNone(a['ocr']['items'][0]['region'])
            self.assertEqual(a['ocr']['region_support'],
                             {'tesseract': True, 'rapidocr': 'unverified',
                              'macos': False, 'ollama': False})
            self.assertFalse(validate_analysis(a))

    def test_ocr_conflicts_mark_but_never_rewrite(self):
        with tempfile.TemporaryDirectory() as td:
            ocr_text = '完全不同的一段画面文字内容'
            transcript_text = '今天我们讨论摄影画面的构图'
            a = self.save_fixture(td,
                segments=[{'start': 0, 'end': 2, 'text': transcript_text}],
                ocr_attempted=True,
                ocr_observations=[{'timestamp': 1.0, 'start': 0, 'end': 2, 'text': ocr_text}])
            self.assertEqual(len(a['ocr_conflicts']), 1)
            conflict = a['ocr_conflicts'][0]
            self.assertLess(conflict['similarity'], 0.6)
            self.assertEqual(conflict['ocr_text'], ocr_text)
            self.assertEqual(conflict['transcript_text'], transcript_text)
            self.assertEqual(a['ocr']['items'][0]['text'], ocr_text)
            self.assertEqual(a['transcript']['raw_segments'][0]['text'], transcript_text)
            self.assertFalse(validate_analysis(a))

    def test_old_package_without_region_fields_migrates_to_null(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'old.mp4'; source.write_bytes(b'old fixture')
            store = AssetStore(Path(td) / 'library'); archive = store.import_file(source)
            analysis = empty_analysis(archive['asset_id'], {'kind': 'local_file'})
            analysis['ocr']['items'] = [{'ocr_id': 'ocr_1', 'timestamp': 1, 'start': 1,
                                         'end': 1, 'text': '旧包字幕'}]
            analysis.pop('ocr_conflicts')
            store.save_analysis(archive['asset_id'], analysis)
            loaded = store.get_asset(archive['asset_id'])['analysis']
            item = loaded['ocr']['items'][0]
            self.assertIsNone(item['region'])
            self.assertIsNone(item['region_norm'])
            self.assertEqual(loaded['ocr']['region_support'],
                             {'tesseract': True, 'rapidocr': 'unverified',
                              'macos': False, 'ollama': False})
            self.assertEqual(loaded['ocr_conflicts'], [])
            self.assertEqual(loaded['ocr']['items'][0]['text'], '旧包字幕')

    def test_transnet_worker_reads_result_file_not_noisy_stdout(self):
        def worker(command, **kwargs):
            Path(command[-1]).write_text(json.dumps({'shots': {'status': 'complete'}, 'evidence': []}))
            return SimpleNamespace(returncode=0, stdout='model logging', stderr='')
        with patch('shot_detect.subprocess.run', side_effect=worker):
            shots, frames = detect_shots_isolated('python', 'input', 'output')
        self.assertEqual(shots['status'], 'complete')
        self.assertEqual(frames, [])

    def test_transnet_worker_error_is_truthful(self):
        with patch('shot_detect.subprocess.run', side_effect=FileNotFoundError('python')):
            shots, frames = detect_shots_isolated('missing-python', 'input', 'output')
        self.assertEqual(shots['status'], 'failed')
        self.assertIn('python', shots['reason'])
        self.assertEqual(frames, [])

    def test_configured_transnet_worker_is_used(self):
        expected = ({'status': 'complete', 'backend': 'transnetv2-pytorch-cpu'}, [])
        with patch.dict('os.environ', {'REELPACK_TRANSNET_PYTHON': '/worker/python'}), patch(
                'shot_detect.detect_shots_isolated', return_value=expected) as worker:
            self.assertEqual(detect_shots('video', 'package'), expected)
            self.assertEqual(worker.call_args.args[0], '/worker/python')
