"""Local-file pipeline contracts; no downloads, model inference or real media fixtures."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
import core
from asset_contract import empty_analysis, validate_analysis
from asset_store import AssetStore


class LocalInputTests(unittest.TestCase):
    def setUp(self):
        guards = ExitStack()
        self.addCleanup(guards.close)
        self.network = guards.enter_context(patch(
            'requests.sessions.Session.request', side_effect=AssertionError('network access attempted')))
        self.process = guards.enter_context(patch.object(
            core.subprocess, 'run', side_effect=AssertionError('external process attempted')))

    def test_supported_local_files_resolve_without_network_or_processes(self):
        with tempfile.TemporaryDirectory() as td:
            for suffix in ('.mp4', '.MOV', '.mkv'):
                with self.subTest(suffix=suffix):
                    source = Path(td) / f'合成 视频{suffix}'
                    source.write_bytes(b'synthetic local fixture')
                    for reference in (str(source), os.path.relpath(source)):
                        self.assertEqual(core.resolve_input(reference), str(source.resolve()))
                    self.assertEqual(source.read_bytes(), b'synthetic local fixture')
        self.network.assert_not_called()
        self.process.assert_not_called()

    def test_url_schemes_are_rejected_before_filesystem_resolution(self):
        references = (
            'https://invalid.example/video.mp4', '  HTTP://invalid.example/video.MOV',
            'ftp://invalid.example/video.mkv', 'file:///tmp/video.mp4',
            'rtsp://invalid.example/live', 'custom+video://invalid.example/clip',
        )
        with patch.object(core.Path, 'expanduser', side_effect=AssertionError('URL reached filesystem')) as expand:
            for reference in references:
                with self.subTest(reference=reference), self.assertRaisesRegex(ValueError, '仅支持本地'):
                    core.resolve_input(reference)
            expand.assert_not_called()
        self.network.assert_not_called()
        self.process.assert_not_called()

    def test_missing_directory_and_unsupported_inputs_have_distinct_errors(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(FileNotFoundError, '文件不存在'):
                core.resolve_input(str(root / 'missing.mp4'))
            directory = root / 'folder.mp4'
            directory.mkdir()
            with self.assertRaisesRegex(ValueError, '必须是本地文件'):
                core.resolve_input(str(directory))
            unsupported = root / 'clip.avi'
            unsupported.write_bytes(b'synthetic fixture')
            with self.assertRaisesRegex(ValueError, '不支持的格式'):
                core.resolve_input(str(unsupported))
        self.network.assert_not_called()
        self.process.assert_not_called()

    def test_main_rejects_url_before_archive_or_analysis(self):
        with patch.object(core, 'check_dependencies', return_value=[]), \
             patch.object(core, 'emit_progress'), patch.object(core, 'emit_error') as error, \
             patch.object(core, 'AssetStore') as store, \
             patch.object(core, 'transcribe_audio') as transcribe, \
             patch.object(core, 'analyze_audio_isolated') as music, \
             patch.object(core, 'detect_shots') as shots:
            with self.assertRaises(SystemExit) as caught:
                core.main(['--input', 'https://invalid.example/video.mp4'])
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(error.call_args.args[:2], ('core', 'PIPELINE_ERROR'))
        self.assertIn('仅支持本地', error.call_args.args[2])
        for stage in (store, transcribe, music, shots, self.network, self.process):
            stage.assert_not_called()


class LocalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / '合成 clip.mp4'
        self.original = b'synthetic bytes: no real video or model required'
        self.source.write_bytes(self.original)
        self.library = self.root / 'library'
        self.arguments = ['--input', str(self.source), '--output-dir', str(self.root / 'output'),
                          '--library-root', str(self.library), '--ocr-backend', 'tesseract',
                          '--ocr-max-width', '960']
        guards = ExitStack()
        self.addCleanup(guards.close)
        self.messages = []
        guards.enter_context(patch.object(core, 'emit', side_effect=lambda item: self.messages.append(dict(item))))
        guards.enter_context(patch.object(core, 'check_dependencies', return_value=[]))
        self.network = guards.enter_context(patch(
            'requests.sessions.Session.request', side_effect=AssertionError('network access attempted')))
        self.process = guards.enter_context(patch.object(
            core.subprocess, 'run', side_effect=AssertionError('external process attempted')))
        self.probe = guards.enter_context(patch.object(core, '_probe_video_metadata', return_value={
            'duration_seconds': 10.0, 'width': 2560, 'height': 1440, 'fps': 24.0, 'codec': 'synthetic',
        }))
        self.audio_track = guards.enter_context(patch.object(core, 'probe_audio_track', return_value={'status': 'complete', 'present': True}))
        self.preflight = guards.enter_context(patch.object(core, 'check_transcription_model'))
        self.video = guards.enter_context(patch.object(core, 'has_video_track', return_value=True))
        self.shots = guards.enter_context(patch.object(core, 'detect_shots', return_value=(
            {'status': 'complete', 'backend': 'synthetic-shot-stub', 'items': []},
            [{'evidence_id': 'shot_proxy', 'kind': 'image', 'timestamp': 0,
              'relative_path': 'evidence/shot_proxy.jpg'}],
        )))
        self.music_result = {'status': 'complete', 'backend': 'librosa', 'tempo_bpm': 120,
                             'beat_times_seconds': [1.0], 'genre': {'status': 'not_run'}}
        self.music = guards.enter_context(patch.object(core, 'analyze_audio_isolated', return_value=self.music_result))
        self.transcribe = guards.enter_context(patch.object(core, 'transcribe_audio', return_value=(
            [{'start': 1.0, 'end': 2.0, 'text': 'synthetic speech'}],
            {'language': 'en', 'language_probability': .99, 'params': {}},
        )))
        self.frames = guards.enter_context(patch.object(core, 'extract_frames', return_value=[(2.5, b'synthetic jpeg')]))
        self.ocr = guards.enter_context(patch('ocr_tesseract.recognize_frames', return_value={
            'status': 'complete', 'backend': 'tesseract', 'model': 'synthetic-ocr-stub',
            'params': {}, 'items': [], 'errors': [],
        }))
        self.correction = guards.enter_context(patch.object(core, 'correct_transcript'))
        self.summary = guards.enter_context(patch.object(core, 'generate_summary'))
        self.rewrite = guards.enter_context(patch('transcript_rewrite.rewrite_transcript', return_value={
            'status':'not_run','reason':'fixture local model unavailable','text':'','segments':[],
            'edits':[],'params':{'mode':'off'}}))

    def completion(self):
        completions = [item for item in self.messages if item['type'] == 'complete']
        self.assertEqual(len(completions), 1)
        return completions[0]['output']

    def assert_no_external_work(self):
        self.network.assert_not_called()
        self.process.assert_not_called()
        self.correction.assert_not_called()
        self.summary.assert_not_called()

    def test_local_pipeline_archives_exact_bytes_and_samples_original_media(self):
        core.main(self.arguments)
        output = self.completion()
        self.assertIs(output['cached'], False)
        self.assertEqual(output['video'], str(self.source))
        asset = AssetStore(self.library).get_asset(output['asset_id'])
        self.assertEqual(asset['asset']['source']['kind'], 'local_file')
        self.assertEqual(asset['asset']['source']['display_name'], self.source.name)
        self.assertEqual(Path(asset['media']['local_path']).read_bytes(), self.original)
        self.assertEqual(self.source.read_bytes(), self.original)
        analysis = json.loads(Path(output['analysis']).read_text())
        self.assertEqual(analysis['source']['kind'], 'local_file')
        self.assertIsNone(analysis['source']['reference'])
        self.assertEqual(validate_analysis(analysis), [])
        self.shots.assert_called_once()
        self.assertEqual(self.shots.call_args.args[0], str(self.source))
        self.assertIn('.attempts', str(self.shots.call_args.args[1]))
        self.frames.assert_called_once_with(str(self.source), max_width=960)
        self.ocr.assert_called_once_with([(2.5, b'synthetic jpeg')], {2.5: ['ocr_frame_00000']})
        sampling = analysis['ocr']['sampling']
        self.assertEqual(sampling['frame_source'], 'original_media')
        self.assertEqual(sampling['max_width'], 960)
        self.assertEqual(sampling['jpeg_quality'], 95)
        self.assertEqual(sampling['executed_frame_count'], 1)
        self.assertEqual(sampling['executed_timestamps'], [2.5])
        self.assertTrue(sampling['independent_sampling'])
        self.assertEqual(analysis['ocr']['status'], 'complete')
        self.assertEqual(analysis['music']['tempo_bpm'], 120)
        self.assertEqual(analysis['transcript']['raw_segments'][0]['text'], 'synthetic speech')
        self.assert_no_external_work()

    def test_complete_cached_package_emits_boolean_without_running_analysis(self):
        store = AssetStore(self.library)
        archive = store.import_file(self.source)
        analysis = empty_analysis(archive['asset_id'], {'kind': 'local_file'})
        analysis['status'] = 'complete'
        from pipeline_state import run_signature
        from asset_store import sha256_file
        analysis['run_signature'] = run_signature(sha256_file(self.source), {
            'ocr_backend': 'tesseract', 'ocr_max_width': 960})
        for capability, section in [('transcription', 'transcript'), ('shots', 'shots'), ('ocr', 'ocr')]:
            analysis['capabilities'][capability]['status'] = 'complete'
            analysis[section]['status'] = 'complete'
        store.save_analysis(archive['asset_id'], analysis)
        before = Path(archive['package_dir'], 'analysis.json').read_bytes()
        core.main(self.arguments)
        output = self.completion()
        self.assertIs(output['cached'], True)
        self.assertEqual(output['asset_id'], archive['asset_id'])
        self.assertEqual(output['package'], archive['package_dir'])
        self.assertEqual(Path(output['video']).read_bytes(), self.original)
        self.assertEqual(Path(output['analysis']).read_bytes(), before)
        self.assertTrue(Path(output['markdown']).is_file())
        for stage in (self.video, self.shots, self.music, self.transcribe, self.frames, self.ocr, self.probe):
            stage.assert_not_called()
        self.assert_no_external_work()


if __name__ == '__main__':
    unittest.main()
