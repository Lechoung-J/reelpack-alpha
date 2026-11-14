"""Pipeline-level speaker contract: validation, migration, clamping and failures.

No CAM++ model, sherpa-onnx or ffmpeg is required here: the model-missing path
short-circuits before decoding, and the remaining tests inject the backend.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Core'))
import core
import speaker_campp
from asset_contract import empty_analysis, migrate_speakers, validate_analysis
from asset_store import AssetStore
from pipeline_state import run_signature


def valid_speakers():
    return {
        'status': 'complete',
        'backend': 'campp_sherpa_onnx+kmeans',
        'model': '3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx',
        'experimental': True,
        'segments': [
            {'start': 0.0, 'end': 3.2, 'speaker': 'S1', 'text': 'hello'},
            {'start': 3.2, 'end': 6.0, 'speaker': 'S2', 'text': 'hi'},
        ],
        'confidence': {'selected_k': 2, 'silhouette': 0.45},
    }


def analysis(speakers=None):
    value = empty_analysis('asset_synthetic', {'kind': 'local_file'})
    if speakers is not None:
        value['speakers'] = speakers
    return value


class SpeakerContractTests(unittest.TestCase):
    def test_default_analysis_has_no_speakers_block(self):
        # An off run must not gain a speakers field.
        self.assertNotIn('speakers', analysis())
        self.assertEqual(validate_analysis(analysis()), [])

    def test_valid_complete_block_passes(self):
        self.assertEqual(validate_analysis(analysis(valid_speakers())), [])

    def test_complete_requires_segments_array(self):
        broken = valid_speakers()
        del broken['segments']
        self.assertTrue(any('speakers.segments' in error
                            for error in validate_analysis(analysis(broken))))

    def test_non_complete_states_accept_empty_segments(self):
        for status in ('not_run', 'failed', 'partial'):
            with self.subTest(status=status):
                block = {'status': status, 'backend': 'campp_sherpa_onnx+kmeans',
                         'model': None, 'experimental': True, 'segments': [],
                         'confidence': {}, 'reason': 'because'}
                self.assertEqual(validate_analysis(analysis(block)), [])

    def test_failed_requires_reason(self):
        block = {'status': 'failed', 'segments': []}
        self.assertTrue(any('reason' in error for error in validate_analysis(analysis(block))))

    def test_malformed_shapes_are_reported(self):
        cases = []
        cases.append('not-an-object')
        block = valid_speakers(); block['status'] = 'maybe'; cases.append(block)
        block = valid_speakers(); block['segments'] = {}; cases.append(block)
        block = valid_speakers(); block['segments'][0]['start'] = None; cases.append(block)
        block = valid_speakers(); block['segments'][0]['end'] = -1; cases.append(block)
        block = valid_speakers(); block['segments'][0]['speaker'] = 'alice'; cases.append(block)
        block = valid_speakers(); block['segments'][0]['text'] = 3; cases.append(block)
        block = valid_speakers(); block['segments'][0] = 'text'; cases.append(block)
        block = valid_speakers(); block['confidence'] = []; cases.append(block)
        block = valid_speakers(); block['backend'] = 7; cases.append(block)
        for index, block in enumerate(cases):
            with self.subTest(case=index):
                self.assertNotEqual(validate_analysis(analysis(block)), [])

    def test_migration_adds_not_run_without_touching_existing_blocks(self):
        old = analysis()
        migrated = migrate_speakers(old)
        self.assertEqual(migrated['speakers'], {'status': 'not_run'})
        self.assertEqual(validate_analysis(migrated), [])
        existing = analysis(valid_speakers())
        before = copy.deepcopy(existing)
        self.assertEqual(migrate_speakers(existing), before)

    def test_old_package_read_is_migrated_on_disk_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            store = AssetStore(Path(td) / 'library')
            source = Path(td) / 'old.mp4'
            source.write_bytes(b'old package media')
            imported = store.import_file(source)
            old = empty_analysis(imported['asset_id'], {'kind': 'local_file'})
            store.save_analysis(imported['asset_id'], old)
            path = Path(imported['package_dir']) / 'analysis.json'
            before = path.read_bytes()
            self.assertNotIn('speakers', json.loads(before))
            loaded = store.get_asset(imported['asset_id'])['analysis']
            self.assertEqual(loaded['speakers'], {'status': 'not_run'})
            self.assertEqual(path.read_bytes(), before)


class SpeakerSignatureTests(unittest.TestCase):
    def test_off_is_absent_from_the_signature(self):
        base = run_signature('a' * 64, {})
        self.assertEqual(base, run_signature('a' * 64, {'speakers': 'off'}))

    def test_auto_joins_the_signature(self):
        base = run_signature('a' * 64, {})
        auto = run_signature('a' * 64, {'speakers': 'auto'})
        self.assertNotEqual(base, auto)
        self.assertEqual(auto['options']['speakers'], 'auto')


class SpeakerSegmentMappingTests(unittest.TestCase):
    def test_majority_overlap_decides_each_segment(self):
        windows = [{'start': 0.0, 'end': 10.0, 'cluster': 0},
                   {'start': 10.0, 'end': 20.0, 'cluster': 1}]
        segments = [{'start': 0.0, 'end': 6.0, 'text': 'a'},
                    {'start': 12.0, 'end': 18.0, 'text': 'b'},
                    {'start': 6.0, 'end': 14.0, 'text': 'tie'},
                    {'start': 25.0, 'end': 30.0, 'text': 'silence'}]
        mapped = core._speaker_segments(windows, segments, duration=30.0)
        self.assertEqual([item['speaker'] for item in mapped],
                         ['S1', 'S2', 'S1', None])
        self.assertEqual([item['text'] for item in mapped],
                         ['a', 'b', 'tie', 'silence'])

    def test_clamps_to_media_duration_and_drops_empty_ranges(self):
        windows = [{'start': -5.0, 'end': 2.0, 'cluster': 0},
                   {'start': 8.0, 'end': 14.0, 'cluster': 1},
                   {'start': 'bad', 'end': 3.0, 'cluster': 0},
                   {'start': 1.0, 'end': 2.0, 'cluster': 'bad'}]
        segments = [{'start': -1.0, 'end': 1.5, 'text': 'head'},
                    {'start': 9.0, 'end': 12.0, 'text': 'tail'},
                    {'start': 12.0, 'end': 20.0, 'text': 'past-end'},
                    {'start': 40.0, 'end': 50.0, 'text': 'outside'}]
        mapped = core._speaker_segments(windows, segments, duration=10.0)
        self.assertEqual(mapped, [
            {'start': 0.0, 'end': 1.5, 'speaker': 'S1', 'text': 'head'},
            {'start': 9.0, 'end': 10.0, 'speaker': 'S2', 'text': 'tail'},
        ])
        self.assertEqual(mapped[1]['end'], 10.0)

    def test_labels_are_one_based_and_deterministic(self):
        windows = [{'start': 0.0, 'end': 2.0, 'cluster': 2},
                   {'start': 2.0, 'end': 4.0, 'cluster': 0}]
        mapped = core._speaker_segments(windows, [{'start': 0.0, 'end': 3.0, 'text': 'x'}])
        self.assertEqual(mapped[0]['speaker'], 'S3')


class SpeakerStageTests(unittest.TestCase):
    def setUp(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as handle:
            self.video = handle.name
        self.addCleanup(lambda: Path(self.video).unlink(missing_ok=True))

    def completed(self):
        return subprocess.CompletedProcess(['ffmpeg'], 0, b'', b'')

    def test_missing_model_is_not_run_and_never_decodes(self):
        with mock.patch.dict(os.environ, {'REELPACK_SPEAKER_MODEL': ''}), \
             mock.patch.object(core.subprocess, 'run', return_value=self.completed()) as run:
            state = core._analyze_speakers(
                self.video, [{'start': 0.0, 'end': 1.0, 'text': 'hi'}], 10.0)
        self.assertEqual(state['status'], 'not_run')
        self.assertEqual(state['reason'], 'model_missing')
        self.assertEqual(state['backend'], speaker_campp.BACKEND)
        self.assertTrue(state['experimental'])
        self.assertEqual(state['segments'], [])
        # The 16 kHz mono WAV extraction is requested (and cleaned up afterwards).
        command = run.call_args.args[0]
        self.assertIn('-ar', command)
        self.assertIn('16000', command)
        self.assertIn('-ac', command)
        self.assertFalse(Path(command[-1]).exists())

    def test_ffmpeg_failure_becomes_failed(self):
        fault = subprocess.CalledProcessError(1, 'ffmpeg', stderr=b'decode boom')
        with mock.patch.object(core.subprocess, 'run', side_effect=fault):
            state = core._analyze_speakers(self.video, [], 10.0)
        self.assertEqual(state['status'], 'failed')
        self.assertIn('decode boom', state['reason'])
        self.assertIn('CalledProcessError', state['reason'])

    def test_backend_failure_reason_is_preserved(self):
        failure = {'status': 'failed', 'backend': speaker_campp.BACKEND,
                   'params': {'embedding_model': 'campp.onnx'},
                   'confidence': {}, 'reason': 'injected kmeans fault'}
        with mock.patch.object(core.subprocess, 'run', return_value=self.completed()), \
             mock.patch.object(speaker_campp, 'cluster_speakers', return_value=failure):
            state = core._analyze_speakers(self.video, [], 10.0)
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['reason'], 'injected kmeans fault')
        self.assertEqual(state['model'], 'campp.onnx')
        self.assertEqual(validate_analysis(analysis(state)), [])

    def test_complete_result_maps_windows_into_contract_segments(self):
        result = {
            'status': 'complete', 'backend': speaker_campp.BACKEND,
            'params': {'embedding_model': 'campp.onnx'},
            'confidence': {'selected_k': 2, 'silhouette': 0.45, 'candidates': []},
            'windows': [{'start': 0.0, 'end': 5.0, 'cluster': 0},
                        {'start': 5.0, 'end': 10.0, 'cluster': 1}],
        }
        segments = [{'start': 0.0, 'end': 4.0, 'text': 'first'},
                    {'start': 6.0, 'end': 9.0, 'text': 'second'}]
        with mock.patch.object(core.subprocess, 'run', return_value=self.completed()), \
             mock.patch.object(speaker_campp, 'cluster_speakers', return_value=result):
            state = core._analyze_speakers(self.video, segments, 10.0)
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['model'], 'campp.onnx')
        self.assertEqual(state['confidence'], {'selected_k': 2, 'silhouette': 0.45})
        self.assertEqual([item['speaker'] for item in state['segments']], ['S1', 'S2'])
        self.assertEqual(validate_analysis(analysis(state)), [])


if __name__ == '__main__':
    unittest.main()
