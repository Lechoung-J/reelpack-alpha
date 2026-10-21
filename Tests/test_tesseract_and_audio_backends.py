import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
from ocr_tesseract import parse_tsv, recognize_frames
from music_timbre import describe_timbre, analyze_timbre

TSV = 'level\tleft\ttop\twidth\theight\tconf\ttext\n5\t10\t20\t80\t30\t92\t中文字幕\n'

class TesseractAndAudioBackendTests(unittest.TestCase):
    def setUp(self):
        temp = patch('ocr_tesseract.tempfile.TemporaryDirectory')
        temp.start().return_value.__enter__.return_value = '/tmp/mock-ocr'
        self.addCleanup(temp.stop)

    def test_regions_and_refs(self):
        item = parse_tsv(TSV, 1.5, ['frame1'])[0]
        self.assertEqual(item['text'], '中文字幕')
        self.assertEqual(item['region']['units'], 'pixels')
        self.assertEqual(item['evidence_ids'], ['frame1'])
        self.assertEqual(item['start'], 1.5)

    @patch('ocr_tesseract.Path.write_bytes')
    def test_backend_success(self, write):
        calls = []
        def run(command, **kw):
            calls.append(command)
            return SimpleNamespace(stdout=TSV)
        result = recognize_frames([(0, b'mock')], {0: ['e1']}, runner=run)
        self.assertEqual(result['status'], 'complete')
        self.assertIn('chi_sim+eng', calls[0])
        self.assertEqual(result['backend'], 'tesseract')

    @patch('ocr_tesseract.Path.write_bytes')
    def test_failure_and_partial(self, write):
        def fail(*a, **kw): raise FileNotFoundError('mock missing binary')
        result = recognize_frames([(0, b'')], runner=fail)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('missing binary', result['reason'])
        with patch('ocr_tesseract.subprocess.run', side_effect=[SimpleNamespace(stdout=TSV), FileNotFoundError('missing')]):
            self.assertEqual(recognize_frames([(0,b''),(1,b'')])['status'], 'partial')

    def test_timbre_schema(self):
        result = describe_timbre(800, .1, duration_seconds=3, sample_rate=22050)
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['experimental'])
        self.assertIsNone(result['params']['weights'])
        self.assertEqual(result['scope'], 'whole_audio_mix')
        self.assertEqual(result['result']['spectral_centroid_hz'],800)

    def test_timbre_invalid(self):
        self.assertEqual(describe_timbre(float('nan'), .1, duration_seconds=3, sample_rate=22050)['status'], 'failed')
        self.assertEqual(analyze_timbre([], 22050)['status'], 'failed')

    def test_package_partial_and_timbre(self):
        import core
        from asset_contract import validate_analysis
        result = {'status': 'partial', 'backend': 'tesseract', 'model': 'chi_sim+eng',
                  'params': {}, 'items': parse_tsv(TSV, 1.5, ['frame1']),
                  'errors': [{'timestamp': 2, 'reason': 'mock failure'}], 'reason': 'mock failure'}
        args = SimpleNamespace(model='large', ocr_backend='tesseract', vision_model='mock',
             ocr_attempted=True, ocr_error=result['reason'], ocr_result=result,
             ocr_observations=result['items'], skip_summary=True)
        timbre = describe_timbre(440, .01, duration_seconds=3, sample_rate=22050)
        with patch.object(core, '_probe_video_metadata', return_value={'duration_seconds':3}):
            analysis = core._save_pipeline_analysis(Mock(), {'asset_id':'test'}, 'mock.mp4',
              'mock.mp4', [], '', '', '中文字幕', '', args,
              evidence=[{'evidence_id':'frame1','kind':'image','timestamp':1.5,
                         'relative_path':'evidence/frame1.jpg'}],
              music={'status':'complete','timbre':timbre})
        self.assertEqual(analysis['ocr']['status'], 'partial')
        self.assertEqual(analysis['capabilities']['ocr']['status'], 'partial')
        self.assertEqual(analysis['music']['timbre'], timbre)
        self.assertEqual(validate_analysis(analysis), [])

    def test_audio_worker_failure_schema(self):
        import core
        with patch.object(core.Path, 'is_file', return_value=True), patch.object(core.subprocess,
               'run', side_effect=RuntimeError('mock worker failure')):
            result = core.analyze_audio_isolated('mock.mp4', [])
        self.assertEqual(result['timbre']['status'], 'failed')
        self.assertIn('worker failure', result['timbre']['reason'])
        self.assertEqual(result['genre']['status'], 'not_run')

    def test_cli_option(self):
        import core
        with patch.object(core, 'check_dependencies', return_value=[]), \
             patch.object(core, 'resolve_input', side_effect=RuntimeError('stop after parser')) as local_input, \
             patch.object(core, 'emit_error') as error, patch.object(core, 'emit_progress'):
            with self.assertRaises(SystemExit) as caught:
                core.main(['--input', 'mock.mp4', '--ocr-backend', 'tesseract'])
            self.assertEqual(caught.exception.code, 1)
            local_input.assert_called_once_with('mock.mp4')
            error.assert_called_once_with('core', 'PIPELINE_ERROR', 'stop after parser')

if __name__ == '__main__': unittest.main()
