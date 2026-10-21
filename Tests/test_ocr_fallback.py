import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
from ocr_fallback import NATIVE_TIMEOUT_SECONDS, STRATEGY_VERSION, recognize_auto


class OCRFallbackTests(unittest.TestCase):
    def setUp(self):
        self.frames = [(1.5, b'first-image'), (4.0, b'second-image')]
        self.refs = {1.5: ['frame-a'], 4.0: ['frame-b', 'shot-b']}
        self.paths = []

    def native(self, texts=('', ''), errors=(None, None), mutate=None):
        def runner(command, **kwargs):
            self.assertEqual(command[:2], ['/fake/macos_ocr', '--json'])
            self.assertEqual(kwargs, dict(capture_output=True, text=True, timeout=300))
            self.paths = command[2:]
            self.assertEqual([Path(path).read_bytes() for path in self.paths], [frame[1] for frame in self.frames])
            payload = dict(count=len(self.paths), frames=[dict(file=path, text=text, error=error)
                           for path, text, error in zip(self.paths, texts, errors)])
            if mutate:
                mutate(payload)
            return subprocess.CompletedProcess(command, 0, 'helper log\n' + json.dumps(payload), '')
        return Mock(side_effect=runner)

    def call(self, runner, rapid):
        # Legacy error-only fixtures explicitly select native first. Production
        # defaults are exercised separately with the actual probe adapters.
        return recognize_auto(self.frames, self.refs, '/fake/macos_ocr', runner, rapid,
                              backend_order=('macos', 'rapidocr'),
                              probe=lambda backend: dict(status='complete', text='probe text'))

    def complete(self, timestamps=(1.5, 4.0)):
        return dict(status='complete', model='RapidOCR bundled ONNX', params={'device': 'cpu'},
                    items=[dict(timestamp=ts, text=f'rapid {ts}', confidence=.91,
                                region={'units': 'pixels'}, evidence_ids=['wrong']) for ts in timestamps], errors=[])

    def test_native_success_keeps_blob_refs_and_explicit_empty(self):
        rapid = Mock(side_effect=AssertionError('empty OCR must not fallback'))
        result = self.call(self.native(('第一行\n第二行', '')), rapid)
        self.assertEqual((result['status'], result['backend']), ('complete', 'macos'))
        self.assertEqual([item['text'] for item in result['items']], ['第一行\n第二行', ''])
        self.assertEqual(result['items'][1]['evidence_ids'], ['frame-b', 'shot-b'])
        self.assertEqual(result['items'][1]['status'], 'complete')
        self.assertEqual([frame['status'] for frame in result['attempts'][0]['frames']], ['complete', 'complete'])
        self.assertEqual(result['params']['strategy_version'], STRATEGY_VERSION)
        self.assertEqual(result['params']['native_timeout_seconds'], NATIVE_TIMEOUT_SECONDS)
        self.assertIsNone(result['fallback_from'])
        self.assertEqual(result['errors'], [])
        rapid.assert_not_called()
        self.assertTrue(all(not Path(path).exists() for path in self.paths))

    def test_single_frame_error_falls_back_only_that_frame(self):
        rapid = Mock(return_value=self.complete((4.0,)))
        result = self.call(self.native(('native', ''), (None, 'CRImageReaderError:9')), rapid)
        rapid.assert_called_once_with([self.frames[1]], self.refs)
        self.assertEqual((result['status'], result['backend']), ('complete', 'mixed'))
        native, fallback = result['items']
        self.assertEqual(native['text'], 'native')
        self.assertIsNone(native['fallback_from'])
        self.assertEqual(fallback['backend'], 'rapidocr')
        self.assertEqual(fallback['fallback_from'], 'macos')
        self.assertEqual(fallback['reason'], 'CRImageReaderError:9')
        self.assertEqual(fallback['confidence'], .91)
        self.assertEqual(fallback['region'], {'units': 'pixels'})
        self.assertEqual(fallback['evidence_ids'], self.refs[4.0])
        self.assertEqual((fallback['timestamp'], fallback['start'], fallback['end']), (4.0, 4.0, 4.0))
        self.assertEqual(result['errors'][0]['reason'], 'CRImageReaderError:9')
        self.assertIn('CRImageReaderError:9', result['reason'])
        self.assertEqual([attempt['status'] for attempt in result['attempts']], ['partial', 'complete'])

    def test_successful_native_empty_survives_fallback(self):
        rapid = Mock(return_value=self.complete((4.0,)))
        result = self.call(self.native(errors=(None, 'Vision failed')), rapid)
        self.assertEqual((result['status'], result['backend']), ('complete', 'mixed'))
        self.assertEqual(result['items'][0]['text'], '')
        rapid.assert_called_once_with([self.frames[1]], self.refs)

    def test_batch_process_failures_retry_all_frames_once(self):
        failures = [FileNotFoundError('helper missing'), subprocess.TimeoutExpired('helper', 300),
                    subprocess.CalledProcessError(9, 'helper', stderr='crash')]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                rapid = Mock(return_value=self.complete())
                runner = Mock(side_effect=failure)
                result = self.call(runner, rapid)
                rapid.assert_called_once_with(self.frames, self.refs)
                runner.assert_called_once()
                self.assertEqual((result['status'], result['backend']), ('complete', 'rapidocr'))
                self.assertIn(type(failure).__name__, result['reason'])
                if isinstance(failure, subprocess.CalledProcessError):
                    self.assertIn('crash', result['reason'])
                self.assertEqual(len(result['attempts']), 2)

    def test_native_nonzero_preserves_stderr(self):
        rapid = Mock(return_value=self.complete())
        result = self.call(Mock(return_value=subprocess.CompletedProcess([], -11, '', 'native crash detail')), rapid)
        self.assertIn('native crash detail', result['reason'])
        self.assertEqual(result['attempts'][0]['status'], 'failed')
        rapid.assert_called_once_with(self.frames, self.refs)

    def test_bad_json_retries_whole_batch(self):
        for stdout in ('not-json', '[]', '{"frames":null}', '{broken'):
            with self.subTest(stdout=stdout):
                rapid = Mock(return_value=self.complete())
                result = self.call(Mock(return_value=subprocess.CompletedProcess([], 0, stdout, '')), rapid)
                self.assertEqual(result['status'], 'complete')
                self.assertTrue(result['errors'])
                rapid.assert_called_once_with(self.frames, self.refs)

    def test_invalid_count_paths_or_text_retry_whole_batch(self):
        def short(payload):
            payload['frames'].pop()
        def wrong_count(payload):
            payload['count'] = 99
        def reversed_paths(payload):
            payload['frames'].reverse()
        def missing_path(payload):
            del payload['frames'][0]['file']
        def invalid_text(payload):
            payload['frames'][0]['text'] = None
        for mutate in (short, wrong_count, reversed_paths, missing_path, invalid_text):
            with self.subTest(case=mutate.__name__):
                rapid = Mock(return_value=self.complete())
                result = self.call(self.native(mutate=mutate), rapid)
                rapid.assert_called_once_with(self.frames, self.refs)
                self.assertEqual(result['backend'], 'rapidocr')
                self.assertEqual(result['status'], 'complete')

    def test_unavailable_or_failed_rapid_keeps_native_success(self):
        for status in ('not_run', 'failed'):
            with self.subTest(status=status):
                rapid = Mock(return_value=dict(status=status, reason='missing package or model', items=[], errors=[]))
                result = self.call(self.native(('native', ''), (None, 'Vision failed')), rapid)
                self.assertEqual(result['status'], 'partial')
                self.assertEqual(result['backend'], 'macos')
                self.assertEqual([item['text'] for item in result['items']], ['native'])
                self.assertEqual(result['attempts'][1]['status'], status)
                self.assertIn('missing package or model', result['reason'])
                self.assertEqual(len(result['errors']), 2)

    def test_all_unavailable_is_not_run_and_executed_failure_is_failed(self):
        for failure, expected in ((FileNotFoundError('missing'), 'not_run'),
                                  (subprocess.TimeoutExpired('helper', 300), 'failed')):
            with self.subTest(expected=expected):
                result = self.call(Mock(side_effect=failure), Mock(return_value=dict(status='not_run', reason='RapidOCR unavailable')))
                self.assertEqual(result['status'], expected)
                self.assertEqual(result['items'], [])
                self.assertTrue(result['reason'])
                self.assertEqual(len(result['attempts']), 2)

    def test_all_failed_has_both_real_errors(self):
        result = self.call(self.native(errors=('native one', 'native two')),
                           Mock(return_value=dict(status='failed', reason='rapid failure', errors=[
                               dict(timestamp=1.5, reason='rapid one'), dict(timestamp=4.0, reason='rapid two')])))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['items'], [])
        self.assertEqual([error['reason'] for error in result['errors']],
                         ['native one', 'native two', 'rapid one', 'rapid two'])

    def test_rapid_partial_records_successful_empty_frame(self):
        rapid = Mock(return_value=dict(status='partial', items=[], errors=[dict(timestamp=4.0, reason='bad image')]))
        result = self.call(self.native(errors=('native one', 'native two')), rapid)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['items'][0]['text'], '')
        self.assertEqual(result['items'][0]['timestamp'], 1.5)
        self.assertEqual([frame['status'] for frame in result['attempts'][1]['frames']], ['complete', 'failed'])

    def test_rapid_successful_empty_batch_is_complete(self):
        result = self.call(self.native(errors=('native one', 'native two')),
                           Mock(return_value=dict(status='complete', items=[], errors=[])))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual([item['text'] for item in result['items']], ['', ''])
        self.assertEqual([frame['status'] for frame in result['attempts'][1]['frames']], ['complete', 'complete'])

    def test_rapid_exception_and_invalid_responses_are_failed(self):
        for response in (None, dict(status='unknown'), dict(status='complete', items=[dict(timestamp=99, text='wrong')]),
                         dict(status='partial', errors=['unmapped error'])):
            with self.subTest(response=response):
                result = self.call(self.native(errors=('n1', 'n2')), Mock(return_value=response))
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['reason'])
        result = self.call(self.native(errors=('n1', 'n2')), Mock(side_effect=RuntimeError('rapid broke')))
        self.assertEqual(result['status'], 'failed')
        self.assertIn('rapid broke', result['reason'])

    def test_default_rapid_uses_bounded_offline_runner(self):
        with patch('optional_backends.recognize_frames', return_value=self.complete()) as rapid:
            result = self.call(self.native(errors=('n1', 'n2')), None)
        self.assertEqual(rapid.call_args.args, (self.frames, self.refs))
        bounded = rapid.call_args.kwargs['runner']
        with patch('ocr_fallback.subprocess.run') as process:
            bounded(['python'], timeout=1800, env={'PATH': '/usr/bin'})
        self.assertEqual(process.call_args.kwargs['timeout'], 120)
        self.assertEqual(process.call_args.kwargs['env']['HF_HUB_OFFLINE'], '1')
        self.assertEqual(result['status'], 'complete')

    def test_empty_input_completes_without_processes(self):
        runner, rapid = Mock(), Mock()
        result = recognize_auto([], {}, '/fake/helper', runner, rapid)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['attempts'], [])
        runner.assert_not_called()
        rapid.assert_not_called()


if __name__ == '__main__':
    unittest.main()
