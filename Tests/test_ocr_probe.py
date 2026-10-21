import json
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
from ocr_fallback import recognize_auto, STRATEGY_VERSION
from ocr_probe import probe_png, PROBE_TIMESTAMP


class ProbeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.frames = [(1., b'one'), (2., b'two')]
        self.refs = {1.: ['e1'], 2.: ['e2']}

    def response(self, frames, text='recognized text'):
        return dict(status='complete', items=[dict(timestamp=ts, text=text) for ts, _ in frames], errors=[])

    def native(self, material_error=None, material_text='native material'):
        def run(command, **kwargs):
            paths = command[2:]
            is_probe = Path(paths[0]).read_bytes() == probe_png()
            self.assertEqual(kwargs['timeout'], 10 if is_probe else 300)
            rows = [dict(file=path, text='recognized probe' if is_probe else material_text,
                         error=None if is_probe else material_error) for path in paths]
            return subprocess.CompletedProcess(command, 0, json.dumps(dict(count=len(rows), frames=rows)), '')
        return Mock(side_effect=run)

    def call(self, **kwargs):
        return recognize_auto(self.frames, self.refs, '/fake/helper', **kwargs)

    def test_probe_png_is_deterministic_valid_and_self_contained(self):
        image = probe_png()
        self.assertEqual(image, probe_png())
        self.assertTrue(image.startswith(b'\x89PNG\r\n\x1a\n'))
        width, height = struct.unpack('!II', image[16:24])
        self.assertGreater(width, 400)
        self.assertGreater(height, 60)
        offset, compressed = 8, b''
        while offset < len(image):
            length = struct.unpack('!I', image[offset:offset + 4])[0]
            kind, payload = image[offset + 4:offset + 8], image[offset + 8:offset + 8 + length]
            self.assertEqual(zlib.crc32(kind + payload) & 0xffffffff,
                             struct.unpack('!I', image[offset + 8 + length:offset + 12 + length])[0])
            if kind == b'IDAT':
                compressed += payload
            offset += length + 12
        self.assertEqual(len(zlib.decompress(compressed)), (width + 1) * height)

    def test_default_rapid_probe_runs_adapter_once_then_all_frames(self):
        rapid = Mock(side_effect=lambda frames, refs: self.response(frames, 'R EELPACK'))
        runner, ollama = Mock(), Mock()
        result = self.call(rapid=rapid, runner=runner, ollama=ollama)
        self.assertEqual(result['params']['strategy_version'], 'rapid-probe-v2')
        self.assertEqual(STRATEGY_VERSION, 'rapid-probe-v2')
        self.assertEqual((result['status'], result['backend']), ('complete', 'rapidocr'))
        self.assertEqual(rapid.call_args_list[0].args, ([(PROBE_TIMESTAMP, probe_png())], {}))
        self.assertEqual(rapid.call_args_list[1].args, (self.frames, self.refs))
        self.assertEqual(rapid.call_count, 2)
        self.assertEqual(result['probes'][0]['text'], 'R EELPACK')  # Exact match is deliberately unnecessary.
        runner.assert_not_called()
        ollama.assert_not_called()

    def test_empty_material_is_success_and_does_not_probe_fallbacks(self):
        rapid = Mock(side_effect=[self.response([(PROBE_TIMESTAMP, b'')]), self.response(self.frames, '')])
        runner, ollama = Mock(), Mock()
        result = self.call(rapid=rapid, runner=runner, ollama=ollama)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual([item['text'] for item in result['items']], ['', ''])
        self.assertEqual(len(result['probes']), 1)
        self.assertEqual(len(result['attempts']), 1)
        runner.assert_not_called()
        ollama.assert_not_called()

    def test_empty_probe_fails_but_native_empty_material_succeeds(self):
        rapid = Mock(return_value=dict(status='complete', items=[]))
        native = self.native(material_text='')
        result = self.call(rapid=rapid, runner=native)
        self.assertEqual(result['backend'], 'macos')
        self.assertEqual(result['probes'][0]['status'], 'failed')
        self.assertIn('empty', result['probes'][0]['reason'])
        self.assertEqual([item['text'] for item in result['items']], ['', ''])
        self.assertEqual(native.call_count, 2)
        rapid.assert_called_once()

    def test_temporary_rapid_probe_failure_can_retry_after_native_material_error(self):
        # macOS→RapidOCR is possible under rapid-first priority: the rapid probe
        # temporarily fails, macOS passes its probe, then macOS material fails.
        for first in (RuntimeError('transient probe'), dict(status='complete', items=[])):
            with self.subTest(first=first):
                rapid = Mock(side_effect=[first, self.response(self.frames)])
                native = self.native(material_error='native material failure')
                result = self.call(rapid=rapid, runner=native)
                self.assertEqual((result['status'], result['backend']), ('complete', 'rapidocr'))
                self.assertEqual([p['backend'] for p in result['probes']], ['rapidocr', 'macos'])
                self.assertEqual([a['backend'] for a in result['attempts']], ['macos', 'rapidocr'])
                self.assertTrue(result['attempts'][1]['retry_after_probe_failure'])
                self.assertEqual(rapid.call_count, 2)
                self.assertEqual(result['items'][0]['fallback_from'], 'macos')
                self.assertIn('native material failure', result['items'][0]['reason'])

    def test_missing_rapid_is_skipped_after_native_error(self):
        rapid = Mock(return_value=dict(status='not_run', reason='optional python unavailable'))
        ollama = Mock(return_value=dict(status='not_run', reason='no vision model'))
        result = self.call(rapid=rapid, runner=self.native(material_error='failed'), ollama=ollama)
        rapid.assert_called_once()
        ollama.assert_called_once()
        self.assertEqual([a['backend'] for a in result['attempts']], ['macos'])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['probes'][0]['availability'], 'missing')

    def test_rapid_missing_chooses_native(self):
        rapid = Mock(side_effect=ModuleNotFoundError('rapidocr_onnxruntime'))
        result = self.call(rapid=rapid, runner=self.native())
        self.assertEqual(result['backend'], 'macos')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['probes'][0]['status'], 'not_run')
        rapid.assert_called_once()

    def test_all_probe_empty_does_not_retry_any_material(self):
        rapid = Mock(return_value=dict(status='complete', items=[]))
        ollama = Mock(return_value=dict(status='complete', items=[]))
        def empty_native(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, json.dumps({'frames': [dict(file=command[2], text='')]}), '')
        native = Mock(side_effect=empty_native)
        result = self.call(rapid=rapid, runner=native, ollama=ollama)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['attempts'], [])
        self.assertEqual(len(result['probes']), 3)
        for mock in (rapid, native, ollama):
            mock.assert_called_once()

    def test_all_missing_is_not_run(self):
        result = self.call(rapid=Mock(return_value=dict(status='not_run', reason='missing')),
                           runner=Mock(side_effect=FileNotFoundError('missing')),
                           ollama=Mock(return_value=dict(status='not_run', reason='no model')))
        self.assertEqual(result['status'], 'not_run')
        self.assertEqual(result['attempts'], [])
        self.assertEqual(len(result['probes']), 3)

    def test_per_frame_success_preserved_across_three_adapters(self):
        frames = self.frames + [(3., b'three')]
        rapid = Mock(return_value=dict(status='partial', model='rapid-model',
                     items=[dict(timestamp=1., text='rapid success')],
                     errors=[dict(timestamp=2., reason='rapid bad two'), dict(timestamp=3., reason='rapid bad three')]))
        def native_run(command, **kwargs):
            self.assertEqual([Path(p).read_bytes() for p in command[2:]], [b'two', b'three'])
            return subprocess.CompletedProcess(command, 0, json.dumps({'frames': [
                dict(file=command[2], text='', error=None), dict(file=command[3], text='', error='native bad three')]}), '')
        ollama = Mock(side_effect=lambda selected, refs: self.response(selected, 'ollama success'))
        result = recognize_auto(frames, self.refs, '/fake/helper', runner=native_run, rapid=rapid, ollama=ollama,
                                probe=lambda name: dict(status='complete', text='ok'))
        self.assertEqual((result['status'], result['backend']), ('complete', 'mixed'))
        self.assertEqual([item['text'] for item in result['items']], ['rapid success', '', 'ollama success'])
        self.assertEqual([item['fallback_from'] for item in result['items']], [None, 'rapidocr', 'macos'])
        self.assertIn('rapid bad three', result['items'][2]['reason'])
        self.assertIn('native bad three', result['items'][2]['reason'])
        ollama.assert_called_once_with([(3., b'three')], self.refs)
        self.assertEqual(set(result['model']), {'rapidocr', 'macos', 'ollama'})

    def test_all_material_errors_are_bounded_and_audited(self):
        rapid = Mock(side_effect=subprocess.TimeoutExpired('rapid', 120))
        native = Mock(side_effect=subprocess.TimeoutExpired('native', 300))
        ollama = Mock(side_effect=TimeoutError('ollama'))
        result = self.call(rapid=rapid, runner=native, ollama=ollama,
                           probe=lambda name: dict(status='complete', text='ok'))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['attempts']), 3)
        self.assertEqual(len(result['errors']), 3)
        self.assertEqual(result['items'], [])
        for mock in (rapid, native, ollama):
            mock.assert_called_once()

    def test_default_rapid_probe_and_material_use_bounded_offline_runner(self):
        observed = []
        def recognize(selected, refs, runner):
            with patch('ocr_fallback.subprocess.run') as process:
                runner(['/python', '/Core/optional_backend_worker.py', 'rapidocr'], env={'PATH': '/bin'}, timeout=1800)
                observed.append(process.call_args)
            return self.response(selected)
        with patch('optional_backends.recognize_frames', side_effect=recognize):
            result = self.call()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual([call.kwargs['timeout'] for call in observed], [10, 120])
        for call in observed:
            self.assertEqual(Path(call.args[0][1]).name, 'ocr_probe_worker.py')
            self.assertEqual(call.kwargs['env']['HF_HUB_OFFLINE'], '1')
            self.assertEqual(call.kwargs['env']['TRANSFORMERS_OFFLINE'], '1')

    def test_worker_denies_network_before_optional_import(self):
        worker = Path(__file__).resolve().parents[1] / 'Core/ocr_probe_worker.py'
        code = '''import runpy, socket, sys
original = runpy.run_path
def check(*args, **kwargs):
    for operation in (lambda: socket.create_connection(('example.com', 80)),
                      lambda: socket.getaddrinfo('example.com', 80),
                      lambda: socket.socket().connect(('127.0.0.1', 11434))):
        try:
            operation()
        except RuntimeError as exc:
            assert 'disabled' in str(exc)
        else:
            raise AssertionError('network was not blocked')
runpy.run_path = check
original(sys.argv[1], run_name='__main__')
'''
        result = subprocess.run([sys.executable, '-c', code, str(worker)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_no_frames_skips_probes_and_all_adapters(self):
        probe, rapid, native, ollama = Mock(), Mock(), Mock(), Mock()
        result = recognize_auto([], {}, '/fake/helper', native, rapid, ollama=ollama, probe=probe)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['probes'], [])
        for mock in (probe, rapid, native, ollama):
            mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
