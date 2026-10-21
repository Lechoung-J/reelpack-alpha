import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
from shot_detect import detect_shots
from shot_detect_transnet import select_transnet_boundaries


class BackendTests(unittest.TestCase):
    def test_custom_dispatch(self):
        with patch('shot_detect.detect_shots_custom', return_value=({'status': 'complete'}, [])) as run:
            detect_shots('input', 'output', backend='custom')
            self.assertEqual(run.call_args.args[2], .28)

    def test_network_dispatch(self):
        expected = ({'status': 'complete', 'items': []}, [])
        with patch('shot_detect.detect_shots_isolated', return_value=expected) as run:
            self.assertEqual(detect_shots('input', 'output', backend='transnetv2'), expected)
            self.assertEqual(run.call_args.args[3], .5)

    def test_strict_missing_does_not_fallback(self):
        missing = ({'status': 'not_run', 'backend': 'transnetv2-pytorch-cpu', 'reason': 'missing', 'items': []}, [])
        with patch('shot_detect.detect_shots_isolated', return_value=missing), patch('shot_detect.detect_shots_custom') as custom:
            self.assertEqual(detect_shots('input', 'output', fallback=False, backend='transnetv2'), missing)
            custom.assert_not_called()

    def test_fallback_is_visible(self):
        with patch('shot_detect.detect_shots_isolated', return_value=({'status': 'failed', 'backend': 'network', 'reason': 'bad weights'}, [])), patch('shot_detect.detect_shots_custom', return_value=({'status': 'complete', 'backend': 'custom', 'items': []}, [])):
            result, _ = detect_shots('input', 'output', backend='transnetv2')
            self.assertEqual(result['backend'], 'custom')
            self.assertEqual(result['fallback']['reason'], 'bad weights')

    def test_environment_switch(self):
        with patch.dict(os.environ, {'REELPACK_SHOT_BACKEND': 'custom'}), patch('shot_detect.detect_shots_custom', return_value=({}, [])) as custom:
            detect_shots('input', 'output')
            custom.assert_called_once()

    def test_invalid_backend(self):
        with self.assertRaises(ValueError):
            detect_shots('input', 'output', backend='invalid')

    def test_run_peak_and_edge_policy(self):
        self.assertEqual([c[0] for c in select_transnet_boundaries([.9, .1, .6, .9, .7, .1, .8])], [3, 6])
        self.assertEqual(select_transnet_boundaries([.1, .5, .1]), [])

    def test_missing_assets_status(self):
        from shot_detect_transnet import detect_shots_transnet
        # Test the adapter's availability guard directly, independent of worker Python.
        with patch('shot_detect_transnet.Path.exists', return_value=False), \
             patch('shot_detect.FFmpegCapture') as decoder, \
             patch('builtins.__import__', wraps=__import__) as imports:
            result, evidence = detect_shots_transnet('missing.mp4', 'output',
                                                     model_dir='/synthetic/missing-transnet')
        self.assertEqual(result['status'], 'not_run')
        self.assertIn('model_dir does not exist', result['reason'])
        self.assertEqual(result['items'], [])
        self.assertEqual(evidence, [])
        decoder.assert_not_called()
        self.assertFalse(any(call.args[0] in ('torch', 'transnetv2_pytorch')
                             for call in imports.call_args_list))

    def test_window_alignment(self):
        import numpy as np
        from shot_detect_transnet import predict_frames
        class Tensor:
            def __init__(self, value): self.value = value
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.value
        class Torch:
            from contextlib import nullcontext as inference_mode
            @staticmethod
            def from_numpy(value): return value
            @staticmethod
            def sigmoid(value): return Tensor(value)
        frames = np.zeros((123, 27, 48, 3), dtype=np.uint8)
        frames[:, 0, 0, 0] = np.arange(123)
        values = predict_frames(frames, lambda x: x[:, :, 0, 0, 0], Torch)
        np.testing.assert_equal(values, np.arange(123))

    def test_adapter_contract_with_stub_predictions(self):
        import shutil
        import subprocess
        import tempfile
        import numpy as np
        from asset_contract import empty_analysis, assert_valid_analysis
        if not shutil.which('ffmpeg'):
            self.skipTest('ffmpeg unavailable')
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / 'input.mp4'
            subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                            'color=c=red:s=96x54:r=10:d=2', '-c:v', 'libx264', str(video)], check=True)
            values = np.zeros(20, dtype=np.float32)
            values[10] = .9
            with patch('shot_detect_transnet.load_model', return_value=(None, None, 'stub-not-real-weights')), patch('shot_detect_transnet.predict_frames', return_value=values):
                from shot_detect_transnet import detect_shots_transnet
                result, evidence = detect_shots_transnet(video, Path(td) / 'package')
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(result['items']), 2)
            self.assertEqual(result['items'][1]['start'], 1.)
            self.assertTrue(all((Path(td) / 'package' / e['relative_path']).is_file() for e in evidence))
            analysis = empty_analysis('asset_test', {'kind': 'local_file'})
            analysis['shots'], analysis['evidence'] = result, evidence
            assert_valid_analysis(analysis)
