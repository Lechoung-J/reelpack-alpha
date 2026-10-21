import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
import core

try:  # faster-whisper (and its bundled Silero VAD API) is only present in the pipeline runtime env.
    from faster_whisper.vad import VadOptions as _VadOptions  # noqa: F401
    HAS_FASTER_WHISPER_VAD = True
except Exception:
    HAS_FASTER_WHISPER_VAD = False


class TranscriptionBoundsTests(unittest.TestCase):
    def run_transcription(self, rows, language='auto', duration=10):
        model = Mock()
        model.transcribe.return_value = (
            iter(SimpleNamespace(start=s, end=e, text=t) for s, e, t in rows),
            SimpleNamespace(language='ja', language_probability=.99))
        with patch.object(core, 'WhisperModel', return_value=model), \
             patch.object(core, '_probe_video_metadata', return_value={'duration_seconds': duration}), \
             patch.object(core, 'emit_progress'):
            result = core.transcribe_audio('fixture.mp4', return_info=True, language=language)
        return result, model.transcribe.call_args.kwargs

    def test_clips_crossing_and_drops_beyond_end(self):
        (segments, info), _ = self.run_transcription([
            (0, 2, ' normal '), (9, 12, 'crossing'), (10, 11, 'boundary'),
            (13, 14, 'beyond'), (8, 8, 'empty'), (math.nan, 4, 'invalid')])
        self.assertEqual(segments, [{'start': 0, 'end': 2, 'text': 'normal'},
                                    {'start': 9, 'end': 10, 'text': 'crossing'}])
        self.assertEqual(info['timestamp_bounds']['dropped_segments'], 4)
        self.assertEqual(info['timestamp_bounds']['clipped_segments'], 1)

    @unittest.skipUnless(HAS_FASTER_WHISPER_VAD, 'faster-whisper.vad (Silero) 未安装；该运行时由 test_default_vad_fallback_is_reported 覆盖')
    def test_vad_and_condition_parameters_are_forwarded(self):
        (_, info), params = self.run_transcription([(0, 1, 'test')], language='ja')
        self.assertTrue(params['vad_filter'])
        self.assertFalse(params['condition_on_previous_text'])
        self.assertEqual(params['language'], 'ja')
        self.assertEqual(params['vad_parameters'], {'min_silence_duration_ms': 500, 'speech_pad_ms': 400})
        for key, value in params.items():
            self.assertEqual(info['params'][key], value)

    def test_auto_language_and_unknown_duration_are_truthful(self):
        (segments, info), params = self.run_transcription([(0, 12, 'test')], duration=None)
        self.assertNotIn('language', params)
        self.assertEqual(segments[0]['end'], 12)
        self.assertIsNone(info['timestamp_bounds']['duration_seconds'])
        self.assertTrue(info['warnings'])

    def test_missing_large_model_explains_cost_before_any_model_download(self):
        with patch.object(core, 'WhisperModel') as model, \
             patch('runtime_environment.inspect_runtime_roots', return_value={'models': {'whisper': {'ready': False}}}), \
             patch.object(core, 'emit_progress') as progress:
            with self.assertRaisesRegex(RuntimeError, '没有自动下载'):
                core.transcribe_audio('not-read.mp4', model_name='large')
        model.assert_not_called()
        self.assertTrue(any('3.1 GB' in item.args[2] and '确认' in item.args[2] for item in progress.call_args_list))

    def test_default_vad_fallback_is_reported(self):
        with patch.dict(sys.modules, {'faster_whisper.vad': None}):
            (_, info), params = self.run_transcription([(0, 1, 'test')])
        self.assertTrue(params['vad_filter'])
        self.assertNotIn('vad_parameters', params)
        self.assertIn('Silero custom parameters unavailable', info['warnings'][0])


if __name__ == '__main__':
    unittest.main()
