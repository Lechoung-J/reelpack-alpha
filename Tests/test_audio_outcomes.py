import json
from pathlib import Path
from unittest.mock import patch
from test_core_local import LocalPipelineTests
import core
from asset_store import AssetStore


class AudioOutcomeTests(LocalPipelineTests):
    def saved(self):
        return json.loads(Path(self.completion()['analysis']).read_text())

    def test_empty_successful_asr_is_complete_and_reusable(self):
        self.transcribe.return_value = ([], {'language': 'en', 'params': {}})
        core.main(self.arguments)
        a = self.saved()
        self.assertEqual(a['transcript']['status'], 'complete')
        self.assertEqual(a['transcript']['outcome'], 'speech_not_detected')
        self.assertEqual(a['transcript']['raw_text'], '')
        self.assertFalse(any(e['type'] == 'error' for e in self.messages))
        self.messages.clear(); self.transcribe.reset_mock(); self.shots.reset_mock()
        core.main(self.arguments)
        self.assertTrue(self.completion()['cached'])
        self.transcribe.assert_not_called(); self.shots.assert_not_called()

    def test_absent_audio_track_skips_asr_music_and_is_reusable(self):
        self.audio_track.return_value = {'status':'complete', 'present':False}
        core.main(self.arguments)
        a = self.saved()
        self.assertEqual(a['transcript']['outcome'], 'no_audio_track')
        self.assertEqual(a['transcript']['status'], 'not_run')
        self.assertEqual(a['music']['status'], 'not_run')
        self.assertEqual(a['ocr']['status'], 'complete')
        self.transcribe.assert_not_called(); self.music.assert_not_called(); self.preflight.assert_not_called()
        self.messages.clear()
        core.main(self.arguments)
        self.assertTrue(self.completion()['cached'])

    def test_inference_failure_preserves_new_partial_checkpoint(self):
        self.transcribe.side_effect = RuntimeError('decoder fixture failure')
        with self.assertRaises(SystemExit) as error: core.main(self.arguments)
        self.assertEqual(error.exception.code, 1)
        packages=list((self.library/'packages').glob('asset_*/analysis.json'))
        self.assertEqual(len(packages), 1)
        a=json.loads(packages[0].read_text())
        self.assertEqual(a['execution']['status'], 'failed')
        self.assertEqual(a['transcript']['status'], 'failed')
        self.assertEqual(a['music']['status'], 'complete')
        self.assertEqual(a['shots']['status'], 'complete')
        self.assertFalse(any(e['type']=='complete' for e in self.messages))

    def test_failed_force_attempt_preserves_old_package_and_readable(self):
        core.main(self.arguments)
        output=self.completion();pkg=Path(output['package']);md=Path(output['markdown'])
        before={str(p.relative_to(pkg)):p.read_bytes() for p in pkg.rglob('*') if p.is_file()}
        readable=md.read_bytes()
        self.messages.clear(); self.transcribe.side_effect=RuntimeError('inference failed on rerun')
        with self.assertRaises(SystemExit):core.main(self.arguments+['--force'])
        after={str(p.relative_to(pkg)):p.read_bytes() for p in pkg.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        self.assertEqual(readable,md.read_bytes())
        self.assertTrue(list((self.library/'.attempts').rglob('analysis.json')))

    def test_export_failure_after_commit_uses_saved_package_markdown(self):
        export=self.root/'blocked-export';export.write_text('file, not directory')
        arguments=list(self.arguments);arguments[arguments.index('--output-dir')+1]=str(export)
        core.main(arguments)
        output=self.completion()
        self.assertTrue(Path(output['markdown']).is_file())
        self.assertTrue(Path(output['markdown']).is_relative_to(Path(output['package'])))
        self.assertEqual(self.saved()['execution']['status'],'complete')
        self.assertTrue(any('导出失败' in x.get('message','') for x in self.messages))

    def test_probe_failure_is_not_treated_as_no_audio_track(self):
        self.audio_track.side_effect=RuntimeError('ffprobe failed')
        with self.assertRaises(SystemExit):core.main(self.arguments)
        self.transcribe.assert_not_called();self.music.assert_not_called();self.shots.assert_not_called()
        a=json.loads(next((self.library/'packages').glob('asset_*/analysis.json')).read_text())
        self.assertEqual(a['video']['audio_track']['status'],'failed')
        self.assertIsNone(a['video']['audio_track']['present'])
        self.assertEqual(a['execution']['status'],'failed')
