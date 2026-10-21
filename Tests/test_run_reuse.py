import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Core'))
from pipeline_state import run_signature, reusable, annotate_analysis, capability_summary
from test_core_local import LocalPipelineTests
from asset_store import AssetStore
import core


class RunSignatureTests(unittest.TestCase):
    def complete(self, options=None):
        signature = run_signature('a'*64, options or {})
        analysis = {'status':'partial', 'run_signature':signature,
                    'capabilities':{name:{'status':'complete'} for name in ['transcription','shots','ocr']},
                    'transcript':{'status':'complete'},'shots':{'status':'complete'},'ocr':{'status':'complete'}}
        return analysis, signature

    def test_rewrite_effective_cloud_configuration_changes_signature_without_exposing_secret(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY':'secret-test-key','REELPACK_CLOUD_MODEL':'model-a','OPENAI_BASE_URL':'https://one.invalid/v1'}):
            first=run_signature('a'*64,{'cloud':True})
            with patch.dict('os.environ', {'REELPACK_CLOUD_MODEL':'model-b'}):
                self.assertNotEqual(first,run_signature('a'*64,{'cloud':True}))
            with patch.dict('os.environ', {'OPENAI_BASE_URL':'https://two.invalid/v1'}):
                self.assertNotEqual(first,run_signature('a'*64,{'cloud':True}))
        self.assertNotIn('secret-test-key',json.dumps(first))

    def test_partial_package_reuses_complete_core_capabilities(self):
        analysis, signature = self.complete()
        analysis['music']={'status':'failed'}
        self.assertTrue(reusable(analysis,signature))

    def test_every_result_affecting_option_changes_signature(self):
        base=run_signature('a'*64,{})
        for name,value in {'language':'zh','model':'different','ocr_backend':'macos','ocr_max_width':960,
                           'music_backend':'panns','summary_backend':'cloud','correction_backend':'ollama',
                           'vision_model':'different','genre_threshold':.1,'genre_top_k':4}.items():
            with self.subTest(name=name):
                self.assertNotEqual(base,run_signature('a'*64,{name:value}))
        self.assertNotEqual(base,run_signature('b'*64,{}))

    def test_missing_signature_or_required_state_does_not_reuse(self):
        analysis, signature=self.complete()
        for key in ['transcription','shots','ocr']:
            changed=copy.deepcopy(analysis);changed['capabilities'][key]['status']='failed'
            self.assertFalse(reusable(changed,signature))
        analysis.pop('run_signature');self.assertFalse(reusable(analysis,signature))


class PipelineReuseTests(LocalPipelineTests):
    def first_run(self):
        core.main(self.arguments)
        first=self.completion();self.messages.clear()
        for spy in (self.video,self.shots,self.music,self.transcribe,self.frames,self.ocr,self.probe):spy.reset_mock()
        return first

    def test_second_run_with_same_signature_never_invokes_costly_stages(self):
        first=self.first_run()
        core.main(self.arguments)
        self.assertIs(self.completion()['cached'],True)
        for spy in (self.video,self.shots,self.music,self.transcribe,self.frames,self.ocr,self.probe):spy.assert_not_called()
        self.assertEqual(self.completion()['asset_id'],first['asset_id'])

    def test_cache_root_degradation_does_not_prevent_matching_cached_results(self):
        self.first_run()
        with patch.dict('os.environ', {'REELPACK_CACHE_PREPARE_ERROR': 'No writable Reelpack cache root'}):
            core.main(self.arguments)
        self.assertIs(self.completion()['cached'], True)
        self.transcribe.assert_not_called(); self.shots.assert_not_called(); self.frames.assert_not_called()
        self.assertTrue(any(item.get('stage') == 'runtime_cache' for item in self.messages))

    def test_language_change_really_calls_asr_and_overwrites_analysis_signature(self):
        first=self.first_run()
        core.main(self.arguments+['--language','zh'])
        self.assertIs(self.completion()['cached'],False)
        self.transcribe.assert_called_once()
        self.assertEqual(self.transcribe.call_args.args[2],'zh')
        saved=json.loads(Path(first['analysis']).read_text())
        self.assertEqual(saved['run_signature']['options']['language'],'zh')
        self.assertEqual(self.source.read_bytes(),self.original)

    def test_force_repeats_even_matching_request(self):
        self.first_run();core.main(self.arguments+['--force'])
        self.assertIs(self.completion()['cached'],False)
        self.transcribe.assert_called_once();self.shots.assert_called_once();self.ocr.assert_called_once()

    def test_required_failure_retries_with_matching_signature(self):
        first=self.first_run();path=Path(first['analysis']);saved=json.loads(path.read_text())
        saved['capabilities']['ocr']['status']='failed';path.write_text(json.dumps(saved))
        core.main(self.arguments);self.assertIs(self.completion()['cached'],False)
        self.transcribe.assert_called_once()

    def test_optional_music_failure_preserves_exit_and_printable_summary(self):
        self.music.return_value={'status':'failed','backend':'librosa','error':'cache failed'}
        core.main(self.arguments)
        saved=json.loads(Path(self.completion()['analysis']).read_text())
        self.assertEqual(saved['music']['status'],'failed')
        for key in ('reason','impact','remedy'):self.assertTrue(saved['music'][key])
        summary=json.loads(self.completion()['capability_summary'])
        self.assertEqual(summary['complete'],3)
        self.assertEqual(next(x for x in summary['capabilities'] if x['name']=='music')['status'],'failed')


if __name__=='__main__':unittest.main()
