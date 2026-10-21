"""Default-off is a no-model/no-network contract; explicit modes remain functional."""
import copy
import inspect
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
from transcript_rewrite import rewrite_transcript
from pipeline_state import run_signature
import core
from test_core_local import LocalPipelineTests


class RewriteDefaultTests(unittest.TestCase):
    def setUp(self):
        self.raw = [{'segment_id':'s1','start':0.0,'end':2.0,'text':'原始口播，不应被默认改写。'}]

    def test_helper_defaults_off_without_contacting_local_or_cloud(self):
        self.assertEqual(inspect.signature(rewrite_transcript).parameters['mode'].default, 'off')
        for cloud in (False, True):
            session, generator = Mock(), Mock()
            before = copy.deepcopy(self.raw)
            value = rewrite_transcript(self.raw, use_cloud=cloud, session=session, generator=generator)
            self.assertEqual(value['status'], 'not_run')
            self.assertEqual(value['reason'], 'disabled')
            self.assertEqual(value['params']['mode'], 'off')
            self.assertEqual(value['raw_segments'], before)
            self.assertEqual(value['segments'], before)
            self.assertEqual(value['edits'], [])
            self.assertEqual(self.raw, before)
            generator.assert_not_called(); session.get.assert_not_called(); session.post.assert_not_called()

    def test_explicit_short_and_film_still_generate_and_preserve_source(self):
        for mode in ('short', 'film'):
            generator = Mock(return_value={'segments':[{'segment_id':'s1','text':'原始口播，应保留。'}]})
            value = rewrite_transcript(self.raw, mode=mode, generator=generator)
            self.assertEqual(value['status'], 'complete')
            self.assertEqual(value['params']['mode'], mode)
            generator.assert_called_once()
            self.assertEqual(value['raw_segments'], self.raw)
            self.assertEqual(value['edits'][0]['before'], self.raw[0]['text'])

    def test_default_signature_matches_explicit_off_but_not_explicit_modes(self):
        default = run_signature('a'*64, {})
        self.assertEqual(default['options']['rewrite'], 'off')
        self.assertEqual(default, run_signature('a'*64, {'rewrite':'off'}))
        for mode in ('short','film'):
            self.assertNotEqual(default, run_signature('a'*64, {'rewrite':mode}))

    def test_app_explicitly_passes_off_and_cli_help_names_off(self):
        root = Path(__file__).resolve().parents[1]
        self.assertIn('"--rewrite", "off"', (root/'App/UIWorkflow.swift').read_text())
        cli = (root/'CLI/main.swift').read_text()
        self.assertIn('var rewrite: String = "off"', cli)
        self.assertIn('off（默认）', cli)


class CoreRewriteDefaultTests(LocalPipelineTests):
    def test_default_core_pipeline_preserves_raw_without_model_requests(self):
        # Replace the optional rewrite mock with the real helper; all media work
        # remains the existing local fixture. Its network spy must see no calls.
        self.rewrite.side_effect = rewrite_transcript
        core.main(self.arguments)
        self.network.assert_not_called()
        self.assertEqual(self.rewrite.call_args.kwargs['mode'], 'off')
        output = self.completion()
        analysis = json.loads(Path(output['analysis']).read_text())
        self.assertEqual(analysis['run_signature']['options']['rewrite'], 'off')
        self.assertEqual(analysis['transcript']['rewrite']['reason'], 'disabled')
        self.assertEqual(analysis['transcript']['rewrite']['status'], 'not_run')
        self.assertEqual(analysis['transcript']['raw_segments'][0]['text'], 'synthetic speech')
        self.assertEqual(analysis['execution']['status'], 'complete')
        self.messages.clear(); self.rewrite.reset_mock()
        core.main(self.arguments+['--rewrite','off'])
        self.assertTrue(self.completion()['cached']); self.rewrite.assert_not_called()

    def test_explicit_modes_reach_real_helper_and_keep_parallel_results(self):
        for mode in ('short','film'):
            self.messages.clear()
            generator = Mock(return_value={'segments':[{'segment_id':'seg_1','text':'Synthetic speech.'}]})
            self.rewrite.side_effect = lambda segments, **kw: rewrite_transcript(segments, generator=generator, **kw)
            core.main(self.arguments+['--rewrite',mode])
            self.assertEqual(self.rewrite.call_args.kwargs['mode'],mode)
            output = self.completion()
            analysis = json.loads(Path(output['analysis']).read_text())
            self.assertEqual(analysis['transcript']['rewrite']['status'],'complete')
            self.assertEqual(analysis['transcript']['raw_segments'][0]['text'],'synthetic speech')
            self.assertIn('Synthetic speech.',Path(output['markdown']).read_text())
            generator.assert_called_once()
            self.network.assert_not_called()


if __name__ == '__main__': unittest.main()
