import copy,json
from pathlib import Path
from test_core_local import LocalPipelineTests
import core

class RewritePipelineTests(LocalPipelineTests):
    def test_original_rewrite_and_screen_text_survive_export_package_and_cache(self):
        self.arguments += ['--rewrite', 'short']
        raw='你把我这套东西你把我的二度认卖打通了'
        clean='你把我的任督二脉打通了。'
        self.transcribe.return_value=([{'start':1,'end':2,'text':raw}],{'language':'zh','params':{}})
        self.rewrite.return_value={'status':'complete','backend':'ollama','text':clean,'segments':[{'start':1,'end':2,'text':clean}],
            'params':{'mode':'short'},'edits':[{'segment_id':0,'before':raw,'after':clean,'reason':'语境术语纠正'}]}
        self.ocr.return_value.update(items=[{'timestamp':2.5,'start':2.5,'end':2.5,'text':'字幕证据','evidence_ids':['ocr_frame_00000']}])
        core.main(self.arguments)
        result=self.completion();a=json.loads(Path(result['analysis']).read_text())
        self.assertEqual(a['transcript']['raw_segments'][0]['text'],raw)
        self.assertEqual(a['transcript']['rewrite']['text'],clean)
        for path in [Path(result['markdown']),Path(result['package'])/'analysis.md']:
            md=path.read_text();self.assertIn('## 原始转录',md);self.assertIn('## 整理稿',md)
            self.assertIn('## 画面文字',md);self.assertIn(raw,md);self.assertIn(clean,md);self.assertIn('字幕证据',md)
        self.messages.clear();self.rewrite.reset_mock();core.main(self.arguments)
        self.assertTrue(self.completion()['cached']);self.rewrite.assert_not_called()
        self.messages.clear();core.main(self.arguments+['--rewrite','film'])
        self.assertFalse(self.completion()['cached'])

    def test_missing_rewrite_model_keeps_raw_and_empty_ocr_section(self):
        core.main(self.arguments+['--rewrite','short'])
        result=self.completion();md=Path(result['markdown']).read_text()
        self.assertIn('synthetic speech',md);self.assertIn('未整理',md)
        self.assertIn('（无可见文字）',md)
        self.assertEqual(json.loads(Path(result['analysis']).read_text())['execution']['status'],'complete')
