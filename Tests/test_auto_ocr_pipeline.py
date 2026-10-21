import json
from pathlib import Path
from unittest.mock import patch
from test_core_local import LocalPipelineTests
import core

class AutoOCRPipelineTests(LocalPipelineTests):
    def test_auto_records_actual_backend_and_frame_provenance(self):
        args=list(self.arguments)
        args[args.index('--ocr-backend')+1]='auto'
        result={'status':'complete','backend':'rapidocr','model':'RapidOCR','params':{'strategy_version':'native-error-fallback-v1'},
                'items':[{'timestamp':2.5,'start':2.5,'end':2.5,'text':'中文','evidence_ids':['ocr_frame_00000'],
                          'backend':'rapidocr','fallback_from':'macos','reason':'native fixture error','status':'complete'}],
                'errors':[{'backend':'macos','reason':'native fixture error'}],
                'fallback_from':'macos','reason':'native fixture error','attempts':[{'backend':'macos','status':'failed'},{'backend':'rapidocr','status':'complete'}]}
        with patch('ocr_fallback.recognize_auto',return_value=result) as recognize:
            core.main(args)
        a=json.loads(Path(self.completion()['analysis']).read_text())
        self.assertEqual(a['ocr']['backend'],'rapidocr')
        self.assertEqual(a['ocr']['frame_provenance'][0]['fallback_from'],'macos')
        self.assertEqual(a['run_signature']['options']['ocr_backend'],'auto')
        self.assertEqual(a['capabilities']['ocr']['status'],'complete')
        self.assertEqual(a['capabilities']['ocr']['fallback_from'],'macos')
        self.assertEqual(a['ocr']['attempts'][0]['status'],'failed')
        self.messages.clear()
        with patch('ocr_fallback.recognize_auto',side_effect=AssertionError('cached run invoked OCR')):
            core.main(args)
        self.assertTrue(self.completion()['cached'])
