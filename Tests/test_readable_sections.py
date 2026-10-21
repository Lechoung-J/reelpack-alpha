import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Core'))
from readable_sections import screen_text_section,rewrite_section

class ReadableSectionsTests(unittest.TestCase):
    def test_screen_text_deduplicates_and_retains_first_timestamp(self):
        text=screen_text_section({'status':'complete','backend':'rapidocr','items':[
            {'timestamp':1.25,'text':'画面文字'}, {'timestamp':2,'text':'画面文字'},
            {'text':'无时间文字'}, {'text':' '}]})
        self.assertEqual(text.count('画面文字'),2) # heading and one observation
        self.assertIn('[1.25s]',text);self.assertNotIn('[2.00s]',text)
        self.assertIn('- 无时间文字',text)
    def test_empty_success_and_failure_are_distinct(self):
        self.assertIn('（无可见文字）',screen_text_section({'status':'complete','items':[]}))
        failed=screen_text_section({'status':'failed','items':[],'reason':'decoder failed'})
        self.assertNotIn('（无可见文字）',failed);self.assertIn('不可用',failed)
    def test_rewrite_does_not_invent_success_on_failure(self):
        self.assertIn('未整理',rewrite_section({'status':'failed','reason':'offline'}))
        good=rewrite_section({'status':'complete','text':'任督二脉。','params':{'mode':'short'},
            'edits':[{'segment_id':'s1','before':'二度认卖','after':'任督二脉','reason':'语境纠错'}]})
        self.assertIn('二度认卖',good);self.assertIn('任督二脉',good)

if __name__=='__main__':unittest.main()
