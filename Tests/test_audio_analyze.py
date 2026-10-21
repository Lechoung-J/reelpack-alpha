import unittest, numpy as np, sys
sys.path.insert(0,'Core')
from audio_analyze import analyze_array
from audio_alignment import align_cuts_to_beats
class AudioTests(unittest.TestCase):
 def test_alignment(self):
  r=align_cuts_to_beats([1.02,2.3],[1,2],.15); self.assertEqual(r['aligned_cut_count'],1)
 @unittest.skipUnless(__import__("importlib").util.find_spec("librosa"), "requires .venv-audio; run with audio interpreter")
 def test_synthetic_tone(self):
  sr=22050; t=np.arange(sr*4)/sr; y=np.zeros_like(t)
  y[(np.arange(len(t)) % int(sr*.5)) < int(sr*.02)] = .4*np.sin(2*np.pi*120*t[(np.arange(len(t)) % int(sr*.5)) < int(sr*.02)])
  r=analyze_array(y,sr); self.assertGreaterEqual(r['tempo_bpm'],0); self.assertIn('integrated_lufs',r['loudness'])
if __name__=='__main__': unittest.main()
