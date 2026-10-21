import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Core'))
from shot_detect import activity_mask, select_boundaries
import numpy as np

class AdaptiveShotTests(unittest.TestCase):
    def test_packaging_is_masked(self):
        samples=np.zeros((20,90,160,3),dtype=np.uint8)
        for i in range(20): samples[i,30:60]=i*10
        mask,region,static,fallback=activity_mask(samples)
        self.assertFalse(fallback)
        self.assertTrue(static)
        self.assertEqual(mask[0].sum(),0)
        self.assertGreater(mask[45].sum(),0)
        self.assertLess(region['height'],.6)

    def test_static_screen_falls_back(self):
        mask,region,static,fallback=activity_mask(np.zeros((10,90,160,3)))
        self.assertTrue(fallback)
        self.assertTrue(mask.all())
        self.assertFalse(static)

    def test_fast_cuts_survive_and_motion_is_rejected(self):
        f=np.zeros((100,3)); f[:,1]=.15
        f[20]=(.8,.8,.5); f[26]=(.7,.8,.4)
        cuts,params=select_boundaries(f,30)
        self.assertEqual([c[0] for c in cuts],[20,26])
        self.assertTrue(all(0<=c[2]<=1 for c in cuts))

    def test_debounce_keeps_stronger_peak(self):
        f=np.zeros((60,3)); f[20]=(.6,.6,.4); f[23]=(.9,.9,.5)
        cuts,_=select_boundaries(f,30)
        self.assertEqual([c[0] for c in cuts],[23])
