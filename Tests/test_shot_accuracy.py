"""Synthetic frame sequences verify shot-boundary selection behavior."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
from shot_detect import select_boundaries
from shot_detect_transnet import select_transnet_boundaries


class AccuracyTests(unittest.TestCase):
    def test_known_frame_cuts_have_exact_time_mapping(self):
        features = np.zeros((120, 3))
        features[24] = (.8, .8, .5)
        features[96] = (.7, .7, .4)
        scores = np.zeros(120)
        scores[[24, 96]] = .9
        custom, _ = select_boundaries(features, 24)
        transnet = select_transnet_boundaries(scores)
        for cuts in (custom, transnet):
            self.assertEqual([cut[0] for cut in cuts], [24, 96])
            self.assertEqual([cut[0] / 24 for cut in cuts], [1.0, 4.0])

    def test_media_start_is_not_a_detected_cut(self):
        features = np.zeros((60, 3))
        features[[0, 24]] = (.8, .8, .5)
        scores = np.zeros(60)
        scores[[0, 24]] = .9
        cuts, _ = select_boundaries(features, 24)
        self.assertEqual([cut[0] for cut in cuts], [24])
        self.assertEqual([cut[0] for cut in select_transnet_boundaries(scores)], [24])

    def test_high_frame_rate_keeps_cuts_at_minimum_spacing(self):
        features = np.zeros((90, 3))
        features[20] = (.9, .9, .6)
        features[30] = (.8, .8, .5)
        cuts, parameters = select_boundaries(features, 60)
        self.assertEqual([cut[0] for cut in cuts], [20, 30])
        self.assertEqual(parameters['minimum_shot_seconds'], .16)

    def test_transnet_rejects_nonfinite_and_nonvector_predictions(self):
        for scores in ([float('nan')], [float('inf')], [[.9, .1]]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                select_transnet_boundaries(scores)

    def test_empty_and_constant_sequences_have_no_boundaries(self):
        self.assertEqual(select_boundaries([], 24), ([], {}))
        self.assertEqual(select_transnet_boundaries([]), [])
        cuts, _ = select_boundaries(np.zeros((48, 3)), 24)
        self.assertEqual(cuts, [])
        self.assertEqual(select_transnet_boundaries(np.zeros(48)), [])


if __name__ == '__main__':
    unittest.main()
