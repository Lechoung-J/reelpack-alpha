import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Core'))
from audio_alignment import align_cuts_to_beats


class AudioAlignmentTests(unittest.TestCase):
    def test_legacy_numbers_and_media_start_preserved(self):
        cuts = [0., 1.02, 2.3, 3.]
        beats = [0., 1., 2., 3.]
        result = align_cuts_to_beats(cuts, beats)
        self.assertEqual(result['status'], 'measured')
        self.assertEqual(result['aligned_cut_count'], 3)
        self.assertEqual(result['aligned_ratio'], 3 / 4)
        self.assertEqual(result['cut_count'], 4)
        self.assertTrue(result['includes_media_start'])
        self.assertAlmostEqual(result['mean_absolute_offset_seconds'], .08)
        self.assertEqual(result['units']['offsets'], 'seconds')

    def test_no_beats_is_unmeasured_not_zero_alignment(self):
        result = align_cuts_to_beats([0., 1., 2.], [])
        self.assertEqual(result['status'], 'not_measured')
        self.assertIsNone(result['aligned_cut_count'])
        self.assertIsNone(result['aligned_ratio'])
        self.assertIsNone(result['mean_signed_offset_seconds'])
        self.assertIsNone(result['mean_absolute_offset_seconds'])
        self.assertEqual(result['offsets_seconds'], [])
        self.assertEqual(result['cut_count'], 3)
        self.assertIn('no detected beats', result['reason'])

    def test_empty_cuts_keep_null_ratio(self):
        result = align_cuts_to_beats([], [1.])
        self.assertEqual(result['status'], 'not_measured')
        self.assertEqual(result['aligned_cut_count'], 0)
        self.assertIsNone(result['aligned_ratio'])

    def test_nearest_offset_sign_and_original_tie_order(self):
        result = align_cuts_to_beats([1.5, 1.9, 2.1], [1., 2.])
        self.assertEqual(result['offsets_seconds'][0], .5)
        self.assertAlmostEqual(result['offsets_seconds'][1], -.1)
        self.assertAlmostEqual(result['offsets_seconds'][2], .1)
        self.assertEqual(align_cuts_to_beats([1.5], [2., 1.])['offsets_seconds'], [-.5])
        self.assertEqual(align_cuts_to_beats([.15], [0.])['aligned_cut_count'], 1)

    def test_invalid_input_is_explicit(self):
        for tolerance in (-1., math.inf):
            with self.assertRaises(ValueError):
                align_cuts_to_beats([1.], [1.], tolerance)
        with self.assertRaises(ValueError):
            align_cuts_to_beats([math.nan], [1.])


if __name__ == '__main__':
    unittest.main()
