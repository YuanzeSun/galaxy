import unittest

import numpy as np

from galaxy_astro.calibration import calibrate_linear, calibrated_cfa_bridge, quantize_unsigned_cfa


class CalibrationTests(unittest.TestCase):
    def test_negative_signal_and_variance(self):
        out, var, good = calibrate_linear([99.0, 100.0, 102.0], 100.0, 2.0, variance=8.0)
        np.testing.assert_array_equal(out, [-0.5, 0.0, 1.0])
        np.testing.assert_array_equal(var, [2.0, 2.0, 2.0])
        self.assertTrue(good.all())

    def test_invalid_response_and_variance(self):
        out, _, good = calibrate_linear(
            [2.0, 2.0, 2.0], 1.0, [0.0, -1.0, 1.0], variance=[1.0, 1.0, -1.0]
        )
        self.assertFalse(good.any())
        self.assertTrue(np.isnan(out).all())

    def test_nearest_error_bound_and_not_worse_than_truncation(self):
        x = np.random.default_rng(3).uniform(0.0, 16383.0, 100000)
        rounded = quantize_unsigned_cfa(x, 16383).astype(float)
        truncated = x.astype(np.uint16).astype(float)
        self.assertTrue(np.all(abs(rounded - x) <= 0.5))
        self.assertTrue(np.all((rounded - x) ** 2 <= (truncated - x) ** 2 + 1e-10))

    def test_ties_clipping_and_per_pixel_black(self):
        np.testing.assert_array_equal(
            quantize_unsigned_cfa([-0.5, 0.5, 1.5, 2.5, 20.0], 10), [0, 0, 2, 2, 10]
        )
        np.testing.assert_array_equal(
            calibrated_cfa_bridge([10.0, 10.0], [2.0, 3.0], [1.0, 2.0], 20), [9, 9]
        )

    def test_bad_inputs(self):
        for white in [0, -1, 65536, np.nan, 12.5]:
            with self.assertRaises(ValueError):
                quantize_unsigned_cfa([1.0], white)
        with self.assertRaises(ValueError):
            quantize_unsigned_cfa([np.inf], 100)
        with self.assertRaises(ValueError):
            calibrated_cfa_bridge([1.0], [1.0, 2.0], 1.0, 10)
