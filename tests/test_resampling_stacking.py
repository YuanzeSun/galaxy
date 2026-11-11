import unittest

import numpy as np

from galaxy_astro.resampling import resample_cubic
from galaxy_astro.stacking import combine_group_means, weighted_mean


class ResamplingTests(unittest.TestCase):
    def test_plane_and_integer_sample_variance(self):
        yy, xx = np.mgrid[:20, :30]
        image = 3.0 + 2.0 * xx - 0.5 * yy
        xy = np.array([[4.3, 7.4], [10.2, 12.7], [8.0, 8.0]])
        got, var, good = resample_cubic(image, xy, variance=np.full(image.shape, 2.0))
        # Keys a=-0.75 does not reproduce arbitrary linear ramps exactly.
        self.assertTrue(good.all())
        self.assertTrue(np.isfinite(got).all())
        self.assertAlmostEqual(got[-1], image[8, 8])
        self.assertAlmostEqual(var[-1], 2.0)
        constant, _, _ = resample_cubic(np.full(image.shape, -2.0), xy)
        np.testing.assert_allclose(constant, -2.0, atol=1e-12)

    def test_invalid_support_is_not_filled_or_renormalized(self):
        image = np.ones((8, 8))
        valid = np.ones(image.shape, bool)
        valid[3, 3] = False
        got, _, good = resample_cubic(image, [[3.2, 3.3], [0.2, 0.2], [np.nan, 2.0]], valid=valid)
        self.assertFalse(good.any())
        self.assertTrue(np.isnan(got).all())

    def test_area_changes_variance_squared(self):
        image = np.full((8, 8), 5.0)
        got, var, _ = resample_cubic(image, [[3.0, 3.0]], variance=np.ones((8, 8)), area_scale=2.0)
        np.testing.assert_allclose(got, [10.0])
        np.testing.assert_allclose(var, [4.0])


class StackingTests(unittest.TestCase):
    def test_inverse_variance_and_empty_pixels(self):
        values = [[0.0, np.nan], [-2.0, 3.0], [2.0, 4.0]]
        valid = [[True, False], [True, False], [True, False]]
        result = weighted_mean(values, [[1.0, 1.0], [2.0, 1.0], [2.0, 1.0]], valid)
        self.assertAlmostEqual(result["mean"][0], 0.0)
        self.assertAlmostEqual(result["variance"][0], 0.5)
        self.assertEqual(result["count"][0], 3)
        self.assertEqual(result["count"][1], 0)
        self.assertTrue(np.isnan(result["mean"][1]))

    def test_unequal_coverage_negative_values_and_nan_zero_count(self):
        values = np.array([[[0.0, -2.0], [np.nan, np.nan]], [[4.0, 2.0], [3.0, 4.0]]])
        result = combine_group_means(values, [[1, 0], [3, 0]])
        np.testing.assert_allclose(result["mean"][0], [3.0, 1.0])
        self.assertEqual(result["coverage"][0], 4)
        self.assertTrue(np.isnan(result["mean"][1]).all())
        self.assertEqual(result["accepted_count"][1], 0)

    def test_explicit_clipping_and_actual_vector_fallback(self):
        result = combine_group_means(
            np.array([[[1.0, 1.0]], [[1.0, 1.0]], [[20.0, 20.0]]]),
            np.ones((3, 1)),
            clip_sigma=3.5,
            scale_floor=0.1,
        )
        np.testing.assert_allclose(result["mean"], [[1.0, 1.0]])
        self.assertEqual(result["accepted_count"][0], 2)
        vectors = np.array([[[0.0, 10.0, 10.0]], [[10.0, 0.0, 10.0]], [[10.0, 10.0, 0.0]]])
        result = combine_group_means(vectors, np.ones((3, 1)), clip_sigma=3.5, scale_floor=0.1)
        self.assertTrue(result["fallback"][0])
        self.assertEqual(result["accepted_count"][0], 1)
        self.assertTrue(any(np.array_equal(result["mean"][0], v[0]) for v in vectors))

    def test_clipping_requires_signal_scale(self):
        with self.assertRaises(ValueError):
            combine_group_means(np.ones((3, 2, 3)), np.ones((3, 2)), clip_sigma=3.5)
