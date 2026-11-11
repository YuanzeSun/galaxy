import unittest

import numpy as np

from galaxy_astro.background import fit_background_plane, sample_background_cells
from galaxy_astro.photometry import aperture_measure
from galaxy_astro.validation import (
    block_mean_residuals,
    bootstrap_mean,
    one_standard_error_choice,
    spatial_folds,
)


class BackgroundTests(unittest.TestCase):
    def test_off_center_bright_nebula_and_dark_cloud_share_custom_protection(self):
        yy, xx = np.mgrid[:79, :127]
        plane = 12.0 + 0.04 * xx - 0.03 * yy
        bright_radius = ((xx - 18) / 23.0) ** 2 + ((yy - 62) / 19.0) ** 2
        dark_radius = ((xx - 94) / 14.0) ** 2 + ((yy - 17) / 11.0) ** 2
        bright = 8.0 * np.maximum(1.0 - bright_radius, 0.0) ** 2
        dark = -4.0 * np.maximum(1.0 - dark_radius, 0.0) ** 2
        signal = bright + dark
        protected = (bright_radius < 1.0) | (dark_radius < 1.0)
        xy, values, _ = sample_background_cells(plane + signal, protected, cell_size=8)
        model = fit_background_plane(xy, values, plane.shape)
        predicted = model.predict(np.column_stack([xx.ravel(), yy.ravel()]))[:, 0].reshape(
            plane.shape
        )
        np.testing.assert_allclose(predicted, plane, atol=1e-12)
        np.testing.assert_allclose(plane + signal - predicted, signal, atol=1e-12)

    def test_whole_cell_protection_and_refit_injection(self):
        yy, xx = np.mgrid[:64, :80]
        plane = 3.0 + 0.02 * xx - 0.03 * yy
        protected = np.zeros(plane.shape, bool)
        protected[17:36, 19:48] = True
        xy, values, cells = sample_background_cells(plane, protected, 8)
        for ix, iy in cells:
            self.assertFalse(protected[iy * 8 : (iy + 1) * 8, ix * 8 : (ix + 1) * 8].any())
        first = fit_background_plane(xy, values, plane.shape)
        injected = plane.copy()
        injected[protected] += 100.0
        xy2, values2, cells2 = sample_background_cells(injected, protected, 8)
        second = fit_background_plane(xy2, values2, plane.shape)
        np.testing.assert_array_equal(xy, xy2)
        np.testing.assert_array_equal(values, values2)
        np.testing.assert_array_equal(cells, cells2)
        np.testing.assert_array_equal(first.coefficients, second.coefficients)
        allxy = np.column_stack([xx.ravel(), yy.ravel()])
        corrected = injected - second.predict(allxy)[:, 0].reshape(plane.shape)
        np.testing.assert_allclose(corrected[protected], 100.0, atol=1e-12)

    def test_all_protected_has_no_fake_samples(self):
        xy, values, cells = sample_background_cells(np.ones((10, 10)), np.ones((10, 10), bool), 4)
        self.assertEqual(xy.shape, (0, 2))
        self.assertEqual(values.shape, (0, 1))
        with self.assertRaises(ValueError):
            fit_background_plane(xy, values, (10, 10))

    def test_forced_aperture_on_sloping_background(self):
        yy, xx = np.mgrid[:170, :170]
        star = 100.0 * np.exp(-((xx - 85) ** 2 + (yy - 85) ** 2) / (2 * 2.0**2))
        flux, error, noise = aperture_measure(
            star + 30 + 0.1 * xx - 0.03 * yy, [[85, 85]], radii=(12,)
        )
        self.assertAlmostEqual(flux[0, 0], star.sum(), delta=0.001)
        self.assertTrue(np.isfinite(error).all())
        self.assertTrue(np.isfinite(noise).all())


class ValidationTests(unittest.TestCase):
    def test_spatial_blocks_and_signed_residual_means(self):
        cells = np.array([[0, 0], [1, 1], [4, 0], [5, 1]])
        folds = spatial_folds(cells)
        self.assertEqual(folds[0], folds[1])
        self.assertNotEqual(folds[0], folds[2])
        blocks, means = block_mean_residuals([[1, 2], [-1, 2], [4, 6], [0, 2]], cells)
        np.testing.assert_array_equal(blocks, [[0, 0], [1, 0]])
        np.testing.assert_array_equal(means, [[0, 2], [2, 4]])

    def test_bootstrap_deterministic_and_coarsest_within_one_se(self):
        a = bootstrap_mean([1, 2, 4, 5, 8], [0, 0, 1, 1, 2], draws=200, seed=9)
        b = bootstrap_mean([1, 2, 4, 5, 8], [0, 0, 1, 1, 2], draws=200, seed=9)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a["mean"], 4.0)
        self.assertGreater(a["standard_error"], 0.0)
        self.assertEqual(
            one_standard_error_choice([1.1, 1.0, 0.98], [0.05, 0.1, 0.15], [1, 2, 3]), 0
        )
        with self.assertRaises(ValueError):
            bootstrap_mean([1, 2], [0, 0])
