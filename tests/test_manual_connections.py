import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

if importlib.util.find_spec("astropy") is None:
    raise unittest.SkipTest("Install .[manual] to test the optional FITS/WCS handoffs")

from astropy.io import fits
from astropy.wcs import WCS, Sip

from examples.connect_steps import worked_example
from examples.manual_steps import (
    join_rows,
    measure_rows,
    pixel_solid_angles,
    read_linear_fits,
    reference_to_source,
    register_surface_brightness,
)
from galaxy_astro.background import BackgroundPlane, fit_background_plane, sample_background_cells


def toy_wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [120.0, 20.0]
    wcs.wcs.crpix = [30.5, 20.5]
    wcs.wcs.cdelt = [-0.02, 0.02]
    return wcs


class ManualConnectionTests(unittest.TestCase):
    def test_other_shapes_scales_rotations_and_ra_wrap(self):
        for shape, dec in [((37, 91), -35.0), ((83, 45), 82.0)]:
            with self.subTest(shape=shape, dec=dec):
                reference, source = toy_wcs(), toy_wcs()
                reference.wcs.crval = [359.85, dec]
                source.wcs.crval = [0.05, dec + 0.04]
                reference.wcs.crpix = [(shape[1] + 1) / 2, (shape[0] + 1) / 2]
                source.wcs.crpix = [17.0, 13.0]
                source.wcs.cdelt = [-0.006, 0.008]
                angle = 0.15
                source.wcs.pc = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                mapped = reference_to_source(reference, source, shape)
                sky = source.pixel_to_world(mapped[..., 0], mapped[..., 1])
                rx, ry = reference.world_to_pixel(sky)
                yy, xx = np.mgrid[: shape[0], : shape[1]]
                np.testing.assert_allclose(rx, xx, atol=1e-7)
                np.testing.assert_allclose(ry, yy, atol=1e-7)

    def test_wcs_distortion_is_used_in_maps_and_pixel_areas(self):
        wcs = toy_wcs()
        wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
        a, b = np.zeros((4, 4)), np.zeros((4, 4))
        a[2, 0], b[0, 2] = 2e-5, -3e-5
        wcs.sip = Sip(a, b, None, None, wcs.wcs.crpix)
        yy, xx = np.mgrid[:40, :60]
        mapped = reference_to_source(wcs, wcs, (40, 60))
        np.testing.assert_allclose(mapped, np.stack([xx, yy], -1), atol=1e-4)
        x, y = xx - 29.5, yy - 19.5
        scale = np.deg2rad(0.02)
        u, v = scale * (x + a[2, 0] * x * x), scale * (y + b[0, 2] * y * y)
        determinant = scale**2 * (1 + 2 * a[2, 0] * x) * (1 + 2 * b[0, 2] * y)
        expected = determinant / (1 + u * u + v * v) ** 1.5
        np.testing.assert_allclose(pixel_solid_angles(wcs, (40, 60)), expected, rtol=1e-7)

    def test_monochrome_input_can_use_the_same_registration_interface(self):
        image = np.arange(100, dtype=float).reshape(10, 10) - 70.0
        xy = np.array([[[3.0, 4.0], [5.0, 6.0]]])
        out, valid = register_surface_brightness(image, xy)
        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(out, [[image[4, 3], image[6, 5]]])

    def test_relative_background_handoff_keeps_common_signal_and_zero_mean(self):
        yy, xx = np.mgrid[:64, :80]
        common = 20.0 + 5.0 * np.exp(-((xx - 40) ** 2 + (yy - 32) ** 2) / (2 * 8.0**2))
        backgrounds = [2.0 + 0.02 * xx, -3.0 + 0.03 * yy, 1.0 - 0.01 * xx - 0.02 * yy]
        images = [(common + background)[..., None] * [0.8, 1.0, 1.2] for background in backgrounds]
        protected = (xx - 40) ** 2 + (yy - 32) ** 2 < 15.0**2
        models = []
        for image in images:
            xy, delta, _ = sample_background_cells(image - images[0], protected, cell_size=8)
            models.append(fit_background_plane(xy, delta, common.shape))
        coefficients = np.stack([model.coefficients for model in models])
        pixels = np.column_stack([xx.ravel(), yy.ravel()])
        matched = []
        for index, image in enumerate(images):
            relative = coefficients[index] - coefficients.mean(axis=0)
            correction = (
                BackgroundPlane(relative, common.shape).predict(pixels).reshape(image.shape)
            )
            matched.append(image - correction)
        expected = np.mean(images, axis=0)
        for image in matched:
            np.testing.assert_allclose(image, expected, atol=1e-12)

    def test_reference_to_source_direction_and_full_image_row_coordinates(self):
        reference, source = toy_wcs(), toy_wcs()
        source.wcs.crpix += [3.0, -2.0]
        got = reference_to_source(reference, source, (40, 60), y_start=13, y_stop=19)
        yy, xx = np.mgrid[13:19, :60]
        np.testing.assert_allclose(got, np.stack([xx + 3, yy - 2], -1), atol=1e-9)

    def test_pixel_areas_match_analytic_tangent_projection_and_tiled_evaluation(self):
        wcs = toy_wcs()
        yy, xx = np.mgrid[:40, :60]
        scale = np.deg2rad(0.02)
        r2 = ((xx - 29.5) * scale) ** 2 + ((yy - 19.5) * scale) ** 2
        expected = scale**2 / (1 + r2) ** 1.5
        full = pixel_solid_angles(wcs, (40, 60))
        np.testing.assert_allclose(full, expected, rtol=1e-7)
        tiles = [
            pixel_solid_angles(wcs, (40, 60), y_start=y, y_stop=min(y + 7, 40))
            for y in range(0, 40, 7)
        ]
        np.testing.assert_allclose(np.concatenate(tiles), full, rtol=1e-12)

    def test_read_fits_keeps_pixels_and_original_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "linear.fits"
            data = np.arange(3 * 40 * 60, dtype=np.float32).reshape(3, 40, 60) - 100.0
            fits.PrimaryHDU(data, header=toy_wcs().to_header()).writeto(file)
            before, mtime = hashlib.sha256(file.read_bytes()).hexdigest(), file.stat().st_mtime_ns
            image, wcs = read_linear_fits(file)
            np.testing.assert_array_equal(image, np.moveaxis(data, 0, -1))
            self.assertTrue(wcs.has_celestial)
            self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), before)
            self.assertEqual(file.stat().st_mtime_ns, mtime)

    def test_photometry_quality_mask_nonfinite_positions_and_exact_string_ids(self):
        yy, xx = np.mgrid[:80, :90]
        image = 100 * np.exp(-((xx - 45) ** 2 + (yy - 40) ** 2) / 2) + 20.0
        ids = np.array(["6123456789012345678", "excluded", "undefined"])
        rows, selected = measure_rows(
            image,
            [[45, 40], [45, 40], [np.nan, 20]],
            ids,
            "frame-01",
            [True, False, True],
            radius=5,
            annulus=(8, 12),
            minimum_snr=20,
            fractional_error_floor=0.01,
        )
        np.testing.assert_array_equal(selected, [True, False, False])
        self.assertEqual(rows["star_id"].tolist(), [ids[0]])
        self.assertTrue(np.isfinite(rows["sigma_log"]).all())
        joined = join_rows([rows, rows])
        self.assertEqual(joined["star_id"].tolist(), [ids[0], ids[0]])

    def test_registration_uses_shared_channel_validity_and_keeps_negative_signal(self):
        image = np.ones((10, 10, 3))
        image[..., 2] = -1.0
        image[3, 3, 0] = np.nan
        result, valid = register_surface_brightness(image, np.array([[3.0, 3.0], [6.0, 6.0]]))
        np.testing.assert_array_equal(valid, [False, True])
        self.assertTrue(np.isnan(result[0]).all())
        np.testing.assert_array_equal(result[1], [1.0, 1.0, -1.0])

    def test_invalid_row_ranges_and_unsolved_fits_are_refused(self):
        with self.assertRaises(ValueError):
            reference_to_source(toy_wcs(), toy_wcs(), (40, 60), y_start=10, y_stop=9)
        with self.assertRaises(ValueError):
            pixel_solid_angles(toy_wcs(), (40, 60), y_start=-1)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "unsolved.fits"
            fits.PrimaryHDU(np.ones((8, 8))).writeto(file)
            with self.assertRaises(ValueError):
                read_linear_fits(file)

    def test_known_truth_photometry_response_registration_stack_handoff(self):
        report = worked_example()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["measured_rows"], 160)
        self.assertEqual(report["held_stars"], 5)
        self.assertLess(report["maximum_coefficient_error"], 0.002)
        self.assertLess(report["held_star_log_flux_scatter_max"], 1e-4)
        self.assertFalse(report["background_candidate_applied"])
