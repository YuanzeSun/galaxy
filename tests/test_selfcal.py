import unittest

import numpy as np

from galaxy_astro.selfcal import fit_stellar_response, radial_features, response_information


def observations(seed=5):
    rng = np.random.default_rng(seed)
    stars, frames = np.tile(np.arange(30), 12), np.repeat(np.arange(12), 30)
    features = radial_features(rng.uniform(-1.0, 1.0, (len(stars), 2)), [0.0, 0.0], 1.0)
    theta = np.array([-0.3, -0.09])
    y = features @ theta + rng.normal(7.0, 0.5, 30)[stars] + rng.normal(0.0, 0.02, 12)[frames]
    return y, features, stars, frames, np.full(len(y), 0.01), theta


class SelfCalibrationTests(unittest.TestCase):
    def test_sparse_stars_with_shared_sensor_shifts_recover_response(self):
        rng = np.random.default_rng(31)
        stars = np.tile(np.arange(6), 9)
        frames = np.repeat(np.arange(9), 6)
        positions = rng.uniform(-0.65, 0.65, (6, 2))
        shifts = rng.uniform(-0.2, 0.2, (9, 2))
        features = radial_features(positions[stars] + shifts[frames], [0.0, 0.0], 1.0)
        truth = np.array([-0.25, -0.08])
        y = features @ truth + rng.normal(7.0, 0.5, 6)[stars] + rng.normal(0.0, 0.02, 9)[frames]
        fit = fit_stellar_response(y, features, stars, frames, np.full(len(y), 0.01))
        np.testing.assert_allclose(fit.coefficients, truth, atol=1e-8)

    def test_sparse_stars_without_sensor_motion_cannot_calibrate_response(self):
        stars = np.tile(np.arange(6), 9)
        frames = np.repeat(np.arange(9), 6)
        positions = np.random.default_rng(31).uniform(-0.65, 0.65, (6, 2))
        features = radial_features(positions[stars], [0.0, 0.0], 1.0)
        info = response_information(features, stars, frames, np.full(len(stars), 0.01))
        self.assertFalse(info["necessary_identifiability_check_passed"])
        with self.assertRaises(ValueError):
            fit_stellar_response(
                np.ones(len(stars)), features, stars, frames, np.full(len(stars), 0.01)
            )

    def test_recover_known_response_and_offsets(self):
        y, x, stars, frames, sigma, truth = observations()
        fit = fit_stellar_response(y, x, stars, frames, sigma)
        np.testing.assert_allclose(fit.coefficients, truth, atol=1e-8)
        np.testing.assert_allclose(fit.prediction, y, atol=1e-8)
        self.assertEqual(fit.frame_log_offsets[0], 0.0)
        self.assertTrue(fit.information["necessary_identifiability_check_passed"])

    def test_one_contaminated_measurement(self):
        y, x, stars, frames, sigma, truth = observations()
        y[15] += 3.0
        fit = fit_stellar_response(y, x, stars, frames, sigma, iterations=12)
        np.testing.assert_allclose(fit.coefficients, truth, atol=0.002)

    def test_no_radial_motion_rejected(self):
        y, _, stars, frames, sigma, _ = observations()
        radius = np.linspace(0.1, 1.0, 30)[stars]
        x = np.column_stack([radius, radius**2])
        info = response_information(x, stars, frames, sigma)
        self.assertFalse(info["necessary_identifiability_check_passed"])
        with self.assertRaises(ValueError):
            fit_stellar_response(y, x, stars, frames, sigma)

    def test_constant_feature_absorbed_by_nuisance(self):
        y, _, stars, frames, sigma, _ = observations()
        with self.assertRaises(ValueError):
            fit_stellar_response(y, np.ones((len(y), 1)), stars, frames, sigma)

    def test_disconnected_graph_rejected(self):
        y, x, stars, frames, sigma, _ = observations()
        stars = stars + (frames >= 6) * 30
        info = response_information(x, stars, frames, sigma)
        self.assertEqual(info["graph_components"], 2)
        with self.assertRaises(ValueError):
            fit_stellar_response(y, x, stars, frames, sigma)

    def test_malformed_and_invalid_errors(self):
        y, x, stars, frames, sigma, _ = observations()
        sigma[0] = 0
        with self.assertRaises(ValueError):
            response_information(x, stars, frames, sigma)
        with self.assertRaises(ValueError):
            radial_features([[1, 2]], [0, 0], 0)
