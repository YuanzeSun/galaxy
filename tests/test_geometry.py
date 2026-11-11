import unittest

import numpy as np

from galaxy_astro.geometry import (
    derivative_coefficients,
    grid_map,
    map_pixels,
    polynomial_map,
    solid_angle,
    solid_angle_grid,
)


def camera():
    return dict(
        inverse_coefficients=[[0, 0], [1, 0], [0, 1]],
        inverse_origin=[0, 0],
        inverse_scale=[100, 100],
        forward_coefficients=[[0, 0], [100, 0], [0, 100]],
        forward_origin=[0, 0],
        forward_scale=[1, 1],
    )


class GeometryTests(unittest.TestCase):
    def test_grid_matches_direct_polynomial_and_derivatives(self):
        rng = np.random.default_rng(9)
        coefficients = rng.normal(size=(15, 2))  # total degree four
        x, y = np.linspace(-20, 20, 13), np.linspace(-10, 10, 11)
        xx, yy = np.meshgrid(x, y)
        xy = np.stack([xx, yy], -1)
        expected, jac = polynomial_map(xy, coefficients, [3, -2], [25, 14], True)
        got = np.stack(grid_map(coefficients, x, y, [3, -2], [25, 14], dtype=np.float64), -1)
        np.testing.assert_allclose(got, expected, atol=1e-13)
        for axis in range(2):
            dc = derivative_coefficients(coefficients, axis, [25, 14])
            derivative = polynomial_map(xy, dc, [3, -2], [25, 14])
            np.testing.assert_allclose(derivative, jac[..., :, axis], atol=1e-13)
            step = np.zeros(2)
            step[axis] = 1e-4
            finite = (
                polynomial_map(xy + step, coefficients, [3, -2], [25, 14])
                - polynomial_map(xy - step, coefficients, [3, -2], [25, 14])
            ) / 2e-4
            np.testing.assert_allclose(finite, derivative, atol=1e-9)

    def test_camera_identity_and_rotated_jacobian(self):
        xy = np.array([[0.0, 0.0], [10.0, -20.0]])
        np.testing.assert_allclose(map_pixels(xy, np.eye(3), camera()), xy)
        angle = 0.1
        r = np.array(
            [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        )
        mapped, jac = map_pixels(xy, r, camera(), True)
        self.assertTrue(np.isfinite(mapped).all())
        for axis in range(2):
            step = np.zeros(2)
            step[axis] = 1e-4
            finite = (
                map_pixels(xy + step, r, camera()) - map_pixels(xy - step, r, camera())
            ) / 2e-4
            np.testing.assert_allclose(finite, jac[..., :, axis], atol=1e-9)

    def test_pinhole_solid_angle_and_grid(self):
        x, y = np.arange(-3, 4), np.arange(-2, 3)
        xx, yy = np.meshgrid(x, y)
        xy = np.stack([xx, yy], -1)
        expected = 1e-4 / (1 + (xx / 100.0) ** 2 + (yy / 100.0) ** 2) ** 1.5
        np.testing.assert_allclose(solid_angle(xy, camera()), expected, rtol=1e-12)
        np.testing.assert_allclose(solid_angle_grid(x, y, camera()), expected, rtol=2e-6)

    def test_bad_models(self):
        for coefficients in [[], [[0, 0], [1, 1]], [[np.nan, 0]]]:
            with self.assertRaises(ValueError):
                polynomial_map([[0, 0]], coefficients, [0, 0], 1)
        with self.assertRaises(ValueError):
            derivative_coefficients([[0, 0]], 2, 1)
        with self.assertRaises(ValueError):
            grid_map([[0, 0]], [0], [0], [0, 0], 0)
