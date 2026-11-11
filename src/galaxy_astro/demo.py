"""Small known-truth checks; no photographs or network required."""

import numpy as np

from .background import fit_background_plane, sample_background_cells
from .calibration import calibrate_linear, quantize_unsigned_cfa
from .selfcal import fit_stellar_response, radial_features
from .stacking import combine_group_means


def synthetic_demo(seed=42):
    rng = np.random.default_rng(seed)
    # A tiny toy sensor; these are not calibration parameters for a real camera.
    h, w = 96, 128
    yy, xx = np.mgrid[:h, :w]
    plane = 20.0 + 0.02 * xx + 0.03 * yy
    source = 3.0 * np.exp(-((xx - 64) ** 2 + (yy - 48) ** 2) / (2 * 9.0**2))
    protected = (xx - 64) ** 2 + (yy - 48) ** 2 < 35.0**2
    xy, sky, _ = sample_background_cells(plane + source, protected, cell_size=8)
    model = fit_background_plane(xy, sky, (h, w))
    predicted = model.predict(np.column_stack([xx.ravel(), yy.ravel()]))[:, 0].reshape(h, w)
    sky_error = float(abs(predicted - plane).max())

    stars = np.tile(np.arange(24), 10)
    frames = np.repeat(np.arange(10), 24)
    positions = rng.uniform([0.0, 0.0], [w - 1.0, h - 1.0], size=(len(stars), 2))
    features = radial_features(positions, ((w - 1) / 2, (h - 1) / 2), w / 2)
    theta = np.array([-0.2, -0.08])
    true_star = rng.normal(7.0, 0.5, 24)
    transparency = rng.normal(0.0, 0.02, 10)
    log_flux = features @ theta + true_star[stars] + transparency[frames]
    fitted = fit_stellar_response(log_flux, features, stars, frames, np.full(len(stars), 0.01))
    response_error = float(abs(fitted.coefficients - theta).max())

    response = np.exp(-0.3 * ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (w / 2) ** 2)
    raw = (plane + source) * response + 100.0
    calibrated, _, _ = calibrate_linear(raw, 100.0, response)
    calibration_error = float(abs(calibrated - (plane + source)).max())
    v = np.stack([calibrated, calibrated, calibrated])[..., None]
    result = combine_group_means(v, np.ones(v.shape[:-1], int))
    stack_error = float(abs(result["mean"][..., 0] - calibrated).max())

    values = rng.uniform(1.0, 1000.0, 10000)
    nearest = quantize_unsigned_cfa(values, 65535).astype(float)
    floor = values.astype(np.uint16).astype(float)
    rounding = bool(np.all((nearest - values) ** 2 <= (floor - values) ** 2 + 1e-12))
    checks = dict(
        known_response_max_error=response_error,
        linear_calibration_max_error=calibration_error,
        protected_background_plane_max_error=sky_error,
        constant_stack_max_error=stack_error,
        nearest_rounding_squared_error_never_worse=rounding,
    )
    passed = (
        response_error < 1e-7
        and calibration_error < 1e-10
        and sky_error < 0.01
        and stack_error < 1e-10
        and rounding
    )
    return dict(
        synthetic_only=True,
        seed=seed,
        passed=bool(passed),
        checks=checks,
        limitation="Known-truth software checks, not proof of real-sky flat or diffuse-flux recovery.",
    )
