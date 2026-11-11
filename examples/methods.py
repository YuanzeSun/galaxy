"""Run with: python examples/methods.py (after installing the package)."""

import numpy as np

from galaxy_astro.calibration import calibrate_linear
from galaxy_astro.selfcal import fit_stellar_response, radial_features
from galaxy_astro.stacking import combine_group_means

rng = np.random.default_rng(7)
stars = np.tile(np.arange(20), 8)
frames = np.repeat(np.arange(8), 20)
xy = rng.uniform(-1.0, 1.0, size=(len(stars), 2))
features = radial_features(xy, center=(0.0, 0.0), scale=1.0)
truth = np.array([-0.2, -0.05])
flux = features @ truth + rng.normal(8.0, 0.5, 20)[stars] + rng.normal(0.0, 0.02, 8)[frames]
fit = fit_stellar_response(flux, features, stars, frames, np.full(len(stars), 0.01))
print("Synthetic response:", fit.coefficients, "; known truth:", truth)

raw = np.array([[101.0, 99.0], [102.0, 98.0], [100.0, 100.0]])
signal, variance, valid = calibrate_linear(raw, dark=100.0, variance=4.0)
combined = combine_group_means(signal[..., None], valid.astype(int))
print("Stack:", combined["mean"][:, 0], "; coverage:", combined["coverage"])
print("No real files were read or written.")
