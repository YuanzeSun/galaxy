"""Explicit protected samples and a global additive plane; no flat-field fitting."""

from dataclasses import dataclass

import numpy as np


def sample_background_cells(image, protected, cell_size=64):
    """Return xy, channel medians, and integer cell coordinates.

    ANY protected pixel excludes the whole cell. Empty/nonfinite pixels do
    not become artificial dark samples. Object masks must be supplied by the
    caller; there is no fixed Galactic mask suitable for every target.
    """
    image = np.asarray(image, float)
    protected = np.asarray(protected, bool)
    if image.ndim == 2:
        image = image[..., None]
    if (
        image.ndim != 3
        or min(image.shape) == 0
        or image.shape[:2] != protected.shape
        or not isinstance(cell_size, int)
        or cell_size < 1
    ):
        raise ValueError("Image HxWxC, matching protection mask and positive cell size required")
    h, w, channels = image.shape
    xy, values, cells = [], [], []
    for iy, y0 in enumerate(range(0, h, cell_size)):
        for ix, x0 in enumerate(range(0, w, cell_size)):
            y1, x1 = min(h, y0 + cell_size), min(w, x0 + cell_size)
            if protected[y0:y1, x0:x1].any():
                continue
            pixels = image[y0:y1, x0:x1].reshape(-1, channels)
            pixels = pixels[np.isfinite(pixels).all(axis=1)]
            if len(pixels):
                xy.append([(x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2])
                values.append(np.median(pixels, axis=0))
                cells.append([ix, iy])
    return (
        np.asarray(xy, float).reshape(-1, 2),
        np.asarray(values, float).reshape(-1, channels),
        np.asarray(cells, int).reshape(-1, 2),
    )


def _plane_design(xy, shape):
    h, w = shape
    xy = np.asarray(xy, float)
    if h < 2 or w < 2 or xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Finite Nx2 coordinates and image dimensions >=2 required")
    return np.column_stack(
        [np.ones(len(xy)), (xy[:, 0] - (w - 1) / 2) / (w - 1), (xy[:, 1] - (h - 1) / 2) / (h - 1)]
    )


@dataclass(frozen=True)
class BackgroundPlane:
    coefficients: np.ndarray
    shape: tuple

    def predict(self, xy):
        return _plane_design(xy, self.shape) @ self.coefficients


def fit_background_plane(xy, values, shape, *, iterations=6):
    """Fit one robust additive plane per channel to caller-approved samples.

    A fit is an estimate, not an adoption decision or a celestial zero point.
    Protected pixels must be excluded before calling, including any bounds.
    """
    x = _plane_design(xy, shape)
    y = np.asarray(values, float)
    if y.ndim == 1:
        y = y[:, None]
    if (
        y.ndim != 2
        or len(y) != len(x)
        or not np.isfinite(y).all()
        or len(x) < 3
        or not isinstance(iterations, int)
        or iterations < 1
        or np.linalg.matrix_rank(x) != 3
    ):
        raise ValueError("Finite matching samples and a full-rank plane required")
    weights = np.ones(len(x))
    for _ in range(iterations):
        sw = np.sqrt(weights)
        coefficient = np.linalg.lstsq(x * sw[:, None], y * sw[:, None], rcond=None)[0]
        residual = np.linalg.norm(x @ coefficient - y, axis=1)
        med = np.median(residual)
        mad = max(1e-12, 1.4826 * np.median(abs(residual - med)))
        weights = np.minimum(1.0, (med + 2.7 * mad) / np.maximum(residual, 1e-12))
    return BackgroundPlane(coefficient, tuple(shape))
