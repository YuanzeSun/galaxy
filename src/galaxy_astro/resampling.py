"""Continuous cubic reference sampler with explicit support and noise limits."""

import numpy as np


def resample_cubic(image, source_xy, variance=None, valid=None, area_scale=None):
    """Reference Keys cubic sampler, with validity and diagonal noise propagation.

    This uses continuous coordinates (unlike OpenCV's 1/32-pixel coefficient
    table). Any invalid nonzero contributor invalidates an output pixel.
    No renormalization across bad pixels or image borders. Negative values are
    retained. For count-per-source-pixel units an appropriate local abs(det J)
    may be passed as area_scale; omit it to interpolate surface brightness.
    This area approximation is not exact pixel-overlap drizzle. Neighbouring
    output samples share inputs and therefore have correlated noise.
    """
    image = np.asarray(image, dtype=float)
    source_xy = np.asarray(source_xy, dtype=float)
    if image.ndim != 2 or source_xy.shape[-1] != 2:
        raise ValueError("Expected a 2-D image and (..., 2) coordinate map.")
    shape = source_xy.shape[:-1]
    finite = np.all(np.isfinite(source_xy), axis=-1)
    x = np.where(finite, source_xy[..., 0], -10.0)
    y = np.where(finite, source_xy[..., 1], -10.0)
    finite &= (x >= -2) & (x < image.shape[1] + 1) & (y >= -2) & (y < image.shape[0] + 1)
    # Avoid unsafe integer conversion for arbitrarily large invalid coordinates.
    x = np.where(finite, x, -10.0)
    y = np.where(finite, y, -10.0)
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    good_in = np.isfinite(image)
    if valid is not None:
        good_in &= np.broadcast_to(valid, image.shape)
    if variance is not None:
        variance = np.broadcast_to(np.asarray(variance, float), image.shape)
        good_in &= np.isfinite(variance) & (variance >= 0)
    result = np.zeros(shape)
    propagated = np.zeros(shape) if variance is not None else None
    good = finite.copy()

    def kernel(t):
        t = abs(t)
        a = -0.75
        return np.where(
            t < 1,
            (a + 2) * t**3 - (a + 3) * t**2 + 1,
            np.where(t < 2, a * t**3 - 5 * a * t**2 + 8 * a * t - 4 * a, 0.0),
        )

    for j in [-1, 0, 1, 2]:
        sy = iy + j
        wy = kernel(y - sy)
        for i in [-1, 0, 1, 2]:
            sx = ix + i
            weight = wy * kernel(x - sx)
            active = abs(weight) > 1.0e-14
            inside = (sx >= 0) & (sx < image.shape[1]) & (sy >= 0) & (sy < image.shape[0])
            cx = np.clip(sx, 0, image.shape[1] - 1)
            cy = np.clip(sy, 0, image.shape[0] - 1)
            usable = inside & good_in[cy, cx]
            good &= ~active | usable
            result += weight * np.where(usable, image[cy, cx], 0.0)
            if variance is not None:
                propagated += weight**2 * np.where(usable, variance[cy, cx], 0.0)
    if area_scale is not None:
        area_scale = np.broadcast_to(np.asarray(area_scale, float), shape)
        good &= np.isfinite(area_scale) & (area_scale > 0)
        result *= area_scale
        if propagated is not None:
            propagated *= area_scale**2
    result = np.where(good, result, np.nan)
    if propagated is not None:
        propagated = np.where(good, propagated, np.nan)
    return result, propagated, good
