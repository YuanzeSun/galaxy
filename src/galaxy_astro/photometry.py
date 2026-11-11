"""Forced aperture measurements; returned errors are diagonal-noise proxies."""

import numpy as np


def aperture_measure(image, xy, radii=(12, 18, 24, 32), annulus=(42, 54)):
    """Forced photometry with a clipped local sky plane; full pixel units.

    Error is a diagonal-noise approximation, not a covariance-aware error bar.
    Fixed coordinates avoid centroid selection bias in low-SNR measurements.
    """
    radius = int(np.ceil(annulus[1])) + 2
    xy = np.asarray(xy)
    out = np.full((len(xy), len(radii)), np.nan)
    error = out.copy()
    sigma_out = np.full(len(xy), np.nan)
    h, w = image.shape
    for k, (cx, cy) in enumerate(xy):
        ix, iy = int(round(cx)), int(round(cy))
        if ix < radius or iy < radius or ix >= w - radius or iy >= h - radius:
            continue
        patch = image[iy - radius : iy + radius + 1, ix - radius : ix + radius + 1].astype(float)
        yy, xx = np.mgrid[iy - radius : iy + radius + 1, ix - radius : ix + radius + 1]
        xx, yy = xx - cx, yy - cy
        rr = np.hypot(xx, yy)
        ann = (rr > annulus[0]) & (rr < annulus[1]) & np.isfinite(patch)
        mat = np.stack([np.ones_like(xx), xx, yy], axis=-1)
        fitmask = ann.copy()
        if fitmask.sum() < 150:
            continue
        for _ in range(3):
            coef = np.linalg.lstsq(mat[fitmask], patch[fitmask], rcond=None)[0]
            residual = patch - mat @ coef
            median = np.median(residual[fitmask])
            sigma = max(1.0e-6, 1.4826 * np.median(abs(residual[fitmask] - median)))
            fitmask = ann & (abs(residual - median) < 3 * sigma)
            if fitmask.sum() < 150:
                break
        if fitmask.sum() < 150:
            continue
        sigma_out[k] = sigma
        for j, rad in enumerate(radii):
            weight = np.clip(rad + 0.5 - rr, 0, 1)
            if np.any((weight > 0) & ~np.isfinite(patch)):
                continue
            out[k, j] = np.sum(np.where(weight > 0, residual * weight, 0))
            error[k, j] = sigma * np.sqrt((weight**2).sum() + weight.sum() ** 2 / fitmask.sum())
    return out, error, sigma_out
