"""Combination with explicit valid coverage, retaining zero and negative signals."""

import numpy as np


def weighted_mean(values, variances, valid=None):
    """Inverse-variance mean along axis 0, excluding invalid samples.

    No clipping against the observed signal: callers supply independently
    estimated variances and an explicit rejection mask. Empty pixels stay NaN.
    n_eff=(sum w)^2/sum(w^2) describes weight concentration, not exposure time.
    """
    v, var = np.broadcast_arrays(np.asarray(values, float), np.asarray(variances, float))
    good = np.isfinite(v) & np.isfinite(var) & (var > 0)
    if valid is not None:
        good &= np.broadcast_to(valid, v.shape)
    weights = np.zeros(v.shape)
    np.divide(1.0, var, out=weights, where=good)
    sw, sw2 = weights.sum(axis=0), (weights**2).sum(axis=0)
    mean = np.full(sw.shape, np.nan)
    np.divide((weights * np.where(good, v, 0)).sum(axis=0), sw, out=mean, where=sw > 0)
    variance = np.full(sw.shape, np.nan)
    np.divide(1.0, sw, out=variance, where=sw > 0)
    n_eff = np.zeros(sw.shape)
    np.divide(sw**2, sw2, out=n_eff, where=sw2 > 0)
    return dict(mean=mean, variance=variance, sum_weights=sw, n_eff=n_eff, count=good.sum(axis=0))


def combine_group_means(values, counts, *, clip_sigma=None, scale_floor=None):
    """Combine RGB/channel group means with their actual exposure counts.

    Values have shape (groups, ..., channels); counts have shape (groups, ...).
    With clip_sigma=None this is a count-weighted mean. Explicitly requesting
    clipping reproduces the historical joint-channel MAD rule, with the
    caller's positive scale_floor in standardized signal units. It has known
    small-group inefficiency and is not a universal optimal estimator.

    Empty coverage remains NaN. If clipping rejects every usable group, keep
    one actual group vector closest to the robust center and flag the fallback.
    No synthetic RGB vector or fabricated exposure count is substituted.
    """
    import warnings

    v, n = np.asarray(values, float), np.asarray(counts, float)
    if (
        v.ndim < 2
        or v.shape[0] == 0
        or v.shape[-1] == 0
        or n.shape != v.shape[:-1]
        or not np.isfinite(n).all()
        or np.any(n < 0)
        or np.any(n != np.floor(n))
    ):
        raise ValueError("Group channel means and matching nonnegative integer counts required")
    if (clip_sigma is None) != (scale_floor is None):
        raise ValueError("Supply both clipping threshold and positive scale floor, or neither")
    if clip_sigma is not None and (
        not np.isfinite(clip_sigma)
        or clip_sigma <= 0
        or not np.isfinite(scale_floor)
        or scale_floor <= 0
    ):
        raise ValueError("Positive finite clipping threshold and scale floor required")
    valid = (n > 0) & np.isfinite(v).all(axis=-1)
    good = valid.copy()
    fallback = np.zeros(v.shape[1:-1], bool)
    if clip_sigma is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # Empty coverage is handled below.
            center = np.nanmedian(np.where(valid[..., None], v, np.nan), axis=0)
            z = abs(v - center) * np.sqrt(n)[..., None]
            mad = 1.4826 * np.nanmedian(np.where(valid[..., None], z, np.nan), axis=0)
        limit = np.maximum(clip_sigma * mad, scale_floor)
        good &= np.all(z <= limit, axis=-1)
        fallback = valid.any(axis=0) & ~good.any(axis=0)
        distance = np.max(z / limit, axis=-1)
        nearest = np.argmin(np.where(valid, distance, np.inf), axis=0)
        for group in range(len(v)):
            good[group] |= fallback & (nearest == group) & valid[group]
    weights = np.where(good, n, 0.0)
    total = weights.sum(axis=0)
    mean = np.full(v.shape[1:], np.nan)
    numerator = (np.where(good[..., None], v, 0.0) * weights[..., None]).sum(axis=0)
    np.divide(numerator, total[..., None], out=mean, where=total[..., None] > 0)
    return dict(
        mean=mean,
        coverage=np.where(valid, n, 0.0).sum(axis=0),
        accepted_count=total,
        rejected_groups=(valid & ~good).sum(axis=0),
        fallback=fallback,
    )
