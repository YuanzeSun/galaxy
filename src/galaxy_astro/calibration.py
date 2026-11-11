"""Dark subtraction, multiplicative response, and an explicit integer bridge."""

import numpy as np


def calibrate_linear(raw, dark, response=1.0, variance=None, valid=None):
    """Return (signal, diagonal_variance, valid_mask), retaining negatives.

    Dark includes the bias pedestal. Response is dimensionless and positive.
    A supplied variance must already include raw AND dark uncertainty. Shared
    calibration errors and response uncertainty are not modeled here.
    """
    raw, dark, response = np.broadcast_arrays(
        np.asarray(raw, float), np.asarray(dark, float), np.asarray(response, float)
    )
    good = np.isfinite(raw) & np.isfinite(dark) & np.isfinite(response) & (response > 0)
    if valid is not None:
        good &= np.broadcast_to(np.asarray(valid, bool), raw.shape)
    propagated = None
    if variance is not None:
        variance = np.broadcast_to(np.asarray(variance, float), raw.shape)
        good &= np.isfinite(variance) & (variance >= 0)
        propagated = np.full(raw.shape, np.nan)
        np.divide(variance, response**2, out=propagated, where=good)
    result = np.full(raw.shape, np.nan)
    np.divide(raw - dark, response, out=result, where=good)
    return result, propagated, good


def quantize_unsigned_cfa(values, white_level):
    """Clip and round to nearest/even for an unavoidable unsigned CFA interface.

    Prefer floating point when supported. This does not undo ADC saturation,
    subsequent demosaic clipping, or quantization inside a decoder.
    """
    values = np.asarray(values, dtype=float)
    if (
        not np.isfinite(values).all()
        or not np.isscalar(white_level)
        or not np.isfinite(white_level)
        or not 0 < white_level <= 65535
        or float(white_level) != int(white_level)
    ):
        raise ValueError("Finite values and an integer white level in [1, 65535] required")
    return np.rint(np.clip(values, 0, white_level)).astype(np.uint16)


def calibrated_cfa_bridge(observed, dark_with_black, black_level, white_level):
    """Dark-subtract, restore the declared black pedestal, then quantize.

    A scalar or per-pixel black map can be supplied. The raw adapter is
    responsible for constructing that map from its camera's CFA metadata.
    No particular sensor dimensions, black pedestal or Bayer pattern is assumed.
    """
    observed = np.asarray(observed, float)
    dark = np.asarray(dark_with_black, float)
    if observed.shape != dark.shape:
        raise ValueError("Observed and dark CFA shapes differ")
    black = np.broadcast_to(np.asarray(black_level, float), observed.shape)
    return quantize_unsigned_cfa(observed - dark + black, white_level)
