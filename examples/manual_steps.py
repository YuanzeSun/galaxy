"""Small adapters for docs/manual-run.md; callers choose each step explicitly.

These helpers read linear FITS, connect solved WCS coordinates to measurements,
and call the numerical library. They do not scan a photo directory, decode RAW,
query catalogs, plate-solve, choose masks, or decide which model to adopt.
Install the optional dependencies with: python -m pip install '.[manual]'
"""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from galaxy_astro.photometry import aperture_measure
from galaxy_astro.resampling import resample_cubic


def read_linear_fits(path):
    """Read a solved, linear image without modifying its file.

    Accept HxW or FITS channel-first (3,H,W). Return HxW or (H,W,3),
    respectively. A matching, two-axis celestial WCS is required.
    Linear calibration, units, white balance and channel order are supplied
    by the user's decoder; no automatic brightness/normalization is applied.
    """
    with fits.open(Path(path), mode="readonly", memmap=False) as hdus:
        header = hdus[0].header.copy()
        image = np.asarray(hdus[0].data, dtype=float).copy()
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    elif image.ndim != 2:
        raise ValueError("Expected a 2-D plane or a channel-first RGB FITS image")
    wcs = WCS(header, naxis=2).celestial
    if not wcs.has_celestial:
        raise ValueError("A solved celestial WCS is required")
    return image, wcs


def reference_to_source(reference_wcs, source_wcs, shape, *, y_start=0, y_stop=None):
    """Return (tile_rows,W,2) source xy for each reference output pixel.

    Coordinates are zero-based pixel centers; arrays remain indexed [y,x].
    High-level SkyCoord transformations handle celestial frame differences.
    """
    h, w = shape
    stop = h if y_stop is None else y_stop
    if not 0 <= y_start < stop <= h:
        raise ValueError("Invalid reference row range")
    yy, xx = np.mgrid[y_start:stop, :w]
    sky = reference_wcs.pixel_to_world(xx, yy)
    sx, sy = source_wcs.world_to_pixel(sky)
    return np.stack([sx, sy], axis=-1)


def pixel_solid_angles(wcs, shape, *, y_start=0, y_stop=None):
    """Approximate steradians/pixel from half-pixel direction differences.

    This includes the supplied WCS distortion, unlike a single center pixel
    scale. It is a local derivative approximation, not exact pixel integration.
    Use the same reference-center normalization for ALL frames. Tile real
    full-resolution images to avoid allocating four entire vector grids.
    """
    h, w = shape
    stop = h if y_stop is None else y_stop
    if not 0 <= y_start < stop <= h:
        raise ValueError("Invalid sensor row range")
    yy, xx = np.mgrid[y_start:stop, :w]

    def directions(x, y):
        sky = wcs.pixel_to_world(x, y)
        return np.moveaxis(sky.cartesian.xyz.value, 0, -1)

    dx = directions(xx + 0.5, yy) - directions(xx - 0.5, yy)
    dy = directions(xx, yy + 0.5) - directions(xx, yy - 0.5)
    omega = np.linalg.norm(np.cross(dx, dy), axis=-1)
    if not np.isfinite(omega).all() or np.any(omega <= 0):
        raise ValueError("WCS has invalid angular pixel areas")
    return omega


def measure_rows(
    image, xy, star_ids, frame_id, admitted, *, radius, annulus, minimum_snr, fractional_error_floor
):
    """Turn fixed-position aperture results into a selfcal input table.

    `admitted` is the user's independent quality/saturation/blend mask.
    Returns (accepted rows, original-row selection mask). The aperture error
    is a background diagonal-noise proxy, not full source/calibration noise.
    Set errors from a better noise model before fitting real data if available.
    No sample is selected by its fitted response residual.
    """
    image = np.asarray(image, float)
    if image.ndim == 3 and image.shape[-1] == 3:
        image = image[..., 1]
    elif image.ndim != 2:
        raise ValueError("Expected a green plane or HxWx3 RGB image")
    xy = np.asarray(xy, float)
    ids = np.asarray(star_ids, dtype=str)
    admitted = np.asarray(admitted, bool)
    if (
        xy.ndim != 2
        or xy.shape[1] != 2
        or ids.shape != (len(xy),)
        or admitted.shape != ids.shape
        or not np.isfinite(minimum_snr)
        or minimum_snr <= 0
        or not np.isfinite(fractional_error_floor)
        or fractional_error_floor <= 0
    ):
        raise ValueError("Matching star IDs/quality mask and positive SNR/error floor required")
    finite = np.isfinite(xy).all(axis=1)
    flux, error = np.full(len(xy), np.nan), np.full(len(xy), np.nan)
    if finite.any():
        measured, uncertainty, _ = aperture_measure(
            image, xy[finite], radii=(radius,), annulus=annulus
        )
        flux[finite], error[finite] = measured[:, 0], uncertainty[:, 0]
    keep = (
        admitted
        & np.isfinite(flux)
        & np.isfinite(error)
        & (error > 0)
        & (flux > minimum_snr * error)
    )
    rows = dict(
        x=xy[keep, 0],
        y=xy[keep, 1],
        star_id=ids[keep],
        frame_id=np.full(keep.sum(), str(frame_id)),
        flux=flux[keep],
        log_flux=np.log(flux[keep]),
        sigma_log=np.sqrt((error[keep] / flux[keep]) ** 2 + fractional_error_floor**2),
    )
    return rows, keep


def join_rows(tables):
    """Join already measured exposure tables; ID strings remain exact."""
    if not tables:
        raise ValueError("At least one table required")
    return {key: np.concatenate([table[key] for table in tables]) for key in tables[0]}


def register_surface_brightness(image, source_xy, valid=None):
    """Resample already calibrated surface brightness, channel by channel.

    Do not supply an additional Jacobian area factor after converting to
    surface brightness. One shared validity mask excludes any invalid channel.
    """
    image = np.asarray(image, float)
    mono = image.ndim == 2
    if mono:
        image = image[..., None]
    output, masks = [], []
    for channel in range(image.shape[-1]):
        value, _, good = resample_cubic(image[..., channel], source_xy, valid=valid)
        output.append(value)
        masks.append(good)
    shared = np.logical_and.reduce(masks)
    result = np.stack(output, axis=-1)
    result[~shared] = np.nan
    return (result[..., 0] if mono else result), shared
