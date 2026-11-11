"""Read this top-to-bottom as separate manual steps; inputs are synthetic only.

Run: python examples/connect_steps.py
There is no real-photo directory traversal, network call or file output.
For real FITS/WCS/catalog handoffs, follow docs/manual-run.md.
"""

import json

import numpy as np
from astropy.wcs import WCS

if __package__:
    from .manual_steps import (
        join_rows,
        measure_rows,
        pixel_solid_angles,
        reference_to_source,
        register_surface_brightness,
    )
else:
    from manual_steps import (
        join_rows,
        measure_rows,
        pixel_solid_angles,
        reference_to_source,
        register_surface_brightness,
    )

from galaxy_astro.background import fit_background_plane, sample_background_cells
from galaxy_astro.calibration import calibrate_linear
from galaxy_astro.selfcal import fit_stellar_response, radial_features
from galaxy_astro.stacking import combine_group_means
from galaxy_astro.validation import spatial_folds


def worked_example():
    # 1. Toy calibrated linear counts + solved WCS. On real data these are
    # supplied by your RAW/calibration and astrometry tools, not by this script.
    shape = (160, 200)
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    sensor_xy = np.stack([xx, yy], axis=-1)
    center = ((w - 1) / 2, (h - 1) / 2)
    scale = w / 2
    reference = WCS(naxis=2)
    reference.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    reference.wcs.crval = [120.0, 20.0]
    reference.wcs.crpix = [center[0] + 1, center[1] + 1]  # FITS CRPIX is one-based.
    reference.wcs.cdelt = [-0.015, 0.015]
    omega0 = pixel_solid_angles(reference, shape)[h // 2, w // 2]
    star_xy = np.array([(x, y) for y in [40, 65, 90, 115] for x in [40, 70, 100, 130, 160]], float)
    star_ids = np.array([f"star-{i:02}" for i in range(len(star_xy))])
    sky_stars = reference.pixel_to_world(star_xy[:, 0], star_xy[:, 1])
    theta_truth = np.array([-0.22, -0.06])
    response_truth = np.exp(radial_features(sensor_xy, center, scale) @ theta_truth)
    shifts = [(-4, -4), (0, 4), (4, 0), (4, 4), (-4, 0), (0, -4), (-3, 3), (3, -3)]
    images, worlds, tables = [], [], []
    for frame, (dx, dy) in enumerate(shifts):
        wcs = reference.deepcopy()
        wcs.wcs.crpix += [dx, dy]
        native_xy = np.column_stack(wcs.world_to_pixel(sky_stars))
        area = pixel_solid_angles(wcs, shape) / omega0
        sky = (40.0 + 0.02 * (xx - dx) + 0.03 * (yy - dy)) * area
        green = sky.copy()
        for star, (sx, sy) in enumerate(native_xy):
            green += (700.0 + 20 * star) * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * 1.2**2))
        transmission = np.exp(0.015 * np.sin(frame))  # Reference frame offset is zero.
        green *= response_truth * transmission
        rgb = green[..., None] * np.array([0.8, 1.0, 1.2])
        rows, keep = measure_rows(
            rgb,
            native_xy,
            star_ids,
            str(frame),
            np.ones(len(star_ids), bool),
            radius=5,
            annulus=(8, 12),
            minimum_snr=20,
            fractional_error_floor=0.01,
        )
        if not keep.all():
            raise AssertionError("Unexpected synthetic measurement loss")
        images.append(rgb)
        worlds.append(wcs)
        tables.append(rows)

    # 2. Fit from training stars only; keep the other stars for a later check.
    table = join_rows(tables)
    features = radial_features(np.column_stack([table["x"], table["y"]]), center, scale)
    held_star_ids = star_ids[::4]
    train = ~np.isin(table["star_id"], held_star_ids)
    fit = fit_stellar_response(
        table["log_flux"][train],
        features[train],
        table["star_id"][train],
        table["frame_id"][train],
        table["sigma_log"][train],
    )
    offsets = dict(zip(fit.frame_levels, fit.frame_log_offsets))

    # 3. Turn native counts into relative surface brightness, then resample.
    # Dark was already subtracted at entry; dark=0 avoids subtracting it twice.
    response = np.exp(radial_features(sensor_xy, center, scale) @ fit.coefficients)
    aligned, masks = [], []
    for frame, (rgb, wcs) in enumerate(zip(images, worlds)):
        area = pixel_solid_angles(wcs, shape) / omega0
        denominator = response * area * np.exp(offsets[str(frame)])
        sb, _, valid_rgb = calibrate_linear(rgb, dark=0.0, response=denominator[..., None])
        source_xy = reference_to_source(reference, wcs, shape)
        warped, valid = register_surface_brightness(sb, source_xy, valid_rgb.all(axis=-1))
        aligned.append(warped)
        masks.append(valid)

    # 4. Each example group has one exposure. Real groups may accumulate many;
    # counts must then contain that actual number, not a Boolean group count.
    combined = combine_group_means(np.stack(aligned), np.asarray(masks, int))
    linear_master = combined["mean"]

    # 5. Optional background candidate, separate from the linear master.
    protected = np.zeros(shape, bool)
    for sx, sy in star_xy:
        protected |= (xx - sx) ** 2 + (yy - sy) ** 2 < 7.0**2
    protected |= ~np.isfinite(linear_master).all(axis=-1)
    xy, values, cells = sample_background_cells(linear_master, protected, cell_size=8)
    folds = spatial_folds(cells, block_size=2)
    errors = []
    for fold in np.unique(folds):
        plane = fit_background_plane(xy[folds != fold], values[folds != fold], shape)
        residual = plane.predict(xy[folds == fold]) - values[folds == fold]
        errors.extend(np.linalg.norm(residual, axis=1).tolist())
    # No automatic adoption/subtraction: this example only reports held errors.
    coeff_error = float(abs(fit.coefficients - theta_truth).max())
    correction = features @ fit.coefficients + np.array([offsets[f] for f in table["frame_id"]])
    held_scatter = []
    for star in held_star_ids:
        corrected = (table["log_flux"] - correction)[table["star_id"] == star]
        held_scatter.append(float(np.std(corrected)))
    return dict(
        synthetic_only=True,
        frame_count=len(images),
        measured_rows=len(table["x"]),
        training_stars=int(len(star_ids) - len(held_star_ids)),
        held_stars=len(held_star_ids),
        response_coefficients=fit.coefficients.tolist(),
        response_known_truth=theta_truth.tolist(),
        maximum_coefficient_error=coeff_error,
        held_star_log_flux_scatter_max=max(held_scatter),
        output_shape=list(linear_master.shape),
        actual_coverage_range=[int(combined["coverage"].min()), int(combined["coverage"].max())],
        held_background_error_median=float(np.median(errors)),
        background_candidate_applied=False,
        passed=bool(
            coeff_error < 0.02 and np.isfinite(linear_master[combined["coverage"] > 0]).all()
        ),
    )


if __name__ == "__main__":
    report = worked_example()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
