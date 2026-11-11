"""Relative stellar illumination response with star and exposure nuisance terms.

This is a log-flux model, not a recovery of a complete physical flat field.
Callers must select suitable unsaturated stars and supply appropriate errors.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr


def radial_features(xy, center, scale, powers=(1, 2)):
    """Columns are powers of r^2 with caller-supplied center and scale."""
    xy = np.asarray(xy, float)
    center = np.asarray(center, float)
    powers = tuple(powers)
    if (
        xy.ndim < 1
        or xy.shape[-1] != 2
        or center.shape != (2,)
        or not np.isfinite(xy).all()
        or not np.isfinite(center).all()
        or not np.isscalar(scale)
        or not np.isfinite(scale)
        or scale <= 0
        or not powers
        or any(int(p) != p or p < 1 for p in powers)
    ):
        raise ValueError("Finite xy/center, positive scale and positive integer powers required")
    r2 = np.sum(((xy - center) / scale) ** 2, axis=-1)
    return np.stack([r2**p for p in powers], axis=-1)


def _design(features, star_ids, frame_ids, sigma):
    x = np.asarray(features, float)
    sigma = np.asarray(sigma, float)
    stars_in, frames_in = np.asarray(star_ids), np.asarray(frame_ids)
    if (
        x.ndim != 2
        or min(x.shape) == 0
        or sigma.shape != (len(x),)
        or stars_in.shape != (len(x),)
        or frames_in.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(sigma).all()
        or np.any(sigma <= 0)
    ):
        raise ValueError("Nonempty finite feature rows, matching IDs and positive errors required")
    star_levels, stars = np.unique(stars_in, return_inverse=True)
    frame_levels, frames = np.unique(frames_in, return_inverse=True)
    n, ns, nf = len(x), len(star_levels), len(frame_levels)
    obs = np.arange(n)
    nonzero = frames > 0
    nuisance = coo_matrix(
        (
            np.ones(n + nonzero.sum()),
            (np.r_[obs, obs[nonzero]], np.r_[stars, ns + frames[nonzero] - 1]),
        ),
        shape=(n, ns + nf - 1),
    ).tocsr()
    graph = coo_matrix(
        (np.ones(2 * n), (np.r_[stars, ns + frames], np.r_[ns + frames, stars])),
        shape=(ns + nf, ns + nf),
    ).tocsr()
    components = int(connected_components(graph, directed=False, return_labels=False))
    return x, sigma, nuisance, star_levels, frame_levels, components


def _solve(matrix, target):
    solved = lsqr(
        matrix,
        target,
        atol=1e-12,
        btol=1e-12,
        conlim=1e12,
        iter_lim=max(1000, 10 * matrix.shape[1]),
    )
    if solved[1] not in (0, 1, 2, 4, 5):
        raise ValueError("Nuisance least-squares solve did not converge reliably")
    return solved[0]


def _project(matrix, values):
    if values.ndim == 1:
        return values - matrix @ _solve(matrix, values)
    return np.column_stack([_project(matrix, values[:, j]) for j in range(values.shape[1])])


def response_information(features, star_ids, frame_ids, sigma):
    """Check response information remaining after star/exposure offsets.

    Numerical rank is necessary, not sufficient: atmosphere, aperture effects,
    PSF errors and model bias can still contaminate the inferred response.
    """
    x, sigma, n, stars, frames, components = _design(features, star_ids, frame_ids, sigma)
    weighted = x / sigma[:, None]
    projected = _project(n.multiply((1 / sigma)[:, None]).tocsr(), weighted)
    sv = np.linalg.svd(projected, compute_uv=False)
    tolerance = max(np.linalg.norm(weighted, 2) * 1e-9, np.finfo(float).eps)
    rank = int(np.sum(sv > tolerance))
    norms = np.linalg.norm(weighted, axis=0)
    return dict(
        observations=len(x),
        stars=len(stars),
        frames=len(frames),
        feature_count=x.shape[1],
        graph_components=components,
        numerical_rank=rank,
        singular_values=sv.tolist(),
        condition_number=float(sv[0] / sv[-1]) if rank == x.shape[1] else None,
        fraction_not_absorbed_by_nuisance=np.divide(
            np.linalg.norm(projected, axis=0), norms, out=np.zeros_like(norms), where=norms > 0
        ).tolist(),
        necessary_identifiability_check_passed=(rank == x.shape[1] and components == 1),
    )


@dataclass(frozen=True)
class ResponseFit:
    coefficients: np.ndarray
    star_levels: np.ndarray
    frame_levels: np.ndarray
    star_log_flux: np.ndarray
    frame_log_offsets: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray
    information: dict


def fit_stellar_response(log_flux, features, star_ids, frame_ids, sigma, *, iterations=6, clip=2.7):
    """Fit log(flux)=features@theta+star_offset+exposure_offset by robust LS.

    The first sorted exposure offset is zero. Features that nuisance offsets
    can absorb, or disconnected star/exposure graphs, are explicitly refused.
    Six IRLS iterations are a bounded robust fit, not a convergence guarantee.
    Separate withheld stars, times or regions are still required for adoption.
    """
    if not isinstance(iterations, int) or iterations < 1 or not np.isfinite(clip) or clip <= 0:
        raise ValueError("Positive iterations and clipping threshold required")
    x, sigma, nuisance, stars, frames, _ = _design(features, star_ids, frame_ids, sigma)
    y = np.asarray(log_flux, float)
    if y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("One finite log-flux per feature row required")
    info = response_information(x, star_ids, frame_ids, sigma)
    if not info["necessary_identifiability_check_passed"]:
        raise ValueError("Response is not identifiable apart from star/exposure offsets")
    robust = np.ones(len(x))
    for _ in range(iterations):
        sw = np.sqrt(robust) / sigma
        n = nuisance.multiply(sw[:, None]).tocsr()
        px = _project(n, x * sw[:, None])
        py = _project(n, y * sw)
        theta, _, rank, _ = np.linalg.lstsq(px, py, rcond=1e-9)
        if rank != x.shape[1]:
            raise ValueError("Weighted response model lost rank")
        offsets = _solve(n, (y - x @ theta) * sw)
        prediction = x @ theta + nuisance @ offsets
        residual = y - prediction
        z = residual / sigma
        center = np.median(z)
        scale = max(1.0, 1.4826 * np.median(abs(z - center)))
        robust = np.minimum(1.0, clip * scale / np.maximum(abs(z - center), 1e-15))
    return ResponseFit(
        theta,
        stars,
        frames,
        offsets[: len(stars)],
        np.r_[0.0, offsets[len(stars) :]],
        prediction,
        residual,
        info,
    )
