"""Spatial holdouts and block-level statistics with explicit random seeds."""

import numpy as np


def _cells(cells):
    a = np.asarray(cells)
    if (
        a.ndim != 2
        or a.shape[1] != 2
        or not np.isfinite(a).all()
        or np.any(a < 0)
        or np.any(a != np.floor(a))
    ):
        raise ValueError("Nonnegative integer Nx2 cell coordinates required")
    return a.astype(np.int64)


def spatial_folds(cells, block_size=4, folds=4):
    """Keep neighbouring cells in the same spatial holdout block."""
    a = _cells(cells)
    if not isinstance(block_size, int) or block_size < 1 or not isinstance(folds, int) or folds < 2:
        raise ValueError("Positive block size and at least two folds required")
    return (a[:, 0] // block_size + 2 * (a[:, 1] // block_size)) % folds


def block_mean_residuals(residuals, cells, block_size=4):
    """Average signed residual vectors before evaluating their spatial structure."""
    a = _cells(cells)
    r = np.asarray(residuals, float)
    if r.ndim == 1:
        r = r[:, None]
    if (
        r.ndim != 2
        or len(r) != len(a)
        or not np.isfinite(r).all()
        or not isinstance(block_size, int)
        or block_size < 1
    ):
        raise ValueError("Matching finite residual rows and positive block size required")
    blocks, ids = np.unique(a // block_size, axis=0, return_inverse=True)
    means = np.zeros((len(blocks), r.shape[1]))
    np.add.at(means, ids, r)
    if len(blocks):
        means /= np.bincount(ids)[:, None]
    return blocks, means


def bootstrap_mean(values, block_ids, *, draws=2000, seed=0):
    """Resample whole blocks; retain all rows and their unequal block sizes.

    Scores must come from independent validation predictions. Bootstrap
    uncertainty on one sequence is not an independent new-camera test.
    """
    values, block_ids = np.asarray(values, float), np.asarray(block_ids)
    if (
        values.ndim != 1
        or block_ids.shape != values.shape
        or len(values) == 0
        or not np.isfinite(values).all()
        or not isinstance(draws, int)
        or draws < 2
    ):
        raise ValueError("Nonempty finite scores, matching blocks and >=2 draws required")
    _, ids = np.unique(block_ids, return_inverse=True)
    n = ids.max() + 1
    if n < 2:
        raise ValueError("At least two independent resampling blocks required")
    sums = np.bincount(ids, weights=values)
    counts = np.bincount(ids)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for i in range(draws):
        selected = rng.integers(0, n, size=n)
        estimates[i] = sums[selected].sum() / counts[selected].sum()
    return dict(
        mean=float(values.mean()),
        standard_error=float(estimates.std(ddof=1)),
        interval_95=np.percentile(estimates, [2.5, 97.5]).tolist(),
        blocks=int(n),
        draws=draws,
        seed=seed,
    )


def one_standard_error_choice(scores, standard_errors, complexities):
    """Choose lowest complexity within one SE of the minimum validation score.

    This utility does not replace per-fold, object-protection or visual gates.
    """
    scores, errors, complexity = [
        np.asarray(a, float) for a in (scores, standard_errors, complexities)
    ]
    if (
        scores.ndim != 1
        or len(scores) == 0
        or errors.shape != scores.shape
        or complexity.shape != scores.shape
        or not np.isfinite(scores).all()
        or not np.isfinite(errors).all()
        or not np.isfinite(complexity).all()
        or np.any(errors < 0)
        or np.any(complexity < 0)
    ):
        raise ValueError("Matching finite scores, nonnegative errors and complexities required")
    best = int(np.argmin(scores))
    eligible = np.flatnonzero(scores <= scores[best] + errors[best])
    return int(eligible[np.argmin(complexity[eligible])])
