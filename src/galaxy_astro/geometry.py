"""Polynomial camera maps and pixel solid angle. No sensor dimensions assumed."""

import math

import numpy as np


def polynomial_map(xy, coefficients, origin, scale, jacobian=False):
    """Reference -> source mapping, triangular total-degree coefficient order.

    degree 2 order: 1, x, y, x*x, x*y, y*y. Coordinates and scale are
    caller-supplied; scale may be a scalar or a two-element vector.
    """
    xy = np.asarray(xy, dtype=np.float64)
    coef = np.asarray(coefficients, dtype=np.float64)
    if xy.ndim < 1 or xy.shape[-1] != 2 or coef.ndim != 2 or len(coef) == 0:
        raise ValueError("Expected (..., 2) coordinates and nonempty (nterms, 2) coefficients.")
    degree = int((math.sqrt(8 * len(coef) + 1) - 3) / 2)
    if coef.shape != ((degree + 1) * (degree + 2) // 2, 2):
        raise ValueError("Coefficients must have triangular shape (nterms, 2).")
    scale = np.broadcast_to(np.asarray(scale, dtype=float), (2,))
    origin = np.asarray(origin, float)
    if origin.shape != (2,) or not np.isfinite(origin).all() or not np.isfinite(coef).all():
        raise ValueError("Finite origin and coefficients required.")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Coordinate scales must be positive.")
    uv = (xy - origin) / scale
    x, y = uv[..., 0], uv[..., 1]
    mapped = np.zeros(xy.shape, dtype=np.float64)
    jac = np.zeros(xy.shape[:-1] + (2, 2), dtype=np.float64) if jacobian else None
    k = 0
    for n in range(degree + 1):
        for j in range(n + 1):
            i = n - j
            mapped += (x**i * y**j)[..., None] * coef[k]
            if jacobian:
                if i:
                    jac[..., :, 0] += (i * x ** (i - 1) * y**j / scale[0])[..., None] * coef[k]
                if j:
                    jac[..., :, 1] += (j * x**i * y ** (j - 1) / scale[1])[..., None] * coef[k]
            k += 1
    return (mapped, jac) if jacobian else mapped


def grid_map(coefficients, x_coordinates, y_coordinates, origin, scale, dtype=np.float32):
    c = np.asarray(coefficients, dtype=dtype)
    if c.ndim != 2 or len(c) == 0 or not np.isfinite(c).all():
        raise ValueError("Nonempty finite polynomial coefficients required")
    degree = int((math.sqrt(8 * len(c) + 1) - 3) / 2)
    if len(c) != (degree + 1) * (degree + 2) // 2 or c.shape[1] != 2:
        raise ValueError("Expected triangular polynomial coefficients with 2 outputs")
    scale = np.broadcast_to(np.asarray(scale, dtype=dtype), (2,))
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Finite positive coordinate scales required")
    origin = np.asarray(origin, dtype=dtype)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("Finite two-dimensional origin required")
    if np.ndim(x_coordinates) != 1 or np.ndim(y_coordinates) != 1:
        raise ValueError("One-dimensional x and y coordinates required")
    x = (np.asarray(x_coordinates, dtype=dtype)[None, :] - origin[0]) / scale[0]
    y = (np.asarray(y_coordinates, dtype=dtype)[:, None] - origin[1]) / scale[1]
    index = {(n - j, j): n * (n + 1) // 2 + j for n in range(degree + 1) for j in range(n + 1)}
    result = []
    for channel in range(2):
        z = None
        for j in range(degree, -1, -1):
            q = c[index[degree - j, j], channel]
            for i in range(degree - j - 1, -1, -1):
                q = q * x + c[index[i, j], channel]
            z = q if z is None else z * y + q
        result.append(
            np.ascontiguousarray(np.broadcast_to(z, (y.shape[0], x.shape[1])), dtype=dtype)
        )
    return result


def derivative_coefficients(coefficients, axis, scale):
    c = np.asarray(coefficients, float)
    if (
        axis not in (0, 1)
        or c.ndim != 2
        or len(c) == 0
        or c.shape[1] != 2
        or not np.isfinite(c).all()
    ):
        raise ValueError("Finite two-output polynomial and axis 0 or 1 required")
    degree = int((math.sqrt(8 * len(c) + 1) - 3) / 2)
    if len(c) != (degree + 1) * (degree + 2) // 2:
        raise ValueError("Expected triangular coefficient count")
    scale = np.broadcast_to(np.asarray(scale, float), (2,))
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Finite positive coordinate scales required")
    if degree == 0:
        return np.zeros((1, 2))
    output = np.zeros((degree * (degree + 1) // 2, 2))
    scale = np.broadcast_to(np.asarray(scale, float), (2,))
    index = {(n - j, j): n * (n + 1) // 2 + j for n in range(degree + 1) for j in range(n + 1)}
    outindex = {(n - j, j): n * (n + 1) // 2 + j for n in range(degree) for j in range(n + 1)}
    for (i, j), k in index.items():
        power = (i, j)[axis]
        if power:
            exponent = (i - 1, j) if axis == 0 else (i, j - 1)
            output[outindex[exponent]] = c[k] * power / scale[axis]
    return output


def solid_angle_grid(x_coordinates, y_coordinates, model):
    coefficients = model["inverse_coefficients"]
    origin = model["inverse_origin"]
    scale = model["inverse_scale"]
    u, v = grid_map(coefficients, x_coordinates, y_coordinates, origin, scale)
    ux, vx = grid_map(
        derivative_coefficients(coefficients, 0, scale), x_coordinates, y_coordinates, origin, scale
    )
    uy, vy = grid_map(
        derivative_coefficients(coefficients, 1, scale), x_coordinates, y_coordinates, origin, scale
    )
    return abs(ux * vy - uy * vx) / (1 + u * u + v * v) ** 1.5


def tangent_from_pixels(xy, model, jacobian=False):
    return polynomial_map(
        xy, model["inverse_coefficients"], model["inverse_origin"], model["inverse_scale"], jacobian
    )


def vectors_from_pixels(xy, model):
    tg = tangent_from_pixels(xy, model)
    v = np.concatenate([tg, np.ones(tg.shape[:-1] + (1,))], axis=-1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def project_vectors(vectors, rotation, model):
    rotated = np.asarray(vectors, dtype=float) @ np.asarray(rotation, dtype=float).T
    good = rotated[..., 2] > 0
    tangent = np.divide(
        rotated[..., :2],
        rotated[..., 2:3],
        out=np.full_like(rotated[..., :2], np.nan),
        where=good[..., None],
    )
    return polynomial_map(
        tangent, model["forward_coefficients"], model["forward_origin"], model["forward_scale"]
    )


def map_pixels(xy, rotation, model, jacobian=False):
    if jacobian:
        tg, ji = tangent_from_pixels(xy, model, True)
    else:
        tg = tangent_from_pixels(xy, model)
    unnormalized = np.concatenate([tg, np.ones(tg.shape[:-1] + (1,))], axis=-1)
    r = np.asarray(rotation)
    v = unnormalized @ r.T
    depth = v[..., 2]
    good = depth > 0
    target_tg = np.divide(
        v[..., :2], depth[..., None], out=np.full_like(tg, np.nan), where=good[..., None]
    )
    if not jacobian:
        return polynomial_map(
            target_tg,
            model["forward_coefficients"],
            model["forward_origin"],
            model["forward_scale"],
        )
    mapped, jf = polynomial_map(
        target_tg,
        model["forward_coefficients"],
        model["forward_origin"],
        model["forward_scale"],
        True,
    )
    # Quotient-rule derivative of rotated homogeneous tangent coordinates.
    jrot = (r[:2, :2] * depth[..., None, None] - v[..., :2, None] * r[2, :2]) / depth[
        ..., None, None
    ] ** 2
    jac = jf @ jrot @ ji
    return mapped, jac


def solid_angle(xy, model):
    tg, jac = tangent_from_pixels(xy, model, True)
    return abs(np.linalg.det(jac)) / (1 + np.sum(tg * tg, axis=-1)) ** 1.5
