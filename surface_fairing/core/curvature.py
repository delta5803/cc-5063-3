"""Analytic curvature and reflection-stripe (isophote) diagnostics.

All quantities come from the analytic surface first/second derivatives. No
vertex normals are averaged and no normals are smoothed: a rippled surface
keeps a rippled curvature field and rippled isophotes, so the diagnostics
cannot visually hide geometry.
"""
from __future__ import annotations

import numpy as np

from .bspline import BSplineSurface


def curvature_at(d: dict[str, np.ndarray]) -> dict[str, float]:
    """Gaussian, mean, principal curvatures from fundamental forms."""
    Su, Sv, Suu, Suv, Svv = d["Su"], d["Sv"], d["Suu"], d["Suv"], d["Svv"]
    n = d["n"]
    E = float(Su @ Su)
    F = float(Su @ Sv)
    G = float(Sv @ Sv)
    e = float(Suu @ n)
    f = float(Suv @ n)
    g = float(Svv @ n)
    denom = E * G - F * F
    if denom <= 1e-14:
        return {"gaussian": 0.0, "mean": 0.0, "k1": 0.0, "k2": 0.0}
    K = (e * g - f * f) / denom
    H = 0.5 * (e * G - 2 * f * F + g * E) / denom
    disc = max(H * H - K, 0.0)
    root = np.sqrt(disc)
    return {"gaussian": K, "mean": H,
            "k1": H + root, "k2": H - root}


def curvature_field(surf: BSplineSurface, nu: int = 33, nv: int = 33):
    us = np.linspace(0, 1, nu)
    vs = np.linspace(0, 1, nv)
    K = np.zeros((nu, nv))
    Hm = np.zeros((nu, nv))
    k1 = np.zeros((nu, nv))
    k2 = np.zeros((nu, nv))
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            c = curvature_at(surf.partials(u, v))
            K[i, j] = c["gaussian"]
            Hm[i, j] = c["mean"]
            k1[i, j] = c["k1"]
            k2[i, j] = c["k2"]
    return us, vs, {"gaussian": K, "mean": Hm, "k1": k1, "k2": k2}


def curvature_change(before: BSplineSurface, after: BSplineSurface,
                     nu: int = 33, nv: int = 33) -> dict[str, np.ndarray]:
    """Signed and absolute difference fields of mean and Gaussian curvature."""
    _, _, cb = curvature_field(before, nu, nv)
    _, _, ca = curvature_field(after, nu, nv)
    return {
        "mean_delta": ca["mean"] - cb["mean"],
        "gaussian_delta": ca["gaussian"] - cb["gaussian"],
        "mean_abs_delta": np.abs(ca["mean"] - cb["mean"]),
        "gaussian_abs_delta": np.abs(ca["gaussian"] - cb["gaussian"]),
        "before_mean": cb["mean"], "after_mean": ca["mean"],
        "before_gaussian": cb["gaussian"], "after_gaussian": ca["gaussian"],
    }


def isophote_field(surf: BSplineSurface, light_direction: np.ndarray,
                   nu: int = 40, nv: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sampling grid and cos(angle) between analytic normal and light.

    Reflection stripes are the level sets of this scalar field; drawing them
    before and after fairing shows whether the fairing actually regularized
    the surface rather than merely hiding ripples in a shaded render.
    """
    L = np.asarray(light_direction, dtype=float)
    L = L / np.linalg.norm(L)
    us = np.linspace(0, 1, nu)
    vs = np.linspace(0, 1, nv)
    val = np.zeros((nu, nv))
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            n = surf.normal(u, v)
            val[i, j] = float(np.dot(n, L))
    return us, vs, val
