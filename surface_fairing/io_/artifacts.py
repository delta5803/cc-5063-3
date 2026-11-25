"""Test artifacts with known, quantified perturbations.

Each artifact places control points on a Greville grid so the unperturbed
surface is an exact plane / smooth curved surface. A deterministic sinusoidal
ripple is then added to interior control points with a documented amplitude,
giving a known shape displacement and a curvature field that genuinely
oscillates. Separate builders construct the required failure and
over-fairing cases.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.bspline import BSplineSurface, Metric, greville_abscissae


@dataclass
class Artifact:
    name: str
    surface: BSplineSurface
    kind: str
    ripple_amplitude: float
    description: str
    expected: dict


def _greville_grid(nu: int, nv: int, degree: int = 3):
    gu = greville_abscissae(nu, degree)
    gv = greville_abscissae(nv, degree)
    return np.meshgrid(gu, gv, indexing="ij")


def rippled_plane(nu: int = 9, nv: int = 9, amplitude: float = 0.04,
                  waves: int = 3, metric: Metric | None = None) -> Artifact:
    G = _greville_grid(nu, nv)
    base_z = np.zeros((nu, nv))
    ripple = amplitude * np.sin(2 * np.pi * waves * G[0]) * np.sin(
        2 * np.pi * waves * G[1])
    # keep boundary rows/columns untouched so the ripple is interior
    ripple[0, :] = ripple[-1, :] = ripple[:, 0] = ripple[:, -1] = 0.0
    P = np.stack([G[0], G[1], base_z + ripple], axis=-1)
    surf = BSplineSurface(P, metric=metric, name="rippled_plane")
    return Artifact(
        name="rippled_plane", surface=surf, kind="plane",
        ripple_amplitude=amplitude,
        description="exact plane plus interior sinusoidal z-ripple",
        expected={"boundary_flat": True, "interior_ripple": amplitude})


def rippled_curved(nu: int = 16, nv: int = 16, amplitude: float = 0.03,
                   waves: int = 6, curvature: float = 0.25,
                   metric: Metric | None = None) -> Artifact:
    G = _greville_grid(nu, nv)
    base_z = curvature * np.sin(np.pi * G[0]) * np.sin(np.pi * G[1])
    ripple = amplitude * np.sin(2 * np.pi * waves * G[0]) * np.cos(
        2 * np.pi * waves * G[1])
    ripple[0, :] = ripple[-1, :] = ripple[:, 0] = ripple[:, -1] = 0.0
    P = np.stack([G[0], G[1], base_z + ripple], axis=-1)
    surf = BSplineSurface(P, metric=metric, name="rippled_curved")
    return Artifact(
        name="rippled_curved", surface=surf, kind="curved",
        ripple_amplitude=amplitude,
        description=(
            "smooth dome plus interior sinusoidal ripple; the control grid "
            f"({nu}x{nv}) resolves the {waves}-wave ripple so curvature "
            "variation energy can separate it from the low-frequency dome"),
        expected={"base_curvature": curvature, "interior_ripple": amplitude,
                  "base_dome_z_at_center": float(
                      curvature * np.sin(np.pi * 0.5) ** 2)})


def over_fairing_case(nu: int = 9, nv: int = 9,
                      metric: Metric | None = None) -> Artifact:
    """A surface carrying a feature the user wants preserved.

    Aggressive fairing without a key-point lock flattens the bump; a correct
    run with the lock keeps it. The artifact ships the key location so the
    test can quantify feature retention.
    """
    G = _greville_grid(nu, nv)
    z = 0.12 * np.exp(-((G[0] - 0.5) ** 2 + (G[1] - 0.5) ** 2) / 0.02)
    P = np.stack([G[0], G[1], z], axis=-1)
    surf = BSplineSurface(P, metric=metric, name="feature_bump")
    return Artifact(
        name="feature_bump", surface=surf, kind="feature",
        ripple_amplitude=0.12,
        description="localized bump that must survive when key-point locked",
        expected={"key_u": 0.5, "key_v": 0.5,
                  "key_height": float(surf.evaluate(0.5, 0.5)[2])})


def conflicting_constraint_case(nu: int = 8, nv: int = 8,
                                metric: Metric | None = None):
    """Two patches whose shared-edge fixed constraints disagree, used to
    verify that the conflict is reported and the concession stays on the
    authorized patch."""
    G = _greville_grid(nu, nv)
    pa = np.stack([G[0], G[1], np.zeros((nu, nv))], axis=-1)
    pb = pa.copy()
    # shift B's shared edge (u=0) so the fixed positions cannot both hold
    pb[0, :, 2] += 0.05
    a = BSplineSurface(pa, metric=metric, name="patch_A")
    b = BSplineSurface(pb, metric=metric, name="patch_B")
    return a, b
