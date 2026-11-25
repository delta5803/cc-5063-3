"""Clamped uniform B-spline surfaces with analytic derivatives.

The surface is

    S(u, v) = sum_i sum_j P_ij * N_i^p(u) * M_j^q(v),  (u, v) in [0, 1]^2

with P_ij the control points and N, M the clamped B-spline basis functions.
All derivatives used by the energy, curvature and isophote code are evaluated
analytically from the basis functions; no finite differences and no vertex
normal averaging are used anywhere in the package.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def clamped_knots(n: int, degree: int) -> np.ndarray:
    """Clamped uniform knot vector for ``n`` control points and given degree."""
    if n <= degree:
        raise ValueError(f"need n ({n}) > degree ({degree})")
    inner = np.linspace(0.0, 1.0, n - degree + 1)
    return np.concatenate([np.zeros(degree), inner, np.ones(degree)])


def greville_abscissae(n: int, degree: int) -> np.ndarray:
    """Greville abscissae: control points placed here make a clamped spline
    reproduce the identity parameterization (and hence a flat plane) exactly."""
    knots = clamped_knots(n, degree)
    return np.array([knots[i + 1:i + 1 + degree].mean() if degree > 0
                     else knots[i] for i in range(n)])


def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-15 else 0.0


def basis_table(t: float, knots: np.ndarray, degree: int):
    """Return (N, dN, ddN, dddN): values and first three derivatives of every
    basis function at parameter ``t`` for a clamped knot vector. Entries beyond
    the basis degree are returned as zeros (e.g. dddN for a quadratic spline).

    The implementation keeps, for every recursion order d, a full-length array
    of global basis values, so every index below is a global basis index and
    there is no local/global index remapping to get wrong.
    """
    n = len(knots) - degree - 1

    def order_zero() -> np.ndarray:
        # Length n+1 with a trailing zero so derivative formulas can index i+1
        # at the last global basis without special cases.
        out = np.zeros(n + 1)
        if t >= 1.0:
            out[n - 1] = 1.0
            return out
        for i in range(n):
            if knots[i] <= t < knots[i + 1]:
                out[i] = 1.0
                break
        return out

    levels = [order_zero()]
    for d in range(1, degree + 1):
        prev = levels[d - 1]
        cur = np.zeros(n + 1)
        for i in range(n):
            a = _safe_div(t - knots[i], knots[i + d] - knots[i])
            b = _safe_div(knots[i + d + 1] - t,
                          knots[i + d + 1] - knots[i + 1])
            cur[i] = a * prev[i] + b * prev[i + 1]
        levels.append(cur)

    N = levels[degree][:n]

    def derivative(order: int) -> np.ndarray:
        """Derivative of the order-``degree`` basis by applying the standard
        recursion ``order`` times to the lower-order basis arrays."""
        if order == 0:
            return N.copy()
        vals = levels[degree - order].copy()  # order-(degree-order) basis
        # Each recursion differentiates the current basis, whose order starts
        # at (degree-order) and grows by one after every step.
        for step in range(order):
            base_degree = degree - order + step
            out = np.zeros(n + 1)
            for i in range(n):
                a = _safe_div(base_degree + 1,
                              knots[i + base_degree + 1] - knots[i])
                b = _safe_div(base_degree + 1,
                              knots[i + base_degree + 2] - knots[i + 1])
                out[i] = a * vals[i] - b * vals[i + 1]
            vals = out
        return vals[:n]

    dN = derivative(1) if degree >= 1 else np.zeros(n)
    ddN = derivative(2) if degree >= 2 else np.zeros(n)
    dddN = derivative(3) if degree >= 3 else np.zeros(n)
    return N, dN, ddN, dddN


@dataclass
class Metric:
    """Declared parameter-to-physical scale.

    ``scale_u`` / ``scale_v`` are assumed constant over the patch: physical
    arc length along a parameter line is approximated by scale * d(parameter).
    This makes the energy cheap and inspectable, but it cannot capture
    non-uniform parameterization; that limitation is reported in the UI and
    documentation.
    """

    scale_u: float = 1.0
    scale_v: float = 1.0

    @property
    def label(self) -> str:
        return (f"constant declared metric: ds~{self.scale_u:.4g}*du, "
                f"dt~{self.scale_v:.4g}*dv (non-uniform parameterization "
                f"is not represented)")


class BSplineSurface:
    def __init__(self, control_points: np.ndarray, degree_u: int = 3,
                 degree_v: int = 3, metric: Metric | None = None,
                 knots_u: np.ndarray | None = None,
                 knots_v: np.ndarray | None = None,
                 name: str = "patch"):
        self.P = np.asarray(control_points, dtype=float)
        if self.P.ndim != 3 or self.P.shape[2] != 3:
            raise ValueError("control_points must have shape (nu, nv, 3)")
        self.nu, self.nv, _ = self.P.shape
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.knots_u = clamped_knots(self.nu, degree_u) if knots_u is None else np.asarray(knots_u, dtype=float)
        self.knots_v = clamped_knots(self.nv, degree_v) if knots_v is None else np.asarray(knots_v, dtype=float)
        self.metric = metric or Metric()
        self.name = name

    # ------------------------------------------------------------------ basics
    def clone(self) -> "BSplineSurface":
        return BSplineSurface(self.P.copy(), self.degree_u, self.degree_v,
                              Metric(self.metric.scale_u, self.metric.scale_v),
                              self.knots_u.copy(), self.knots_v.copy(), self.name)

    @property
    def n_control(self) -> int:
        return self.nu * self.nv

    def basis_u(self, u: float):
        return basis_table(u, self.knots_u, self.degree_u)

    def basis_v(self, v: float):
        return basis_table(v, self.knots_v, self.degree_v)

    def partials(self, u: float, v: float) -> dict[str, np.ndarray]:
        """Analytic S through third partials and unit normal at (u, v)."""
        Nu, dNu, ddNu, dddNu = self.basis_u(u)
        Nv, dNv, ddNv, dddNv = self.basis_v(v)
        P = self.P
        S = np.einsum("i,j,ijk->k", Nu, Nv, P)
        Su = np.einsum("i,j,ijk->k", dNu, Nv, P)
        Sv = np.einsum("i,j,ijk->k", Nu, dNv, P)
        Suu = np.einsum("i,j,ijk->k", ddNu, Nv, P)
        Suv = np.einsum("i,j,ijk->k", dNu, dNv, P)
        Svv = np.einsum("i,j,ijk->k", Nu, ddNv, P)
        Suuu = np.einsum("i,j,ijk->k", dddNu, Nv, P)
        Suuv = np.einsum("i,j,ijk->k", ddNu, dNv, P)
        Suvv = np.einsum("i,j,ijk->k", dNu, ddNv, P)
        Svvv = np.einsum("i,j,ijk->k", Nu, dddNv, P)
        cross = np.cross(Su, Sv)
        norm = np.linalg.norm(cross)
        n = cross / norm if norm > 1e-14 else np.zeros(3)
        return {"S": S, "Su": Su, "Sv": Sv, "Suu": Suu, "Suv": Suv,
                "Svv": Svv, "Suuu": Suuu, "Suuv": Suuv, "Suvv": Suvv,
                "Svvv": Svvv, "n": n, "jacobian": norm}

    def evaluate(self, u: float, v: float) -> np.ndarray:
        return self.partials(u, v)["S"]

    def normal(self, u: float, v: float) -> np.ndarray:
        return self.partials(u, v)["n"]

    def sample_grid(self, nu: int = 24, nv: int = 24):
        us = np.linspace(0.0, 1.0, nu)
        vs = np.linspace(0.0, 1.0, nv)
        pts = np.zeros((nu, nv, 3))
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                pts[i, j] = self.evaluate(u, v)
        return us, vs, pts

    # ------------------------------------------------------------- control data
    def flatten(self) -> np.ndarray:
        return self.P.reshape(-1, 3)

    def set_flat(self, x: np.ndarray) -> None:
        self.P = x.reshape(self.nu, self.nv, 3).copy()

    def edge_indices(self, edge: str) -> np.ndarray:
        """Flat control-point indices of a boundary edge.

        edge in {'u0','u1','v0','v1'}: u0 means u=0 (row i=0), etc.
        """
        if edge == "u0":
            return np.arange(self.nv)
        if edge == "u1":
            return (self.nu - 1) * self.nv + np.arange(self.nv)
        if edge == "v0":
            return np.arange(self.nu) * self.nv
        if edge == "v1":
            return np.arange(self.nu) * self.nv + (self.nv - 1)
        raise ValueError(f"unknown edge {edge!r}")

    def ring_indices(self, edge: str) -> np.ndarray:
        """Flat indices of the control ring adjacent to ``edge`` (second row)."""
        if edge == "u0":
            return self.nv + np.arange(self.nv)
        if edge == "u1":
            return (self.nu - 2) * self.nv + np.arange(self.nv)
        if edge == "v0":
            return np.arange(self.nu) * self.nv + 1
        if edge == "v1":
            return np.arange(self.nu) * self.nv + (self.nv - 2)
        raise ValueError(f"unknown edge {edge!r}")

    def edge_samples(self, edge: str, n: int = 40) -> np.ndarray:
        t = np.linspace(0.0, 1.0, n)
        out = np.zeros((n, 3))
        for k, tv in enumerate(t):
            if edge == "u0":
                out[k] = self.evaluate(0.0, tv)
            elif edge == "u1":
                out[k] = self.evaluate(1.0, tv)
            elif edge == "v0":
                out[k] = self.evaluate(tv, 0.0)
            elif edge == "v1":
                out[k] = self.evaluate(tv, 1.0)
        return out

    def edge_cross_tangents(self, edge: str, n: int = 40) -> np.ndarray:
        """Analytic inward cross-edge tangent vectors along the edge."""
        t = np.linspace(0.0, 1.0, n)
        out = np.zeros((n, 3))
        su, sv = self.metric.scale_u, self.metric.scale_v
        for k, tv in enumerate(t):
            d = self.partials(*{
                "u0": (0.0, tv), "u1": (1.0, tv),
                "v0": (tv, 0.0), "v1": (tv, 1.0)}[edge])
            out[k] = d["Su"] / su if edge in ("u0", "u1") else d["Sv"] / sv
        return out

    def declared_edge_lengths(self) -> tuple[float, float]:
        """Physical edge lengths implied by the declared metric."""
        return self.metric.scale_u, self.metric.scale_v
