"""Hard and soft constraints on control points, all kept inspectable.

A constraint is a linear equality on the flattened control vector
``x`` (length 3*m, coordinates interleaved):

    A x = b

Hard constraints must hold exactly and enter the KKT system. Soft constraints
(ridge/edge locks that conflict with the fairing fit) carry a weight and are
relaxed by the optimizer; the affected control rings and the maximum resulting
geometric deviation are reported instead of being silently ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bspline import BSplineSurface


@dataclass
class HardConstraints:
    """Exact equality constraints A x = b on the flattened control vector."""

    n_dim: int
    rows: list[np.ndarray] = field(default_factory=list)
    rhs: list[float] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)
    pinned_dof: list[int] = field(default_factory=list)

    def pin_dof(self, flat_index: int, axis: int, value: float,
                description: str = "") -> None:
        row = np.zeros(self.n_dim)
        dof = 3 * flat_index + axis
        row[dof] = 1.0
        self.rows.append(row)
        self.rhs.append(float(value))
        self.descriptions.append(description or f"pin P{flat_index}[{axis}]")
        self.pinned_dof.append(dof)

    def pin_point(self, flat_index: int, point: np.ndarray,
                  description: str = "") -> None:
        for axis in range(3):
            self.pin_dof(flat_index, axis, point[axis],
                         description or f"pin point P{flat_index}")

    def pin_edge(self, surf: BSplineSurface, edge: str) -> np.ndarray:
        """Fix every control point of a boundary edge; returns flat indices."""
        idx = surf.edge_indices(edge)
        for k in idx:
            self.pin_point(k, surf.flatten()[k], f"hard edge lock {edge}")
        return idx

    def pin_ring(self, surf: BSplineSurface, edge: str) -> np.ndarray:
        """Fix the control ring adjacent to an edge (used for tangent locks)."""
        idx = surf.ring_indices(edge)
        for k in idx:
            self.pin_point(k, surf.flatten()[k], f"hard ring lock {edge}")
        return idx

    def pin_surface_point(self, surf: BSplineSurface, u: float, v: float,
                          position: np.ndarray | None = None) -> None:
        """Constrain S(u, v) to a position via the active basis coefficients.

        With position=None the point is locked at its current location (key
        feature preservation). The constraint is a genuine linear combination
        of control points, not a control-point average.
        """
        from .bspline import basis_table
        Nu, _, _, _ = basis_table(u, surf.knots_u, surf.degree_u)
        Nv, _, _, _ = basis_table(v, surf.knots_v, surf.degree_v)
        if position is None:
            position = surf.evaluate(u, v)
        for axis in range(3):
            row = np.zeros(self.n_dim)
            for i, nu_i in enumerate(Nu):
                if abs(nu_i) < 1e-14:
                    continue
                for j, nv_j in enumerate(Nv):
                    if abs(nv_j) < 1e-14:
                        continue
                    row[3 * (i * surf.nv + j) + axis] = nu_i * nv_j
            self.rows.append(row)
            self.rhs.append(float(position[axis]))
            self.descriptions.append(
                f"key point S({u:.3f},{v:.3f}) axis {axis}")

    def matrix(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.rows:
            return np.zeros((0, self.n_dim)), np.zeros(0)
        return np.asarray(self.rows), np.asarray(self.rhs)

    @property
    def n_constraints(self) -> int:
        return len(self.rows)

    def residual(self, x: np.ndarray) -> np.ndarray:
        A, b = self.matrix()
        if A.size == 0:
            return np.zeros(0)
        return A @ np.asarray(x).reshape(-1) - b

    def max_residual(self, x: np.ndarray) -> float:
        r = self.residual(x)
        return float(np.max(np.abs(r))) if r.size else 0.0


@dataclass
class SoftConstraint:
    """Weighted equality W (A x = b); the optimizer may relax it."""

    A: np.ndarray
    b: np.ndarray
    weight: float
    description: str
    ring_indices: np.ndarray
    target_points: np.ndarray

    def violation(self, x: np.ndarray) -> np.ndarray:
        return self.A @ np.asarray(x).reshape(-1) - self.b

    def max_deviation(self, x: np.ndarray) -> float:
        """Largest 3D distance of a ring control point from its locked target."""
        xx = np.asarray(x).reshape(-1)
        cur = xx.reshape(-1, 3)[self.ring_indices]
        return float(np.max(np.linalg.norm(cur - self.target_points, axis=1)))


def make_ring_soft_lock(surf: BSplineSurface, edge: str,
                        weight: float) -> SoftConstraint:
    """Soft-lock the control ring adjacent to ``edge`` at its current place."""
    idx = surf.ring_indices(edge)
    m = surf.n_control
    rows, rhs = [], []
    x0 = surf.flatten().reshape(-1)
    for k in idx:
        for axis in range(3):
            row = np.zeros(3 * m)
            row[3 * k + axis] = 1.0
            rows.append(row)
            rhs.append(x0[3 * k + axis])
    return SoftConstraint(
        A=np.asarray(rows), b=np.asarray(rhs), weight=weight,
        description=f"soft ridge/ring lock {edge}",
        ring_indices=idx, target_points=surf.flatten()[idx].copy())


def detect_duplicate_dofs(hard: HardConstraints) -> list[int]:
    """Return DOFs pinned by more than one hard row with conflicting values."""
    seen: dict[int, float] = {}
    conflicts = []
    A, b = hard.matrix()
    if A.size == 0:
        return conflicts
    for row, val in zip(A, b):
        nz = np.flatnonzero(row)
        if nz.size != 1 or row[nz[0]] != 1.0:
            continue  # only single-DOF pin rows can be directly contradictory
        dof = int(nz[0])
        if dof in seen and abs(seen[dof] - val) > 1e-10:
            conflicts.append(dof)
        seen.setdefault(dof, val)
    return conflicts
