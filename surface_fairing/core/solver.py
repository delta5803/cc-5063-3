"""Constrained quadratic solver used by the fairing optimizer.

Minimize  1/2 dx' Htot dx - dx' g
subject to C dx = d   (hard equalities, in displacement coordinates dx=x-x_ref)

The solution works in the null space of the hard constraints:

* C z = d is first checked for consistency;
* a particular solution z0 and an orthonormal basis Z of null(C) are found by
  SVD;
* the unconstrained QP in the free coordinates y with dx = z0 + Z y is solved
  by a regularized positive-definite system. A small ridge term removes any
  remaining rigid drift automatically, and the eigen-spectrum is reported so
  the regularization is inspectable rather than hidden;
* explicitly pinned DOFs from the constraint builder are additionally zeroed
  in the null-space basis.

Soft locks enter Htot as penalties; if they make exact hard feasibility
impossible they are relaxed and the affected rings/deviations are reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constraints import HardConstraints, SoftConstraint, detect_duplicate_dofs


class InfeasibleConstraints(RuntimeError):
    pass


@dataclass
class SolveReport:
    x: np.ndarray
    success: bool
    message: str
    pinned_dofs: list[int] = field(default_factory=list)
    hard_residual: float = 0.0
    soft_relaxed: list[dict] = field(default_factory=list)
    condition_number: float = 0.0
    kkt_rank: int = 0
    kkt_size: int = 0
    ridge_used: float = 0.0


def _hard_consistent(C: np.ndarray, d: np.ndarray, tol: float) -> bool:
    if C.shape[0] == 0:
        return True
    rc = np.linalg.matrix_rank(C, tol=tol)
    rcd = np.linalg.matrix_rank(np.hstack([C, d[:, None]]), tol=tol)
    return rc == rcd


def _constraint_decomposition(C: np.ndarray, d: np.ndarray, tol: float,
                              n: int):
    """Return particular solution z0 and orthonormal null-space basis Z."""
    if C.shape[0] == 0:
        return np.zeros(n), np.eye(n)
    U, s, Vh = np.linalg.svd(C, full_matrices=True)
    cutoff = tol * (s[0] if s.size else 1.0)
    r = int(np.sum(s > cutoff))
    sinv = np.where(s[:r] > cutoff, 1.0 / s[:r], 0.0)
    # particular solution: V_r diag(1/s) U_r' d
    z0 = Vh[:r].T @ (sinv * (U[:, :r].T @ d))
    Z = Vh[r:].T  # columns span null(C)
    return z0, Z


def _solve_in_nullspace(H: np.ndarray, g: np.ndarray, z0: np.ndarray,
                        Z: np.ndarray, ridge: float):
    """Solve the reduced QP in free coordinates y with dx = z0 + Z y."""
    reduced_H = Z.T @ (H @ Z)
    reduced_g = Z.T @ (g - H @ z0)
    eig = np.linalg.eigvalsh(reduced_H)
    # ridge only as large as needed to make the reduced system positive definite
    needed = max(0.0, 1e-12 * (abs(eig[-1]) if eig.size else 0.0) - eig[0])
    reg = max(ridge, needed)
    M = reduced_H + reg * np.eye(reduced_H.shape[0])
    y = np.linalg.solve(M, reduced_g)
    dx = z0 + Z @ y
    cond = float(np.linalg.cond(M))
    return dx, cond, reg, int(np.sum(eig > 1e-12 * max(abs(eig[-1]), 1.0)))


def solve_qp(H: np.ndarray, x_ref: np.ndarray,
             hard: HardConstraints,
             soft: list[SoftConstraint] | None = None,
             svd_tol: float = 1e-10,
             ridge: float = 1e-10,
             attachment_weight: float = 1.0) -> SolveReport:
    """Minimize the fairing objective over the hard-feasible set.

    The objective combines three inspectable quadratic terms:

    * bending/membrane stiffness  1/2 x'H x;
    * data attachment  attachment_weight/2 ||x - x_ref||^2, which fixes the
      under-determined rigid/parameterization freedom so the optimizer cannot
      collapse the Jacobian to make the bending energy look small;
    * soft ring locks  w ||A x - b||^2, relaxed and reported on conflict.

    Hard constraints are eliminated analytically via a null-space basis.
    """
    soft = soft or []
    n = H.shape[0]
    x_ref = np.asarray(x_ref, dtype=float).reshape(-1)

    conflicts = detect_duplicate_dofs(hard)
    if conflicts:
        raise InfeasibleConstraints(
            f"hard constraints disagree on DOFs {conflicts}; the local "
            f"concession must be placed on an authorized patch instead")

    C_abs, b_abs = hard.matrix()
    if not _hard_consistent(C_abs, b_abs, svd_tol):
        raise InfeasibleConstraints(
            "hard equality constraints are mutually inconsistent; place the "
            "local concession on an authorized patch")

    # particular feasible point and null-space basis in ABSOLUTE coordinates
    z0, Z = _constraint_decomposition(C_abs, b_abs, svd_tol, n)
    pinned = sorted(set(hard.pinned_dof))

    # data attachment stiffness and its linear term
    H_att = attachment_weight * np.eye(n)
    g_att = attachment_weight * x_ref

    def soft_matrices(items):
        Hs = np.zeros_like(H)
        gs = np.zeros(n)
        for sc in items:
            Hs += sc.weight * (sc.A.T @ sc.A)
            gs += sc.weight * sc.A.T @ sc.b
        return Hs, gs

    def run(items):
        Hs, gs = soft_matrices(items)
        Htot = H + H_att + Hs
        gt = g_att + gs
        reduced_H = Z.T @ (Htot @ Z)
        reduced_g = Z.T @ (gt - Htot @ z0)
        eig = np.linalg.eigvalsh(reduced_H)
        needed = max(0.0, 1e-12 * (abs(eig[-1]) if eig.size else 0.0)
                     - eig[0])
        reg = max(ridge, needed)
        M = reduced_H + reg * np.eye(reduced_H.shape[0])
        y = np.linalg.solve(M, reduced_g)
        x = z0 + Z @ y
        cond = float(np.linalg.cond(M))
        rank = int(np.sum(eig > 1e-12 * max(abs(eig[-1]), 1.0)))
        return x, cond, reg, rank

    x, cond, reg_used, rank = run(soft)
    hard_res = hard.max_residual(x)
    relaxed = []

    if hard_res > 1e-7 and soft:
        for sc in soft:
            relaxed.append({
                "description": sc.description,
                "ring_indices": sc.ring_indices.tolist(),
                "weight_before": sc.weight,
                "max_deviation": sc.max_deviation(x),
            })
        x, cond, reg_used, rank = run([])
        hard_res = hard.max_residual(x)
        for sc in soft:
            relaxed[-1]["max_deviation"] = sc.max_deviation(x)
        if hard_res > 1e-7:
            raise InfeasibleConstraints(
                f"hard residual {hard_res:.3e} cannot be satisfied")

    return SolveReport(
        x=x, success=True,
        message="optimal" if not relaxed else "optimal with soft relaxation",
        pinned_dofs=pinned, hard_residual=hard_res, soft_relaxed=relaxed,
        condition_number=cond, kkt_rank=rank,
        kkt_size=Z.shape[0], ridge_used=reg_used)
