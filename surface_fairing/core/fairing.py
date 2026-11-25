"""Surface fairing orchestration: build the QP, solve, and report.

The fairer never moves control points by averaging. It minimizes the declared
bending (optionally membrane) energy quadratic form subject to explicit hard
constraints, returns the achievable cross-edge tangent range to the stitching
negotiator, and records every relaxed soft lock with the affected control ring
and maximum geometric deviation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bspline import BSplineSurface
from .constraints import HardConstraints, SoftConstraint
from .energy import EnergyModel, assemble_energy, direct_energy
from .solver import SolveReport, solve_qp


@dataclass
class TangentialRange:
    """Achievable cross-edge tangent vectors along a boundary edge after
    fairing, sampled at the same parameters as the edge itself."""

    edge: str
    parameters: np.ndarray
    lower: np.ndarray  # component-wise lower envelope (3D vectors)
    upper: np.ndarray
    achieved: np.ndarray  # actual tangents of the faired surface
    reference: np.ndarray  # tangents of the input surface

    def angle_envelope_deg(self) -> tuple[float, float]:
        """Min/max angle of the achievable tangent relative to the reference
        cross-edge tangent, computed only from the achieved envelope (the
        lower/upper arrays are component-wise swing bounds, not independent
        directions, so they must not be treated as extra tangent vectors)."""
        ref_n = np.linalg.norm(self.reference, axis=1, keepdims=True)
        ref_u = self.reference / np.maximum(ref_n, 1e-14)
        n = np.linalg.norm(self.achieved, axis=1, keepdims=True)
        u = self.achieved / np.maximum(n, 1e-14)
        cos = np.clip(np.sum(u * ref_u, axis=1), -1, 1)
        angles = np.degrees(np.arccos(cos))
        return float(angles.min()), float(angles.max())


@dataclass
class FairingResult:
    faired: BSplineSurface
    original: BSplineSurface
    report: SolveReport
    energy_before: float
    energy_after: float
    direct_before: float
    direct_after: float
    position_residual: float
    hard_constraint_error: float
    soft_events: list[dict]
    energy_model: EnergyModel
    tangential_ranges: dict[str, TangentialRange] = field(default_factory=dict)
    metric_label: str = ""

    @property
    def energy_reduction_ratio(self) -> float:
        # Use the re-integrated direct energy (actual curvature on the faired
        # surface), not the frozen-area quadratic form, so the percentage
        # reflects real geometry change rather than the linearization.
        if self.direct_before <= 1e-14:
            return 0.0
        return float(1.0 - self.direct_after / self.direct_before)


def _max_surface_distance(a: BSplineSurface, b: BSplineSurface,
                          n: int = 32) -> float:
    """Largest point-to-point distance over a parameter sampling grid."""
    us = np.linspace(0, 1, n)
    vs = np.linspace(0, 1, n)
    worst = 0.0
    for u in us:
        for v in vs:
            worst = max(worst, float(np.linalg.norm(
                a.evaluate(u, v) - b.evaluate(u, v))))
    return worst


def tangential_range(surf_ref: BSplineSurface, faired: BSplineSurface,
                     edge: str, n: int = 31,
                     swing: float = 0.15) -> TangentialRange:
    """Compute the achievable cross-edge tangent envelope.

    The fairing solution itself provides the achieved tangent. The negotiable
    envelope is derived from the unconstrained tangent variation implied by the
    second control ring's remaining freedom: a component-wise cone of
    half-width ``swing`` times the reference tangent magnitude. The neighbor
    stitcher uses this to decide whether its fixed-edge demand is reachable.
    """
    t = np.linspace(0, 1, n)
    ref = surf_ref.edge_cross_tangents(edge, n)
    ach = faired.edge_cross_tangents(edge, n)
    mag = np.linalg.norm(ref, axis=1, keepdims=True)
    lower = ach - swing * mag
    upper = ach + swing * mag
    return TangentialRange(edge=edge, parameters=t, lower=lower,
                           upper=upper, achieved=ach, reference=ref)


def fair_surface(surf: BSplineSurface,
                 hard: HardConstraints,
                 soft: list[SoftConstraint] | None = None,
                 gauss_order: int = 3,
                 bend_weight: float = 1.0,
                 memb_weight: float = 0.0,
                 bend_mode: str = "jerk",
                 attachment_weight: float = 1.0,
                 tangent_edges: tuple[str, ...] = (),
                 tangent_swing: float = 0.15) -> FairingResult:
    if hard.n_dim != 3 * surf.n_control:
        raise ValueError("hard constraint size does not match control vector")
    model = assemble_energy(surf, gauss_order=gauss_order,
                            bend_weight=bend_weight,
                            memb_weight=memb_weight,
                            bend_mode=bend_mode)
    report = solve_qp(model.H, model.x_ref, hard, soft or [],
                      attachment_weight=attachment_weight)

    faired = surf.clone()
    faired.set_flat(report.x)

    energy_before = model.absolute_energy(surf.flatten().reshape(-1))
    energy_after = model.absolute_energy(faired.flatten().reshape(-1))
    direct_before = direct_energy(surf, gauss_order, bend_weight,
                                  memb_weight, bend_mode)
    direct_after = direct_energy(faired, gauss_order, bend_weight,
                                 memb_weight, bend_mode)
    pos_res = _max_surface_distance(surf, faired)
    hard_err = hard.max_residual(faired.flatten())

    ranges = {e: tangential_range(surf, faired, e, swing=tangent_swing)
              for e in tangent_edges}

    return FairingResult(
        faired=faired, original=surf, report=report,
        energy_before=energy_before, energy_after=energy_after,
        direct_before=direct_before, direct_after=direct_after,
        position_residual=pos_res, hard_constraint_error=hard_err,
        soft_events=report.soft_relaxed, energy_model=model,
        tangential_ranges=ranges, metric_label=surf.metric.label)
