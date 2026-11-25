"""Constraint-document maintenance: self-constraint graph, minimal unsatisfiable
subsets, pruning versus relaxation, and reactivation conditions.

The fairing solver refuses a set of hard equalities it cannot satisfy. This
module explains *why* such a set fails and what to do about it, using only
inspectable linear algebra on the same constraint rows the solver consumes:

* A **self-constraint graph** whose nodes are constraint units (one edge lock,
  one key-point pin, one soft ring lock, ...) and whose edges connect units
  that cannot both hold *exactly*.
* Two kinds of resolution are distinguished:
  - ``prune``: a pair of hard equalities that is rank-inconsistent. No
    relaxation amount, weight or tolerance reconciles them, so one of the two
    constraints must be deleted.
  - ``relax``: a hard unit paired with a *soft* lock. The soft lock is a
    penalty, not an equality, so it never invalidates the hard solution; the
    trade-off is how much geometric deviation the lock must accept. A
    weight sweep reports the feasible boundary: the minimum deviation and
    whether it fits a requested tolerance.
* ``find_minimal_unsatisfiable`` uses the deletion filter to shrink an
  inconsistent hard set to a minimal conflicting subset: removing any one of
  its members makes the rest consistent.
* ``recompute_after_removal`` deletes one member, re-runs the real fairing, and
  reports the condition under which the deleted constraint can be re-imposed:
  its target must equal the value the recomputed surface actually produces
  (``A x*``), or the conflicting partner must be removed/relaxed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .bspline import BSplineSurface
from .constraints import HardConstraints, SoftConstraint, make_ring_soft_lock
from .fairing import FairingResult, fair_surface

EDGES = ("u0", "u1", "v0", "v1")


@dataclass
class ConstraintUnit:
    """One logically grouped constraint (all rows that must move together)."""

    label: str
    kind: str                       # "hard" | "soft"
    description: str
    A: np.ndarray                   # (k, n)
    b: np.ndarray                   # (k,)
    weight: float = 0.0
    ring_indices: np.ndarray | None = None
    target_points: np.ndarray | None = None

    @property
    def is_hard(self) -> bool:
        return self.kind == "hard"


@dataclass
class ConflictEdge:
    a: str
    b: str
    kind: str                       # "prune" | "relax"
    detail: str
    min_deviation: float | None = None
    feasible_at_tolerance: bool | None = None


@dataclass
class RelaxationBoundary:
    label: str
    weights: list[float]
    deviations: list[float]
    hard_residuals: list[float]
    min_weight_feasible: float | None
    min_deviation: float
    feasible_at_tolerance: bool
    tolerance: float


@dataclass
class ConstraintGraphReport:
    nodes: list[str]
    edges: list[ConflictEdge]
    minimal_conflict: list[str] | None
    relaxation_boundaries: dict[str, RelaxationBoundary] = field(
        default_factory=dict)

    @property
    def all_consistent(self) -> bool:
        return self.minimal_conflict is None


@dataclass
class RemovalReport:
    removed_label: str
    fairing: FairingResult | None
    error: str | None
    reimposed_target: np.ndarray | None
    achieved_target: np.ndarray | None
    reactivation_gap: float | None
    condition: str


# --------------------------------------------------------------------- units
def _group_label(description: str) -> str:
    """Collapse rows that share one logical constraint.

    A key-point pin emits three rows described ``key point S(u,v) axis k``;
    grouping on the part before `` axis `` keeps them one unit. Edge and ring
    locks already share one description for all their rows.
    """
    if " axis " in description:
        return description.split(" axis ")[0]
    return description


def units_from_hard(hard: HardConstraints) -> list[ConstraintUnit]:
    buckets: dict[str, list[int]] = {}
    order: list[str] = []
    for k, desc in enumerate(hard.descriptions):
        label = _group_label(desc)
        if label not in buckets:
            buckets[label] = []
            order.append(label)
        buckets[label].append(k)
    units = []
    for label in order:
        idx = buckets[label]
        rows = np.asarray([hard.rows[k] for k in idx], dtype=float)
        rhs = np.asarray([hard.rhs[k] for k in idx], dtype=float)
        units.append(ConstraintUnit(
            label=label, kind="hard", description=label, A=rows, b=rhs))
    return units


def units_from_soft(soft: list[SoftConstraint]) -> list[ConstraintUnit]:
    return [ConstraintUnit(
        label=sc.description, kind="soft", description=sc.description,
        A=np.asarray(sc.A, dtype=float), b=np.asarray(sc.b, dtype=float),
        weight=float(sc.weight), ring_indices=np.asarray(sc.ring_indices),
        target_points=np.asarray(sc.target_points)) for sc in soft]


def _stack(units: list[ConstraintUnit]) -> tuple[np.ndarray, np.ndarray]:
    if not units:
        return np.zeros((0, 0)), np.zeros(0)
    A = np.vstack([u.A for u in units])
    b = np.concatenate([u.b for u in units])
    return A, b


def _consistent_units(units: list[ConstraintUnit], tol: float = 1e-9) -> bool:
    """True iff every equality in ``units`` can hold simultaneously."""
    A, b = _stack(units)
    if A.size == 0:
        return True
    atol = tol * max(float(np.linalg.norm(A)), 1.0)
    rank_a = np.linalg.matrix_rank(A, tol=atol)
    rank_ab = np.linalg.matrix_rank(np.hstack([A, b[:, None]]), tol=atol)
    return rank_a == rank_ab


def find_minimal_unsatisfiable(hard_units: list[ConstraintUnit],
                               tol: float = 1e-9) -> list[str] | None:
    """Deletion-filter minimal conflicting subset of hard units.

    Returns ``None`` when the full hard set is consistent. Otherwise returns a
    minimal set: it is inconsistent, but dropping any single member leaves the
    remainder consistent.
    """
    if _consistent_units(hard_units, tol):
        return None
    kept = list(hard_units)
    for unit in list(kept):
        trial = [u for u in kept if u.label != unit.label]
        if trial and not _consistent_units(trial, tol):
            kept = trial  # unit does not participate in the conflict
    # A one-unit conflict cannot occur because a single equality is always
    # self-consistent, so ``kept`` has at least two members here.
    return [u.label for u in kept]


# --------------------------------------------------------------- relaxation
def _hard_from_units(units: list[ConstraintUnit],
                     n_dim: int) -> HardConstraints:
    out = HardConstraints(n_dim)
    for u in units:
        for row, val in zip(u.A, u.b):
            out.rows.append(np.asarray(row, dtype=float))
            out.rhs.append(float(val))
            out.descriptions.append(u.description)
            nz = np.flatnonzero(row)
            if nz.size == 1 and abs(row[nz[0]] - 1.0) < 1e-12:
                out.pinned_dof.append(int(nz[0]))
    return out


def relaxation_sweep(surf: BSplineSurface, hard: HardConstraints,
                     edge: str, weights: list[float],
                     tolerance: float = 1e-3,
                     attachment_weight: float = 0.3) -> RelaxationBoundary:
    """Sweep soft-lock weights and report the feasible deviation boundary.

    The soft ring lock never invalidates a hard solution; the real question is
    how far the ring must deviate to keep the hard equalities exact. For each
    weight the fairing is re-run and the *max geometric deviation of the
    affected ring* is measured; ``min_deviation`` is the smallest such value,
    the least relaxation the lock must accept. ``min_weight_feasible`` is the
    lightest weight whose deviation already fits ``tolerance``.
    """
    idx = surf.ring_indices(edge)
    targets = surf.flatten()[idx].copy()
    devs, hard_res, feasible_weights = [], [], []
    for w in weights:
        soft = [make_ring_soft_lock(surf, edge, weight=float(w))]
        res = fair_surface(surf, _copy_hard(hard), soft=soft,
                           attachment_weight=attachment_weight)
        cur = res.report.x.reshape(-1, 3)[idx]
        dev = float(np.max(np.linalg.norm(cur - targets, axis=1)))
        devs.append(dev)
        hard_res.append(float(res.hard_constraint_error))
        if dev <= tolerance:
            feasible_weights.append(float(w))
    min_dev = float(min(devs)) if devs else float("inf")
    return RelaxationBoundary(
        label=f"soft ring lock {edge}", weights=[float(w) for w in weights],
        deviations=devs, hard_residuals=hard_res,
        min_weight_feasible=min(feasible_weights) if feasible_weights else None,
        min_deviation=min_dev,
        feasible_at_tolerance=bool(min_dev <= tolerance),
        tolerance=float(tolerance))


def _copy_hard(hard: HardConstraints) -> HardConstraints:
    out = HardConstraints(hard.n_dim)
    out.rows = [np.asarray(r, dtype=float).copy() for r in hard.rows]
    out.rhs = [float(v) for v in hard.rhs]
    out.descriptions = list(hard.descriptions)
    out.pinned_dof = list(hard.pinned_dof)
    return out


def _edge_from_label(label: str) -> str:
    for token in EDGES:
        if label.rstrip().endswith(token):
            return token
    return label.rsplit(" ", 1)[-1]


# ------------------------------------------------------------------- graph
def build_conflict_graph(hard_units: list[ConstraintUnit],
                         soft_units: list[ConstraintUnit] | None = None,
                         surf: BSplineSurface | None = None,
                         relaxation_weights: tuple[float, ...] = (),
                         relaxation_tolerance: float = 1e-3,
                         tol: float = 1e-9) -> ConstraintGraphReport:
    """Classify every conflicting pair as ``prune`` or ``relax``.

    Hard-hard pairs are ``prune`` when rank-inconsistent; those are the only
    pairs that make the hard set unsatisfiable. Hard-soft pairs are ``relax``:
    a soft lock is a penalty, so the hard solution stays exact and the
    resolution is an accepted deviation, optionally quantified by a weight
    sweep when ``surf`` and weights are supplied. Soft-soft pairs do not
    conflict (two penalties can coexist).
    """
    soft_units = soft_units or []
    nodes = [u.label for u in hard_units] + [u.label for u in soft_units]
    edges: list[ConflictEdge] = []
    boundaries: dict[str, RelaxationBoundary] = {}

    for a, b in combinations(hard_units, 2):
        if not _consistent_units([a, b], tol):
            edges.append(ConflictEdge(
                a=a.label, b=b.label, kind="prune",
                detail=("hard equalities are rank-inconsistent; no relaxation "
                        "amount can satisfy both, delete one")))

    if surf is not None and relaxation_weights and soft_units:
        hard_all = _hard_from_units(hard_units, 3 * surf.n_control)
        hard_feasible = _consistent_units(hard_units, tol)
        if not hard_feasible:
            # A soft lock cannot resolve a hard-hard conflict; flag it so the
            # caller knows a relaxation sweep is meaningless until one hard
            # constraint is pruned.
            boundaries["_hard_infeasible"] = RelaxationBoundary(
                label="hard set infeasible", weights=[], deviations=[],
                hard_residuals=[], min_weight_feasible=None,
                min_deviation=float("inf"), feasible_at_tolerance=False,
                tolerance=relaxation_tolerance)
        else:
            for s in soft_units:
                edge = _edge_from_label(s.label)
                boundary = relaxation_sweep(
                    surf, hard_all, edge, list(relaxation_weights),
                    tolerance=relaxation_tolerance)
                boundaries[s.label] = boundary
                edges.append(ConflictEdge(
                    a="hard constraints", b=s.label, kind="relax",
                    detail=("soft lock is a penalty, not an equality: the hard "
                            "set stays exact and the ring accepts the reported "
                            "deviation"),
                    min_deviation=boundary.min_deviation,
                    feasible_at_tolerance=boundary.feasible_at_tolerance))

    return ConstraintGraphReport(
        nodes=nodes, edges=edges,
        minimal_conflict=find_minimal_unsatisfiable(hard_units, tol),
        relaxation_boundaries=boundaries)


def recompute_after_removal(surf: BSplineSurface,
                            hard: HardConstraints,
                            remove_label: str,
                            soft: list[SoftConstraint] | None = None,
                            **fairing_kwargs) -> RemovalReport:
    """Delete one hard constraint unit, re-run the real fairing, and state the
    condition under which the removed constraint can take effect again.

    The reactivation target is ``A_removed x*`` where ``x*`` is the control
    vector the remaining constraints actually produce: a linear equality can
    only be re-imposed if its right-hand side matches what the constrained
    surface can deliver.
    """
    units = units_from_hard(hard)
    removed = next((u for u in units if u.label == remove_label), None)
    if removed is None:
        raise KeyError(f"no hard constraint unit labelled {remove_label!r}")
    remaining = [u for u in units if u.label != remove_label]
    rebuilt = _hard_from_units(remaining, hard.n_dim)
    try:
        res = fair_surface(surf, rebuilt, soft=soft or [], **fairing_kwargs)
    except Exception as exc:  # InfeasibleConstraints etc. must be reported
        return RemovalReport(
            removed_label=remove_label, fairing=None, error=str(exc),
            reimposed_target=removed.b.copy(), achieved_target=None,
            reactivation_gap=None,
            condition=f"remaining constraints are still infeasible: {exc}")

    achieved = removed.A @ res.report.x
    gap = float(np.max(np.abs(removed.b - achieved)))
    if gap <= 1e-9:
        condition = ("removed constraint is already satisfied by the recomputed "
                     "surface; it can be re-imposed unchanged")
    else:
        condition = (
            f"removed constraint reactivates only if its target is changed to "
            f"{np.array2string(achieved, precision=6)} (gap {gap:.6g}), or if "
            f"the conflicting partner is removed or relaxed")
    return RemovalReport(
        removed_label=remove_label, fairing=res, error=None,
        reimposed_target=removed.b.copy(), achieved_target=achieved,
        reactivation_gap=gap, condition=condition)


def analyze(surf: BSplineSurface, hard: HardConstraints,
            soft: list[SoftConstraint] | None = None,
            relaxation_weights: tuple[float, ...] = (
                1.0, 10.0, 100.0, 1000.0),
            relaxation_tolerance: float = 1e-3) -> ConstraintGraphReport:
    """Convenience wrapper: build the graph from a live constraint set."""
    return build_conflict_graph(
        units_from_hard(hard), units_from_soft(soft or []),
        surf=surf, relaxation_weights=relaxation_weights,
        relaxation_tolerance=relaxation_tolerance)
