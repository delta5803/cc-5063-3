"""Generate rippled surfaces and verify boundary displacement automatically.

This is the maintenance/verification entry that turns the fairing claims into
checked statements:

1. Build a rippled surface, run the real fairing, and **automatically** verify
   boundary displacement: locked edges must not move (<= 1e-9), unlocked
   interior must actually change (the ripple is removed, not frozen).
2. Exercise the constraint-maintenance tool on self-constraint graphs:
   - a *prune* conflict (edge lock vs. an off-edge key point at the same
     parameter) is reduced to its minimal conflicting subset;
   - a *relax* case (soft ring lock weight sweep) reports the feasible
     deviation boundary;
   - after deleting one member the surface is re-solved and the reactivation
     condition of the removed constraint is printed.
3. Write a machine-readable log so the guide's maintenance section can quote
   executable evidence rather than prose.

Run::

    PYTHONPATH=. .venv/bin/python scripts/verify_maintenance.py
    PYTHONPATH=. .venv/bin/python scripts/verify_maintenance.py --outdir /tmp/m
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from surface_fairing.core.constraints import HardConstraints, make_ring_soft_lock
from surface_fairing.core.fairing import fair_surface
from surface_fairing.core.maintenance import (
    analyze, find_minimal_unsatisfiable, recompute_after_removal, units_from_hard)
from surface_fairing.io_.artifacts import rippled_curved, rippled_plane
from surface_fairing.io_.export import export_obj

EDGES = ("u0", "u1", "v0", "v1")


def _locked(surf):
    hard = HardConstraints(3 * surf.n_control)
    for edge in EDGES:
        hard.pin_edge(surf, edge)
    return hard


def _edge_displacements(a, b, n=64):
    return {edge: float(np.linalg.norm(
        a.edge_samples(edge, n) - b.edge_samples(edge, n), axis=1).max())
        for edge in EDGES}


def _interior_displacement(a, b, grid=41):
    """Max displacement strictly inside the patch (excludes the locked border)."""
    ts = np.linspace(1e-6, 1 - 1e-6, grid)
    worst = 0.0
    for u in ts:
        for v in ts:
            if u < 0.02 or u > 0.98 or v < 0.02 or v > 0.98:
                continue
            worst = max(worst, float(np.linalg.norm(
                a.evaluate(u, v) - b.evaluate(u, v))))
    return worst


def verify_boundary_displacement(artifact, bend_mode, attachment=0.3):
    """Run fairing and check: locked edges fixed, interior actually changes."""
    s = artifact.surface
    res = fair_surface(s, _locked(s), bend_mode=bend_mode,
                       attachment_weight=attachment)
    edges = _edge_displacements(s, res.faired)
    interior = _interior_displacement(s, res.faired)
    checks = {
        "boundary_fixed": all(v < 1e-9 for v in edges.values()),
        "interior_changed": interior > 1e-6,
        "hard_error_ok": res.hard_constraint_error < 1e-9,
        "energy_decreased": res.direct_after < res.direct_before,
    }
    return {
        "artifact": artifact.name,
        "bend_mode": bend_mode,
        "edge_displacements": edges,
        "interior_displacement": interior,
        "hard_constraint_error": res.hard_constraint_error,
        "energy_before": res.direct_before,
        "energy_after": res.direct_after,
        "energy_reduction_ratio": res.energy_reduction_ratio,
        "checks": checks,
        "passes": all(checks.values()),
    }, res


def prune_case():
    """Edge lock vs. off-edge key point at the same parameter: a hard-hard
    conflict that only pruning resolves."""
    art = rippled_plane()
    s = art.surface
    hard = _locked(s)
    hard.pin_surface_point(s, 0.0, 0.5, position=np.array([0.0, 0.5, 0.5]))
    minimal = find_minimal_unsatisfiable(units_from_hard(hard))
    removal = recompute_after_removal(s, hard, minimal[-1],
                                      bend_mode="normal", attachment_weight=0.3)
    return {
        "kind": "prune",
        "all_units": [u.label for u in units_from_hard(hard)],
        "minimal_conflict": minimal,
        "removed": removal.removed_label,
        "recompute_error": removal.error,
        "reactivation_gap": removal.reactivation_gap,
        "reactivation_condition": removal.condition,
        "hard_error_after_removal": None if removal.fairing is None
        else removal.fairing.hard_constraint_error,
        "boundary_after_removal": None if removal.fairing is None
        else _edge_displacements(s, removal.fairing.faired),
    }


def relax_case():
    """Soft ring lock: the hard set stays feasible; the weight sweep reports how
    much deviation the lock must accept."""
    art = rippled_curved()
    s = art.surface
    hard = _locked(s)
    soft = [make_ring_soft_lock(s, "u0", weight=100.0)]
    report = analyze(s, hard, soft)
    boundary = report.relaxation_boundaries.get(
        "soft ridge/ring lock u0") or next(
            (b for k, b in report.relaxation_boundaries.items()
             if k != "_hard_infeasible"), None)
    return {
        "kind": "relax",
        "minimal_conflict": report.minimal_conflict,
        "edges": [{"kind": e.kind, "a": e.a, "b": e.b,
                   "min_deviation": e.min_deviation,
                   "feasible_at_tolerance": e.feasible_at_tolerance}
                  for e in report.edges],
        "relaxation_boundary": None if boundary is None else {
            "weights": boundary.weights,
            "deviations": boundary.deviations,
            "min_weight_feasible": boundary.min_weight_feasible,
            "min_deviation": boundary.min_deviation,
            "feasible_at_tolerance": boundary.feasible_at_tolerance,
            "tolerance": boundary.tolerance,
        },
    }


def main(outdir: str) -> int:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    plane_report, plane_res = verify_boundary_displacement(
        rippled_plane(), "normal", attachment=0.3)
    curved_report, curved_res = verify_boundary_displacement(
        rippled_curved(), "jerk", attachment=0.2)
    export_obj(plane_res.faired, out / "plane_faired.obj")
    export_obj(curved_res.faired, out / "curved_faired.obj")

    payload = {
        "boundary_checks": [plane_report, curved_report],
        "maintenance": {"prune": prune_case(), "relax": relax_case()},
    }
    (out / "maintenance_report.json").write_text(
        json.dumps(payload, indent=2, default=_json_default))

    ok = all(r["passes"] for r in payload["boundary_checks"])
    for r in payload["boundary_checks"]:
        print(f"{r['artifact']} [{r['bend_mode']}] boundary checks "
              f"{'PASS' if r['passes'] else 'FAIL'}: "
              f"max edge {max(r['edge_displacements'].values()):.1e}, "
              f"interior moved {r['interior_displacement']:.5f}, "
              f"energy {r['energy_reduction_ratio']:.1%}")
    pr = payload["maintenance"]["prune"]
    print(f"prune: minimal conflict {pr['minimal_conflict']}; removed "
          f"{pr['removed']}; {pr['reactivation_condition']}")
    rx = payload["maintenance"]["relax"]
    b = rx["relaxation_boundary"]
    if b:
        print(f"relax: soft-lock deviations {[round(d, 6) for d in b['deviations']]} "
              f"-> min {b['min_deviation']:.2e} (feasible at "
              f"{b['tolerance']}: {b['feasible_at_tolerance']})")
    print(f"report: {out / 'maintenance_report.json'}")
    print("MAINTENANCE OK" if ok else "MAINTENANCE FAILURE")
    return 0 if ok else 1


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="output/maintenance")
    args = parser.parse_args()
    sys.exit(main(args.outdir))
