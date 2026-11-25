"""Command-line helper for batch fairing runs.

The graphical surface editing window is the primary interface; this CLI only
supports scripted/batch fairing and report generation. It never replaces the
interactive constraint setup and curvature/reflection inspection available in
the GUI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .core.benchmark import mean_filter
from .core.constraints import HardConstraints, make_ring_soft_lock
from .core.curvature import curvature_change, curvature_field
from .core.energy import direct_energy
from .core.fairing import fair_surface
from .io_.artifacts import rippled_curved, rippled_plane
from .io_.export import export_control_points, export_obj, write_report


def _build_artifact(name: str, metric_u: float, metric_v: float):
    from .core.bspline import Metric
    metric = Metric(metric_u, metric_v)
    if name == "plane":
        return rippled_plane(metric=metric)
    if name == "curved":
        return rippled_curved(metric=metric)
    raise ValueError(f"unknown artifact {name!r}; use 'plane' or 'curved'")


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="surface-fair",
        description="Batch helper for parametric surface fairing. The GUI "
                    "is the primary interface; see docs for its entry point.")
    parser.add_argument("--artifact", choices=["plane", "curved"],
                        default="plane")
    parser.add_argument("--attachment", type=float, default=None)
    parser.add_argument("--membrane", type=float, default=0.0)
    parser.add_argument("--bend", type=float, default=1.0)
    parser.add_argument("--bend-mode", choices=["jerk", "normal", "thin_plate"],
                        default=None,
                        help="jerk = curvature-variation (shape-safe on curved "
                             "patches); normal = normal-projected bending "
                             "(best for near-planar ripple); thin_plate = "
                             "comparison proxy. Defaults: plane->normal, "
                             "curved->jerk.")
    parser.add_argument("--soft-ring", action="store_true")
    parser.add_argument("--metric-u", type=float, default=1.0)
    parser.add_argument("--metric-v", type=float, default=1.0)
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    artifact = _build_artifact(args.artifact, args.metric_u, args.metric_v)
    surf = artifact.surface
    hard = HardConstraints(3 * surf.n_control)
    for edge in ("u0", "u1", "v0", "v1"):
        hard.pin_edge(surf, edge)
    soft = [make_ring_soft_lock(surf, "u0", 100.0)] if args.soft_ring else []

    bend_mode = args.bend_mode or (
        "normal" if args.artifact == "plane" else "jerk")
    attachment = args.attachment if args.attachment is not None else (
        0.3 if args.artifact == "plane" else 0.2)

    result = fair_surface(
        surf, hard, soft=soft, bend_weight=args.bend,
        bend_mode=bend_mode,
        memb_weight=args.membrane, attachment_weight=attachment,
        tangent_edges=("u0", "v1"))

    _, _, before = curvature_field(surf)
    _, _, after = curvature_field(result.faired)
    change = curvature_change(surf, result.faired)
    filtered = mean_filter(surf, iterations=3)

    obj_path = export_obj(result.faired, out / f"{args.artifact}_faired.obj")
    export_obj(surf, out / f"{args.artifact}_input.obj")
    export_control_points(result.faired,
                          out / f"{args.artifact}_control_points.txt")

    report = {
        "artifact": args.artifact,
        "metric": result.metric_label,
        "bend_mode": bend_mode,
        "attachment_weight": attachment,
        "energy_normalization_scale":
            result.energy_model.energy_scale,
        "n_control_points": surf.n_control,
        "gauss_integration_points": result.energy_model.n_integration_points,
        "energy": {
            "before_direct": result.direct_before,
            "after_direct": result.direct_after,
            "reduction_ratio": result.energy_reduction_ratio,
            "mean_filter_energy_for_comparison": direct_energy(filtered),
        },
        "errors": {
            "hard_constraint_max": result.hard_constraint_error,
            "max_surface_position_residual": result.position_residual,
        },
        "curvature": {
            "mean_before_max_abs": float(np.abs(before["mean"]).max()),
            "mean_after_max_abs": float(np.abs(after["mean"]).max()),
            "mean_change_max_abs": float(change["mean_abs_delta"].max()),
            "gaussian_change_max_abs":
                float(change["gaussian_abs_delta"].max()),
        },
        "solver": {
            "message": result.report.message,
            "ridge_used": result.report.ridge_used,
            "condition_number": result.report.condition_number,
            "pinned_dofs": result.report.pinned_dofs,
        },
        "soft_relaxation": result.soft_events,
        "output_obj": str(obj_path),
    }
    report_path = write_report(report, out / f"{args.artifact}_report.json")
    print(f"fairing complete: {obj_path}")
    print(f"report: {report_path}")
    print(f"energy {result.direct_before:.6f} -> {result.direct_after:.6f} "
          f"({result.energy_reduction_ratio:.1%} reduction)")
    print(f"hard constraint error {result.hard_constraint_error:.2e}, "
          f"position residual {result.position_residual:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
