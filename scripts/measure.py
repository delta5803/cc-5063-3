"""Quantitative measurement harness.

Runs the real fairing on the plane and curved artifacts, measures constraint
error, shape displacement and curvature distributions, and explicitly keeps
the failure / over-fairing cases with their outputs. Every number is written
to JSON and compared against the mean-filter benchmark so the optimizer is
never conflated with control-point averaging.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from surface_fairing.core.benchmark import mean_filter
from surface_fairing.core.bspline import BSplineSurface, greville_abscissae
from surface_fairing.core.constraints import HardConstraints, make_ring_soft_lock
from surface_fairing.core.curvature import curvature_change, curvature_field
from surface_fairing.core.energy import BEND_MODES, direct_energy
from surface_fairing.core.fairing import fair_surface
from surface_fairing.io_.artifacts import (
    conflicting_constraint_case, over_fairing_case, rippled_curved,
    rippled_plane)
from surface_fairing.io_.export import export_obj, write_report
from surface_fairing.io_.stitch import build_stitched_mesh


def _edge_locked(surf):
    hard = HardConstraints(3 * surf.n_control)
    for edge in ("u0", "u1", "v0", "v1"):
        hard.pin_edge(surf, edge)
    return hard


def _curvature_stats(surf):
    _, _, f = curvature_field(surf, 33, 33)
    return {
        "mean_abs_max": float(np.abs(f["mean"]).max()),
        "mean_rms": float(np.sqrt(np.mean(f["mean"] ** 2))),
        "gaussian_abs_max": float(np.abs(f["gaussian"]).max()),
        "gaussian_rms": float(np.sqrt(np.mean(f["gaussian"] ** 2))),
        "principal_abs_max": float(
            max(np.abs(f["k1"]).max(), np.abs(f["k2"]).max())),
    }


def _rms_distance(a, b, grid=40):
    ts = np.linspace(0.0, 1.0, grid)
    d = [np.linalg.norm(a.evaluate(u, v) - b.evaluate(u, v))
         for u in ts for v in ts]
    return float(np.sqrt(np.mean(np.square(d))))


def _high_frequency_rms(surf, grid=48, cutoff=3):
    """RMS of the spectral high-frequency part of the sampled z-field, i.e.
    the ripple content rather than the low-frequency base shape."""
    ts = np.linspace(0.0, 1.0, grid)
    Z = np.array([[surf.evaluate(u, v)[2] for v in ts] for u in ts])
    F = np.fft.fft2(Z - Z.mean())
    k = np.minimum(np.arange(grid), grid - np.arange(grid))
    KU, KV = np.meshgrid(k, k, indexing="ij")
    mask = (np.abs(KU) >= cutoff) | (np.abs(KV) >= cutoff)
    return float(np.sqrt(np.mean(np.real(np.fft.ifft2(F * mask)) ** 2)))


def _smooth_dome(art):
    """The ripple-free dome with the artifact's grid and design curvature."""
    surf = art.surface
    gu = greville_abscissae(surf.nu, surf.degree_u)
    gv = greville_abscissae(surf.nv, surf.degree_v)
    U, V = np.meshgrid(gu, gv, indexing="ij")
    curv = art.expected["base_curvature"]
    z = curv * np.sin(np.pi * U) * np.sin(np.pi * V)
    return BSplineSurface(np.stack([U, V, z], axis=-1),
                          name=art.name + "_smooth_base")


def measure_case(artifact, attachment: float | None = None,
                 membrane: float = 0.0, soft: bool = False,
                 bend_mode: str | None = None):
    surf = artifact.surface
    # Default objective follows the geometry: normal-projected bending removes
    # near-planar ripple almost completely and cannot flatten a plane (its
    # target curvature is already zero); curvature-variation (jerk) is the
    # shape-safe choice on a curved patch because it does not drive the smooth
    # dome curvature toward zero. Either can be overridden explicitly.
    if bend_mode is None:
        bend_mode = "normal" if artifact.kind == "plane" else "jerk"
    if attachment is None:
        attachment = 0.3 if artifact.kind == "plane" else 0.2
    hard = _edge_locked(surf)
    soft_locks = [make_ring_soft_lock(surf, "u0", 100.0)] if soft else []
    res = fair_surface(surf, hard, soft=soft_locks,
                       bend_mode=bend_mode,
                       memb_weight=membrane,
                       attachment_weight=attachment,
                       tangent_edges=("u0", "u1", "v0", "v1"))
    change = curvature_change(surf, res.faired, 33, 33)
    filtered = mean_filter(surf, iterations=3)
    out = {
        "name": artifact.name,
        "kind": artifact.kind,
        "bend_mode": bend_mode,
        "known_ripple_amplitude": artifact.ripple_amplitude,
        "energy": {
            "before": res.direct_before,
            "after": res.direct_after,
            "reduction_ratio": res.energy_reduction_ratio,
            "mean_filter_after": direct_energy(
                filtered, bend_mode=bend_mode),
        },
        "constraint_error_max": res.hard_constraint_error,
        "shape_displacement_max": res.position_residual,
        "curvature_before": _curvature_stats(surf),
        "curvature_after": _curvature_stats(res.faired),
        "curvature_mean_change_max": float(change["mean_abs_delta"].max()),
        "high_frequency_rms_before": _high_frequency_rms(surf),
        "high_frequency_rms_after": _high_frequency_rms(res.faired),
        "soft_relaxation_events": res.soft_events,
        "solver_message": res.report.message,
        "faired": res.faired,
    }
    # curved patches carry a known smooth base: quantify shape retention
    # against the ground truth instead of treating "flatter" as success.
    if artifact.kind == "curved":
        base = _smooth_dome(artifact)
        out["base_dome_retention"] = {
            "rms_to_smooth_base_before": _rms_distance(surf, base),
            "rms_to_smooth_base_after": _rms_distance(res.faired, base),
            "center_z_base": float(base.evaluate(0.5, 0.5)[2]),
            "center_z_after": float(res.faired.evaluate(0.5, 0.5)[2]),
        }
    return out


def measure_mode_comparison(artifact, attachment: float, membrane: float = 0.0):
    """Run every bend mode on the same artifact so their trade-offs are
    inspectable: curvature-variation (jerk) vs the two bending proxies."""
    rows = []
    for mode in BEND_MODES:
        r = measure_case(artifact, attachment, membrane, bend_mode=mode)
        rows.append({
            "bend_mode": mode,
            "energy_before": r["energy"]["before"],
            "energy_after": r["energy"]["after"],
            "hf_rms_before": r["high_frequency_rms_before"],
            "hf_rms_after": r["high_frequency_rms_after"],
            "shape_displacement_max": r["shape_displacement_max"],
            "mean_curvature_rms_before": r["curvature_before"]["mean_rms"],
            "mean_curvature_rms_after": r["curvature_after"]["mean_rms"],
        })
        if "base_dome_retention" in r:
            rows[-1]["rms_to_smooth_base_after"] = \
                r["base_dome_retention"]["rms_to_smooth_base_after"]
    return rows


def measure_over_fairing():
    """Retained over-fairing case: the feature shrinks without a key lock."""
    art = over_fairing_case()
    surf = art.surface
    hard = _edge_locked(surf)
    # With the normalized stiffness, a small attachment weight lets the
    # curvature-variation objective move freely and shrink the un-locked bump.
    aggressive = fair_surface(surf, hard, bend_mode="jerk",
                              attachment_weight=0.02)
    locked = HardConstraints(3 * surf.n_control)
    for edge in ("u0", "u1", "v0", "v1"):
        locked.pin_edge(surf, edge)
    locked.pin_surface_point(surf, 0.5, 0.5)
    preserved = fair_surface(surf, locked, bend_mode="jerk",
                             attachment_weight=0.3)
    return {
        "name": "over_fairing_feature_bump",
        "retained_as_failure_case": True,
        "original_key_height": art.expected["key_height"],
        "aggressive_no_lock_key_height":
            float(aggressive.faired.evaluate(0.5, 0.5)[2]),
        "locked_key_height":
            float(preserved.faired.evaluate(0.5, 0.5)[2]),
        "aggressive_displacement": aggressive.position_residual,
        "locked_displacement": preserved.position_residual,
        "aggressive": aggressive.faired,
        "locked": preserved.faired,
    }


def measure_conflict():
    """Retained failure case: mutually exclusive fixed edges across a seam."""
    a, b = conflicting_constraint_case()
    mesh = build_stitched_mesh(a, "u0", b, "u0", n_along=24, n_across=8)
    return {
        "name": "mutually_exclusive_seam",
        "retained_as_failure_case": True,
        "seam_gap": mesh.seam_gap,
        "contour_error_a": mesh.contour_error_a,
        "contour_error_b": mesh.contour_error_b,
        "note": "fixed edges disagree; concession must be authorized explicitly",
    }


def main(outdir: str = "output/measurements") -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    plane_art = rippled_plane()
    curved_art = rippled_curved()
    payload = {
        "plane": measure_case(plane_art),
        "curved": measure_case(curved_art, soft=True),
        "bend_mode_comparison": {
            "plane": measure_mode_comparison(plane_art, attachment=0.3),
            "curved": measure_mode_comparison(curved_art, attachment=0.2),
        },
        "over_fairing": measure_over_fairing(),
        "conflict": measure_conflict(),
    }
    # export real surfaces for every case
    for key in ("plane", "curved"):
        export_obj(payload[key]["faired"], out / f"{key}_faired.obj")
    export_obj(payload["over_fairing"]["aggressive"],
               out / "over_fairing_aggressive.obj")
    export_obj(payload["over_fairing"]["locked"],
               out / "over_fairing_feature_locked.obj")

    serializable = {k: v for k, v in payload.items()}
    write_report(serializable, out / "measurements.json")
    return payload


if __name__ == "__main__":
    data = main()
    for key, item in data.items():
        if isinstance(item, dict) and "energy" in item:
            e = item["energy"]
            print(f"{key} [{item['bend_mode']}]: energy "
                  f"{e['before']:.5f} -> {e['after']:.5f} "
                  f"({item['energy']['reduction_ratio']:.1%}), "
                  f"constraint err {item['constraint_error_max']:.2e}, "
                  f"displacement {item['shape_displacement_max']:.5f}, "
                  f"HF rms {item['high_frequency_rms_before']:.5f} -> "
                  f"{item['high_frequency_rms_after']:.5f}")
            if "base_dome_retention" in item:
                r = item["base_dome_retention"]
                print(f"    dome retention: rms-to-base "
                      f"{r['rms_to_smooth_base_before']:.5f} -> "
                      f"{r['rms_to_smooth_base_after']:.5f}, "
                      f"center z {r['center_z_after']:.4f} "
                      f"(base {r['center_z_base']:.4f})")
        elif key == "bend_mode_comparison":
            for art, rows in item.items():
                print(f"mode comparison ({art}):")
                for row in rows:
                    print("    %-10s E %.4g->%.4g HF %.5f->%.5f H_rms %.4f->%.4f"
                          % (row["bend_mode"], row["energy_before"],
                             row["energy_after"], row["hf_rms_before"],
                             row["hf_rms_after"],
                             row["mean_curvature_rms_before"],
                             row["mean_curvature_rms_after"]))
        else:
            print(key, "retained case:",
                  {k: v for k, v in item.items()
                   if isinstance(v, (int, float, str, bool))})
