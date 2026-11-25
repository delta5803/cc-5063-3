"""Fairing-contract validation in a clean directory.

A *contract check* here means: the documentation's claims about the fairing
kernel are re-derived from the live code and compared against hard, executable
evidence, rather than trusted. It runs end-to-end without any hand-edited data:

* every claim is computed by calling the public fairing API on the shipped
  artifacts; nothing is read from an existing ``output/`` or ``scratch/`` file;
* energy change (``direct_energy`` before/after) and boundary residual
  (max edge/ring movement) are measured on the real solved surface;
* a canonical JSON is written into a fresh directory (default a new
  ``tempfile`` dir, so the run never depends on stale artifacts) and a
  ``.manifest`` SHA-256 hashes the code modules it validated, so a later
  stale-doc check can detect drift.

The script exits non-zero with a concrete list of failed claims if the
documentation and the implementation disagree. It is the entry point the
surface-fairing sample is validated through.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/validate_contract.py
    PYTHONPATH=. .venv/bin/python scripts/validate_contract.py --outdir /tmp/x
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from surface_fairing.core.bspline import greville_abscissae
from surface_fairing.core.constraints import HardConstraints, make_ring_soft_lock
from surface_fairing.core.curvature import curvature_field
from surface_fairing.core.energy import BEND_MODES, direct_energy
from surface_fairing.core.fairing import fair_surface
from surface_fairing.io_.artifacts import (
    over_fairing_case, rippled_curved, rippled_plane)

EDGES = ("u0", "u1", "v0", "v1")


@dataclass
class Claim:
    name: str
    description: str
    measured: dict
    passes: bool
    tolerance: dict = field(default_factory=dict)


def _edge_locked(surf):
    hard = HardConstraints(3 * surf.n_control)
    for edge in EDGES:
        hard.pin_edge(surf, edge)
    return hard


def _max_edge_displacement(a, b, n=64):
    return max(
        float(np.linalg.norm(a.edge_samples(e, n) - b.edge_samples(e, n),
                             axis=1).max()) for e in EDGES)


def _hf_rms(surf, grid=48, cutoff=3):
    ts = np.linspace(0.0, 1.0, grid)
    Z = np.array([[surf.evaluate(u, v)[2] for v in ts] for u in ts])
    F = np.fft.fft2(Z - Z.mean())
    k = np.minimum(np.arange(grid), grid - np.arange(grid))
    KU, KV = np.meshgrid(k, k, indexing="ij")
    mask = (np.abs(KU) >= cutoff) | (np.abs(KV) >= cutoff)
    return float(np.sqrt(np.mean(np.real(np.fft.ifft2(F * mask)) ** 2)))


def _smooth_dome(art):
    from surface_fairing.core.bspline import BSplineSurface
    surf = art.surface
    gu = greville_abscissae(surf.nu, surf.degree_u)
    gv = greville_abscissae(surf.nv, surf.degree_v)
    U, V = np.meshgrid(gu, gv, indexing="ij")
    z = art.expected["base_curvature"] * np.sin(np.pi * U) * np.sin(np.pi * V)
    return BSplineSurface(np.stack([U, V, z], axis=-1))


def _rms_to(a, b, grid=64):
    ts = np.linspace(0.0, 1.0, grid)
    d = [np.linalg.norm(a.evaluate(u, v) - b.evaluate(u, v))
         for u in ts for v in ts]
    return float(np.sqrt(np.mean(np.square(d))))

# ------------------------------------------------------------------ claims
def check_locked_boundary_exact():
    art = rippled_plane()
    s = art.surface
    res = fair_surface(s, _edge_locked(s), bend_mode="normal",
                       attachment_weight=0.3)
    disp = _max_edge_displacement(s, res.faired)
    return Claim(
        name="locked_boundary_exact",
        description="hard edge locks keep every boundary sample fixed to "
                    "machine precision",
        measured={"hard_constraint_error": res.hard_constraint_error,
                  "max_edge_displacement": disp},
        passes=bool(res.hard_constraint_error < 1e-9 and disp < 1e-9),
        tolerance={"hard_constraint_error": 1e-9, "edge_displacement": 1e-9})


def check_energy_decreases():
    """Energy change on the real solved surface (direct re-integration)."""
    rows = {}
    ok = True
    for name, art, mode, att in (
            ("plane", rippled_plane(), "normal", 0.3),
            ("curved", rippled_curved(), "jerk", 0.2)):
        s = art.surface
        res = fair_surface(s, _edge_locked(s), bend_mode=mode,
                           attachment_weight=att)
        rows[name] = {
            "bend_mode": mode,
            "direct_before": res.direct_before,
            "direct_after": res.direct_after,
            "reduction_ratio": res.energy_reduction_ratio,
            # independent re-integration of the same statement
            "reintegr_before": direct_energy(s, bend_mode=mode),
            "reintegr_after": direct_energy(res.faired, bend_mode=mode),
        }
        ok = ok and res.direct_after < res.direct_before
        ok = ok and abs(rows[name]["reintegr_before"] - res.direct_before) < 1e-9
        ok = ok and abs(rows[name]["reintegr_after"] - res.direct_after) < 1e-9
    return Claim(
        name="energy_decreases", description="direct re-integrated energy "
        "strictly decreases and matches the reported values",
        measured=rows, passes=ok)


def check_doc_curve_reproducible():
    """Recompute documented ripple-removal anchors from actual solves.

    The doc quotes a curve of high-frequency ripple removal versus grid
    resolution (normal mode, locked plane), generated by
    ``scripts/generate_doc_curves.py`` into ``docs/curves.json``. This claim
    re-solves two anchor points and checks they match that file, so the
    documented numbers can only come from a real solve and cannot be
    hand-edited: if the code drifts, the anchors disagree and the claim fails.
    """
    from surface_fairing.core.bspline import BSplineSurface
    anchors = [(9, 3), (17, 3)]
    curve_path = (Path(__file__).resolve().parent.parent
                  / "docs" / "curves.json")
    stored = None
    if curve_path.exists():
        stored = {(row["grid"], row["waves"]): row
                  for row in json.loads(curve_path.read_text())["plane_curve"]}
    rows = []
    ok = True
    for n, waves in anchors:
        g = greville_abscissae(n, 3)
        U, V = np.meshgrid(g, g, indexing="ij")
        ripple = 0.04 * np.sin(2 * np.pi * waves * U) * np.sin(
            2 * np.pi * waves * V)
        ripple[0, :] = ripple[-1, :] = ripple[:, 0] = ripple[:, -1] = 0.0
        s = BSplineSurface(np.stack([U, V, ripple], axis=-1),
                           name=f"anchor_n{n}_w{waves}")
        res = fair_surface(s, _edge_locked(s), bend_mode="normal",
                           attachment_weight=0.3)
        hf0, hf1 = _hf_rms(s), _hf_rms(res.faired)
        ratio = float(1 - hf1 / hf0)
        row = {
            "grid": n, "waves": waves,
            "hf_removal_ratio": ratio,
            "max_edge_displacement": _max_edge_displacement(s, res.faired),
            "energy_reduction_ratio": res.energy_reduction_ratio,
        }
        rows.append(row)
        ok = ok and row["max_edge_displacement"] < 1e-9
        ok = ok and row["energy_reduction_ratio"] > 0.0
        if stored is not None:
            ref = stored.get((n, waves))
            ok = ok and ref is not None and \
                abs(ref["hf_removal_ratio"] - ratio) < 1e-6
    return Claim(
        name="doc_curve_reproducible",
        description="documented ripple-removal curve anchors re-solve and "
                    "match docs/curves.json, with exact locked boundary",
        measured={"anchors": rows, "compared_to": str(curve_path),
                  "curve_file_present": stored is not None},
        passes=ok)


def check_curved_patch_retains_dome():
    """Lower energy must not be conflated with keeping all design features:
    the smooth dome height is preserved while against the known base the
    surface is no farther than the rippled input."""
    art = rippled_curved()
    s = art.surface
    base = _smooth_dome(art)
    res = fair_surface(s, _edge_locked(s), bend_mode="jerk",
                       attachment_weight=0.2)
    z_center = float(res.faired.evaluate(0.5, 0.5)[2])
    z_base = float(art.expected["base_dome_z_at_center"])
    rms_before = _rms_to(s, base)
    rms_after = _rms_to(res.faired, base)
    ok = (abs(z_center - z_base) < 0.02
          and rms_after <= rms_before + 1e-3
          and res.direct_after < res.direct_before)
    return Claim(
        name="curved_patch_retains_dome",
        description="curvature-variation fairing keeps the smooth dome and "
                    "reduces ripple instead of flattening",
        measured={"center_z": z_center, "base_center_z": z_base,
                  "rms_to_base_before": rms_before,
                  "rms_to_base_after": rms_after,
                  "energy_reduction_ratio": res.energy_reduction_ratio,
                  "hard_constraint_error": res.hard_constraint_error},
        passes=ok, tolerance={"center_z": 0.02, "rms_growth": 1e-3})


def check_over_fairing_is_a_failure_case():
    """A lower energy result that destroys the feature is still a failure: the
    key-point lock is what preserves it."""
    art = over_fairing_case()
    s = art.surface
    key = art.expected["key_height"]
    aggressive = fair_surface(s, _edge_locked(s), bend_mode="jerk",
                              attachment_weight=0.02)
    locked_hard = _edge_locked(s)
    locked_hard.pin_surface_point(s, 0.5, 0.5)
    locked = fair_surface(s, locked_hard, bend_mode="jerk",
                          attachment_weight=0.3)
    agg_key = float(aggressive.faired.evaluate(0.5, 0.5)[2])
    lock_key = float(locked.faired.evaluate(0.5, 0.5)[2])
    ok = (agg_key < key - 0.01                    # aggressive loses the bump
          and abs(lock_key - key) < 1e-6          # locked keeps it exactly
          and aggressive.direct_after < aggressive.direct_before
          and locked.direct_after < locked.direct_before
          and locked.hard_constraint_error < 1e-9)
    return Claim(
        name="over_fairing_is_failure_not_success",
        description="energy reduction alone does not preserve the design "
                    "feature; the key-point lock does",
        measured={"key_height": key,
                  "aggressive_key_height": agg_key,
                  "locked_key_height": lock_key,
                  "aggressive_energy_reduction":
                      aggressive.energy_reduction_ratio,
                  "locked_energy_reduction": locked.energy_reduction_ratio},
        passes=ok)


def check_quadratic_matches_direct():
    """The frozen quadratic form must equal the direct integration on the
    reference surface for every bend mode (inspectability of the energy)."""
    from surface_fairing.core.energy import assemble_energy
    art = rippled_curved()
    s = art.surface
    rows = {}
    ok = True
    for mode in BEND_MODES:
        model = assemble_energy(s, bend_mode=mode)
        quad = model.absolute_energy(s.flatten().reshape(-1))
        direct = direct_energy(s, bend_mode=mode)
        rel = abs(direct - quad) / max(direct, 1e-30)
        rows[mode] = {"quadratic": quad, "direct": direct, "rel_error": rel}
        ok = ok and rel < 1e-9
    return Claim(
        name="quadratic_matches_direct",
        description="frozen-area quadratic energy equals direct integration "
                    "on the reference surface for all bend modes",
        measured=rows, passes=ok, tolerance={"rel_error": 1e-9})


def check_soft_relaxation_is_reported():
    """A soft lock either holds or its relaxation is recorded with the ring
    indices and maximum deviation; it never breaks the hard residual."""
    art = rippled_curved()
    s = art.surface
    soft = [make_ring_soft_lock(s, "u0", weight=1000.0)]
    res = fair_surface(s, _edge_locked(s), soft=soft, attachment_weight=0.5)
    if res.soft_events:
        ev = res.soft_events[0]
        ok = ("ring_indices" in ev and "max_deviation" in ev
              and ev["max_deviation"] >= 0.0)
    else:
        ok = True
    ok = ok and res.hard_constraint_error < 1e-8
    return Claim(
        name="soft_relaxation_reported",
        description="soft locks are relaxed openly and never corrupt hard "
                    "feasibility",
        measured={"events": res.soft_events,
                  "hard_constraint_error": res.hard_constraint_error},
        passes=ok)


CLAIM_BUILDERS = (
    check_locked_boundary_exact,
    check_energy_decreases,
    check_doc_curve_reproducible,
    check_curved_patch_retains_dome,
    check_over_fairing_is_a_failure_case,
    check_quadratic_matches_direct,
    check_soft_relaxation_is_reported,
)


def code_manifest() -> dict:
    """SHA-256 of the modules the contract depends on, for stale-doc checks."""
    root = Path(__file__).resolve().parent.parent
    out = {}
    for rel in ("core/bspline.py", "core/energy.py", "core/constraints.py",
                "core/solver.py", "core/fairing.py", "core/curvature.py",
                "io_/artifacts.py", "cli.py"):
        p = root / "surface_fairing" / rel
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    for rel in ("scripts/generate_doc_curves.py",):
        p = root / rel
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def run(outdir: str | Path | None = None) -> int:
    if outdir is None:
        outdir = Path(tempfile.mkdtemp(prefix="fairing_contract_"))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    claims = [builder() for builder in CLAIM_BUILDERS]
    payload = {
        "clean_dir": str(outdir),
        "claims": [{
            "name": c.name, "description": c.description,
            "passes": c.passes, "measured": c.measured,
            "tolerance": c.tolerance,
        } for c in claims],
        "code_manifest_sha256": code_manifest(),
    }
    report_path = outdir / "contract_report.json"
    report_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    (outdir / "contract_report.sha256").write_text(
        hashlib.sha256(report_path.read_bytes()).hexdigest() + "\n")

    failures = [c.name for c in claims if not c.passes]
    print(f"contract report: {report_path}")
    for c in claims:
        print(f"  [{'PASS' if c.passes else 'FAIL'}] {c.name}: {c.description}")
    if failures:
        print(f"CONTRACT VIOLATION: {len(failures)} failed claim(s): "
              f"{', '.join(failures)}")
        return 1
    print(f"CONTRACT OK: {len(claims)} claims verified in a clean directory")
    return 0


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=None,
                        help="output directory (default: a fresh temp dir)")
    args = parser.parse_args()
    sys.exit(run(args.outdir))
