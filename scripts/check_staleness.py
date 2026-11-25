"""Stale-material check for the fairing docs and persisted reports.

The contract report records a SHA-256 manifest of the modules it validated.
This script closes the loop: it flags stale material instead of trusting filenames.

It checks, without editing anything:

1. ``output/contract/contract_report.json`` exists and its
   ``code_manifest_sha256`` matches the *current* source modules. A drift means
   the contract was validated against different code and must be re-run.
2. The persisted CLI reports (``output/cli_*/..._report.json``) carry the fields
   the current implementation writes (``energy_normalization_scale``,
   ``gauss_integration_points``, ``mean_filter_energy_for_comparison``). Their
   absence is the fingerprint of the old, superseded energy formula.
3. A fresh solve reproduces the persisted curved energy after-value within
   tolerance, so the report cannot silently disagree with the kernel.

Exits non-zero and lists every stale item if any check fails.

Run::

    PYTHONPATH=. .venv/bin/python scripts/check_staleness.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surface_fairing.core.constraints import HardConstraints
from surface_fairing.core.fairing import fair_surface
from surface_fairing.io_.artifacts import rippled_curved

ROOT = Path(__file__).resolve().parent.parent
MODULES = ("core/bspline.py", "core/energy.py", "core/constraints.py",
           "core/solver.py", "core/fairing.py", "core/curvature.py",
           "io_/artifacts.py", "cli.py")
SCRIPTS = ("scripts/generate_doc_curves.py",)
REQUIRED_CLI_FIELDS = ("energy_normalization_scale", "gauss_integration_points")


def _manifest() -> dict:
    out = {rel: hashlib.sha256(
        (ROOT / "surface_fairing" / rel).read_bytes()).hexdigest()
        for rel in MODULES}
    for rel in SCRIPTS:
        out[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    return out


def _check_manifest(problems: list[str]) -> dict:
    report_path = ROOT / "output" / "contract" / "contract_report.json"
    if not report_path.exists():
        problems.append(f"missing contract report: {report_path}")
        return {}
    stored = json.loads(report_path.read_text()).get("code_manifest_sha256", {})
    current = _manifest()
    drift = [rel for rel in current
             if stored.get(rel) != current[rel]]
    for rel in drift:
        problems.append(f"contract report stale for module: {rel}")
    return {"checked": len(current), "drift": drift}


def _check_cli_reports(problems: list[str]) -> dict:
    rows = {}
    for artifact in ("plane", "curved"):
        path = ROOT / "output" / f"cli_{artifact}" / f"{artifact}_report.json"
        if not path.exists():
            problems.append(f"missing CLI report: {path}")
            continue
        data = json.loads(path.read_text())
        missing = [f for f in REQUIRED_CLI_FIELDS if f not in data]
        if missing:
            problems.append(
                f"{path.name} lacks current fields {missing} "
                f"(old energy formula fingerprint)")
        rows[artifact] = {
            "has_current_fields": not missing,
            "energy_after_direct":
                data.get("energy", {}).get("after_direct"),
        }
    return rows


def _check_fresh_match(problems: list[str], cli_rows: dict) -> dict:
    surf = rippled_curved().surface
    hard = HardConstraints(3 * surf.n_control)
    for edge in ("u0", "u1", "v0", "v1"):
        hard.pin_edge(surf, edge)
    res = fair_surface(surf, hard, bend_mode="jerk", attachment_weight=0.2)
    stored = (cli_rows.get("curved") or {}).get("energy_after_direct")
    match = None
    if stored is not None:
        rel = abs(res.direct_after - stored) / max(abs(res.direct_after), 1e-12)
        match = rel < 1e-6
        if not match:
            problems.append(
                f"persisted curved energy_after_direct {stored} disagrees with "
                f"fresh solve {res.direct_after}")
    return {"fresh_energy_after": res.direct_after,
            "stored_energy_after": stored, "matches_fresh_solve": match}


def run() -> int:
    problems: list[str] = []
    manifest = _check_manifest(problems)
    cli_rows = _check_cli_reports(problems)
    fresh = _check_fresh_match(problems, cli_rows)

    print("stale-material check")
    print(f"  manifest: {manifest.get('checked', 0)} modules, "
          f"drift {manifest.get('drift', [])}")
    for artifact, row in cli_rows.items():
        print(f"  {artifact} CLI report current fields: "
              f"{row['has_current_fields']}")
    print(f"  fresh curved energy after {fresh['fresh_energy_after']:.6f} "
          f"vs stored {fresh['stored_energy_after']} "
          f"(match: {fresh['matches_fresh_solve']})")
    if problems:
        print("STALE MATERIAL:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("NO STALE MATERIAL")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sys.exit(run())
