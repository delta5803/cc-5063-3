"""Real surface output: OBJ mesh export and inspectable JSON reports."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.bspline import BSplineSurface


def export_obj(surf: BSplineSurface, path: str | Path,
               nu: int = 40, nv: int = 40) -> Path:
    """Write the evaluated surface as a triangulated OBJ mesh."""
    _, _, pts = surf.sample_grid(nu, nv)
    rows, cols, _ = pts.shape
    path = Path(path)
    with path.open("w") as fh:
        fh.write(f"# {surf.name} evaluated B-spline surface\n")
        for p in pts.reshape(-1, 3):
            fh.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        for i in range(rows - 1):
            for j in range(cols - 1):
                a = i * cols + j + 1
                b = (i + 1) * cols + j + 1
                c = a + 1
                d = b + 1
                fh.write(f"f {a} {b} {c}\nf {c} {b} {d}\n")
    return path


def export_control_points(surf: BSplineSurface, path: str | Path) -> Path:
    path = Path(path)
    np.savetxt(path, surf.flatten(), header="x y z per control point",
               comments="# ")
    return path


def write_report(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    return path


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)
