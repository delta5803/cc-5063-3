"""Mean-filter baseline, kept ONLY as a comparison benchmark.

This is deliberately separated from the fairing pipeline. Averaging control
points is not an energy optimization: it does not use the metric, has no
constraint algebra and cannot guarantee boundary or feature preservation.
Tests use it to show that the real fairer behaves differently and that the
benchmark alone cannot claim curvature optimization.
"""
from __future__ import annotations

import numpy as np

from ..core.bspline import BSplineSurface


def mean_filter(surf: BSplineSurface, iterations: int = 1,
                keep_edges: bool = True) -> BSplineSurface:
    out = surf.clone()
    for _ in range(iterations):
        P = out.P.copy()
        acc = np.zeros_like(P)
        count = np.zeros(P.shape[:2])
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                shifted = np.roll(np.roll(P, -di, axis=0), -dj, axis=1)
                acc += shifted
                count += 1
        smoothed = acc / count[..., None]
        if keep_edges:
            smoothed[0, :] = P[0, :]
            smoothed[-1, :] = P[-1, :]
            smoothed[:, 0] = P[:, 0]
            smoothed[:, -1] = P[:, -1]
        out.set_flat(smoothed.reshape(-1, 3))
    out.name = surf.name + "_meanfilter"
    return out
