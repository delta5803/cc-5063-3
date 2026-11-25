"""Inspectable bending / curvature-variation energy assembly by Gauss
integration over every non-empty knot span.

Three energies are integrated numerically over the parameter domain [0, 1]^2:

* thin-plate bending energy (``bend_mode='thin_plate'``)
      E_bend = 1/2 * int ( |S_uu/a^2|^2 + 2|S_uv/(ab)|^2 + |S_vv/b^2|^2 ) dA
* normal-projected bending energy (``bend_mode='normal'``), the second
  fundamental form, which replaces each raw second derivative by its
  projection on the analytic reference normal;
* curvature-variation / jerk energy (``bend_mode='jerk'``, the default),
      E_jerk = 1/2 * int ( |S_uuu/a^3|^2 + 3|S_uuv/(a^2b)|^2
                           + 3|S_uvv/(ab^2)|^2 + |S_vvv/b^3|^2 ) dA

where a, b are the constant declared metric scales and dA is the area element
of the *reference* surface (|S_u x S_v| du dv). Because every surface
derivative is linear in the control points, freezing the reference area element
makes each energy an exact quadratic form

    E(x) = 1/2 x^T H x - x^T f + const,

with f = H x_ref and const = 1/2 x_ref^T H x_ref, so E(x_ref) equals the
directly integrated reference energy. The integration points, weights and
matrix H can all be inspected; no control-point averaging and no vertex normal
smoothing occur anywhere.

Why jerk is the default. Second-derivative bending energies measure curvature
itself, so on a curved patch they drive the legitimate base curvature (e.g. a
smooth dome) toward zero and flatten it; the frozen-reference normal projection
is only a small-displacement linearization and concentrates curvature in the
locked boundary rings. The third-derivative energy instead measures the
*variation* of curvature. In Fourier terms a mode of wave number k contributes
k^2 to bending but k^3 to jerk, so the low-frequency dome is barely penalized
while high-frequency ripples are strongly penalized. This preserves the
retained shape quantitatively while removing waviness. The two bending modes
remain available so the difference is inspectable rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bspline import BSplineSurface, basis_table

_GL = {
    2: (np.array([0.2113248654051871, 0.7886751345948129]),
        np.array([0.5, 0.5])),
    3: (np.array([0.1127016653792583, 0.5, 0.8872983346207417]),
        np.array([5 / 18, 8 / 18, 5 / 18])),
    4: (np.array([0.0694318413749223, 0.3300094782075719,
                  0.6699905217924281, 0.9305681586250777]),
        np.array([0.1739274225687269, 0.3260725774312731,
                  0.3260725774312731, 0.1739274225687269])),
}

BEND_MODES = ("normal", "thin_plate", "jerk")


@dataclass
class EnergyModel:
    """Quadratic energy of one patch.

    ``H`` is the integrated stiffness matrix, ``x_ref`` the flattened reference
    control points. Two energies are available:

    * :meth:`absolute_energy` returns 1/2 x'Hx, equal to :func:`direct_energy`
      evaluated with the reference area element;
    * :meth:`energy` returns the displacement form
      1/2 (x-x_ref)' H (x-x_ref), which is zero at the reference and is what
      the fairing optimizer minimizes subject to constraints.

    ``bend_mode`` selects the physical meaning of the leading term:

    * ``"jerk"``: third-derivative curvature-variation energy, which removes
      high-frequency waviness while retaining smooth base curvature;
    * ``"normal"``: second fundamental form (reference normal projection),
      reduced curvature in the small-displacement regime but flattening on
      curved patches;
    * ``"thin_plate"``: raw metric-scaled second derivatives, a common proxy
      that can disagree with measured curvature on curved patches.
    """

    H: np.ndarray
    f: np.ndarray
    const: float
    x_ref: np.ndarray
    ref: BSplineSurface
    gauss_order: int
    bend_weight: float
    memb_weight: float
    bend_mode: str
    n_integration_points: int
    metric_label: str
    linearized: bool
    energy_scale: float = 1.0

    def energy(self, control_flat: np.ndarray) -> float:
        dx = np.asarray(control_flat, dtype=float).reshape(-1) - self.x_ref
        return float(0.5 * dx @ self.H @ dx) * self.energy_scale

    def absolute_energy(self, control_flat: np.ndarray) -> float:
        x = np.asarray(control_flat, dtype=float).reshape(-1)
        return float(0.5 * x @ self.H @ x) * self.energy_scale


def _spans(knots: np.ndarray) -> list[tuple[float, float]]:
    """Non-empty intervals of a (possibly repeated-knot) knot vector."""
    out = []
    for i in range(len(knots) - 1):
        if knots[i + 1] - knots[i] > 1e-15:
            out.append((float(knots[i]), float(knots[i + 1])))
    return out


def _gauss_nodes(knots: np.ndarray, order: int):
    """Gauss nodes and parameter-measure weights over every knot span.

    Each weight already includes the span length, so summing
    ``weight * integrand`` integrates over the whole [0, 1] parameter domain
    without missing or double-counting a span.
    """
    base_nodes, base_weights = _GL[order]
    pts, wts = [], []
    for lo, hi in _spans(knots):
        length = hi - lo
        for n, w in zip(base_nodes, base_weights):
            pts.append(lo + length * n)
            wts.append(w * length)
    return np.asarray(pts), np.asarray(wts)


def _mixed_basis(surf: BSplineSurface, u: float, v: float):
    Nu, dNu, ddNu, dddNu = basis_table(u, surf.knots_u, surf.degree_u)
    Nv, dNv, ddNv, dddNv = basis_table(v, surf.knots_v, surf.degree_v)
    return (
        np.einsum("i,j->ij", Nu, Nv).reshape(-1),
        np.einsum("i,j->ij", dNu, Nv).reshape(-1),
        np.einsum("i,j->ij", Nu, dNv).reshape(-1),
        np.einsum("i,j->ij", ddNu, Nv).reshape(-1),
        np.einsum("i,j->ij", dNu, dNv).reshape(-1),
        np.einsum("i,j->ij", Nu, ddNv).reshape(-1),
        np.einsum("i,j->ij", dddNu, Nv).reshape(-1),
        np.einsum("i,j->ij", ddNu, dNv).reshape(-1),
        np.einsum("i,j->ij", dNu, ddNv).reshape(-1),
        np.einsum("i,j->ij", Nu, dddNv).reshape(-1))


def _expand(B: np.ndarray) -> np.ndarray:
    """Expand an m-vector of basis values into the coordinate-interleaved 3m
    design vector (x0,y0,z0, x1,y1,z0, ...)."""
    return np.repeat(B, 3)


def assemble_energy(ref: BSplineSurface, gauss_order: int = 3,
                    bend_weight: float = 1.0,
                    memb_weight: float = 0.0,
                    bend_mode: str = "jerk") -> EnergyModel:
    if gauss_order not in _GL:
        raise ValueError(f"unsupported Gauss order {gauss_order}")
    if bend_mode not in BEND_MODES:
        raise ValueError(f"bend_mode must be one of {BEND_MODES}")
    u_pts, u_w = _gauss_nodes(ref.knots_u, gauss_order)
    v_pts, v_w = _gauss_nodes(ref.knots_v, gauss_order)
    m = ref.n_control
    H = np.zeros((3 * m, 3 * m))
    a, b = ref.metric.scale_u, ref.metric.scale_v

    for gu, wu in zip(u_pts, u_w):
        for gv, wv in zip(v_pts, v_w):
            d = ref.partials(gu, gv)
            dA = max(d["jacobian"], 1e-12) * a * b
            w = wu * wv
            normal = d["n"]
            (B, Bu, Bv, Buu, Buv, Bvv,
             Buuu, Buuv, Buvv, Bvvv) = _mixed_basis(ref, gu, gv)
            if bend_weight > 0:
                c = bend_weight * dA * w
                if bend_mode == "jerk":
                    # binomial 1:3:3:1 weighting of the four third partials
                    for Bdd, sc, cross in (
                            (Buuu, 1.0 / a ** 3, 1.0),
                            (Buuv, 1.0 / (a ** 2 * b), 3.0),
                            (Buvv, 1.0 / (a * b ** 2), 3.0),
                            (Bvvv, 1.0 / b ** 3, 1.0)):
                        v = _expand(Bdd * sc)
                        H += cross * c * np.outer(v, v)
                elif bend_mode == "normal":
                    # second fundamental form: weight each coordinate by the
                    # analytic reference normal so the term measures curvature
                    for Bdd, sc, cross in (
                            (Buu, 1.0 / a ** 2, 1.0),
                            (Bvv, 1.0 / b ** 2, 1.0),
                            (Buv, 1.0 / (a * b), 2.0)):
                        v = _expand(Bdd * sc) * np.tile(normal, m)
                        H += cross * c * np.outer(v, v)
                else:  # thin_plate
                    for Bdd, sc, cross in (
                            (Buu, 1.0 / a ** 2, 1.0),
                            (Bvv, 1.0 / b ** 2, 1.0),
                            (Buv, 1.0 / (a * b), 2.0)):
                        v = _expand(Bdd * sc)
                        H += cross * c * np.outer(v, v)
            if memb_weight > 0:
                c = memb_weight * dA * w
                for B in (Bu / a, Bv / b):
                    v = _expand(B)
                    H += c * np.outer(v, v)

    x_ref = ref.flatten().reshape(-1)
    # Normalize the stiffness matrix to dimensionless, order-one magnitude so
    # the data-attachment and soft-penalty weights keep their meaning across
    # bend modes and parameter scales (raw third derivatives are O(1e5) while
    # second derivatives are O(1e2)). The factor is stored so the physical
    # energy can still be recovered; direct_energy remains the authoritative
    # re-integrated physical value used in the reports.
    diag = np.diag(H)
    energy_scale = float(np.mean(diag[diag > 0])) if np.any(diag > 0) else 1.0
    H = H / energy_scale
    f = H @ x_ref
    const = 0.5 * float(x_ref @ H @ x_ref)
    return EnergyModel(
        H=H, f=f, const=const, x_ref=x_ref, ref=ref, gauss_order=gauss_order,
        bend_weight=bend_weight, memb_weight=memb_weight,
        bend_mode=bend_mode,
        n_integration_points=len(u_pts) * len(v_pts),
        metric_label=ref.metric.label, linearized=True,
        energy_scale=energy_scale)


def direct_energy(surf: BSplineSurface, gauss_order: int = 3,
                  bend_weight: float = 1.0,
                  memb_weight: float = 0.0,
                  bend_mode: str = "jerk") -> float:
    """Numerically integrate the actual energy of a surface over knot spans.

    For ``bend_mode='normal'`` the bending term uses the analytic normal
    projection of the second derivatives (the second fundamental form). For
    ``'thin_plate'`` it uses raw metric-scaled second derivatives, and for
    ``'jerk'`` it uses the metric-scaled third partials (curvature variation).
    Re-integrating on the faired surface with its own area element means the
    reported value tracks the real geometry rather than the linearization.
    """
    if bend_mode not in BEND_MODES:
        raise ValueError(f"bend_mode must be one of {BEND_MODES}")
    u_pts, u_w = _gauss_nodes(surf.knots_u, gauss_order)
    v_pts, v_w = _gauss_nodes(surf.knots_v, gauss_order)
    a, b = surf.metric.scale_u, surf.metric.scale_v
    total = 0.0
    for gu, wu in zip(u_pts, u_w):
        for gv, wv in zip(v_pts, v_w):
            d = surf.partials(gu, gv)
            dA = max(d["jacobian"], 1e-12) * a * b
            n = d["n"]
            if bend_mode == "jerk":
                suuu = d["Suuu"] / a ** 3
                suuv = d["Suuv"] / (a ** 2 * b)
                suvv = d["Suvv"] / (a * b ** 2)
                svvv = d["Svvv"] / b ** 3
                bend = (np.dot(suuu, suuu) + 3 * np.dot(suuv, suuv)
                        + 3 * np.dot(suvv, suvv) + np.dot(svvv, svvv))
            elif bend_mode == "normal":
                eu = float(np.dot(d["Suu"] / a ** 2, n))
                ev = float(np.dot(d["Svv"] / b ** 2, n))
                euv = float(np.dot(d["Suv"] / (a * b), n))
                bend = eu * eu + 2 * euv * euv + ev * ev
            else:
                suu = d["Suu"] / a ** 2
                svv = d["Svv"] / b ** 2
                suv = d["Suv"] / (a * b)
                bend = (np.dot(suu, suu) + 2 * np.dot(suv, suv)
                        + np.dot(svv, svv))
            su, sv = d["Su"] / a, d["Sv"] / b
            memb = np.dot(su, su) + np.dot(sv, sv)
            total += wu * wv * dA * (bend_weight * bend + memb_weight * memb)
    return 0.5 * total
