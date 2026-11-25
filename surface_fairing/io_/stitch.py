"""Neighbor patch stitching with declared fixed edges and seam diagnostics.

Workflow mandated by the design:

1. The neighbor stitcher first declares which common edges cannot move
   (:class:`SeamDeclaration`); the fairer then returns the achievable
   cross-edge tangent range on its side.
2. If the two sides' constraints are mutually exclusive, the local concession
   is applied only on an explicitly authorized patch — never silently moved
   on a locked neighbor.
3. A real stitched mesh is produced for a pair of patches whose shared edge is
   numbered in opposite directions, and the seam gap and each patch's contour
   error are measured on actual sampled geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.bspline import BSplineSurface
from ..core.fairing import TangentialRange


@dataclass
class SeamDeclaration:
    """One side of a common edge."""

    patch_name: str
    edge: str
    fixed_position: bool          # edge control points cannot move
    fixed_tangent: bool           # adjacent ring cannot move
    authorized_to_yield: bool = False
    samples: np.ndarray | None = None
    tangents: np.ndarray | None = None


@dataclass
class SeamNegotiation:
    compatible: bool
    reason: str
    yielding_patch: str | None
    position_gap: float
    tangent_mismatch_deg: float
    tangent_envelope: dict[str, tuple[float, float]] = field(default_factory=dict)


def _opposite(edge: str) -> str:
    return {"u0": "u1", "u1": "u0", "v0": "v1", "v1": "v0"}[edge]


def declare_edge(surf: BSplineSurface, edge: str,
                 fixed_position: bool, fixed_tangent: bool,
                 authorized_to_yield: bool = False) -> SeamDeclaration:
    return SeamDeclaration(
        patch_name=surf.name, edge=edge, fixed_position=fixed_position,
        fixed_tangent=fixed_tangent,
        authorized_to_yield=authorized_to_yield,
        samples=surf.edge_samples(edge),
        tangents=surf.edge_cross_tangents(edge))


def _aligned(decl_a: SeamDeclaration, decl_b: SeamDeclaration):
    """Align B's samples/tangents to A's traversal direction.

    Edges are oppositely numbered when they are opposite parametric sides
    (u0/u1 or v0/v1); same-side declarations already run in the same
    direction and must not be reversed.
    """
    oppositely_numbered = decl_a.edge in (
        "u1" if decl_b.edge == "u0" else
        "u0" if decl_b.edge == "u1" else
        "v1" if decl_b.edge == "v0" else
        "v0" if decl_b.edge == "v1" else "")
    if oppositely_numbered:
        return decl_b.samples[::-1], decl_b.tangents[::-1]
    return decl_b.samples, decl_b.tangents


def negotiate(decl_a: SeamDeclaration, range_a: TangentialRange,
              decl_b: SeamDeclaration, range_b: TangentialRange,
              n: int = 31) -> SeamNegotiation:
    """Check compatibility of the two declared sides of a common edge.

    Position compatibility uses the declared edge samples. Tangent
    compatibility uses the achieved cross-edge tangents carried by the
    fairer's range objects (which reflect any perturbation/relaxation), and
    compares them against the achievable angular envelopes. B's arrays are
    reversed only when the edges are oppositely numbered.
    """
    pb, _ = _aligned(decl_a, decl_b)
    pa = decl_a.samples
    gap = float(np.max(np.linalg.norm(pa - pb, axis=1)))

    ta = range_a.achieved
    tb_achieved = range_b.achieved
    u_edges = {"u0", "u1"}
    v_edges = {"v0", "v1"}
    oppositely_numbered = (
        decl_a.edge != decl_b.edge
        and (({decl_a.edge, decl_b.edge} == u_edges)
             or ({decl_a.edge, decl_b.edge} == v_edges)))
    if oppositely_numbered:
        tb_achieved = tb_achieved[::-1]
    denom = np.maximum(
        np.linalg.norm(ta, axis=1) * np.linalg.norm(tb_achieved, axis=1),
        1e-14)
    cosang = np.clip(np.sum(ta * tb_achieved, axis=1) / denom, -1, 1)
    tangent_mismatch = float(np.degrees(np.max(np.arccos(cosang))))

    lo_a, hi_a = range_a.angle_envelope_deg()
    lo_b, hi_b = range_b.angle_envelope_deg()
    # A's achievable angular range is [lo_a, hi_a]; the neighbor demand
    # (the measured mismatch) must fit inside it for the locked tangent to be
    # reachable without moving A's fixed edge.
    a_reachable = lo_a - 1e-9 <= tangent_mismatch <= hi_a + 1e-9

    position_locked_both = decl_a.fixed_position and decl_b.fixed_position
    tangent_locked_both = decl_a.fixed_tangent and decl_b.fixed_tangent
    conflict = (position_locked_both and gap > 1e-6) or (
        tangent_locked_both and not a_reachable)

    if not conflict:
        return SeamNegotiation(
            compatible=True, reason="constraints compatible",
            yielding_patch=None, position_gap=gap,
            tangent_mismatch_deg=tangent_mismatch,
            tangent_envelope={decl_a.patch_name: (lo_a, hi_a),
                              decl_b.patch_name: (lo_b, hi_b)})

    # Mutually exclusive: the concession must land on an authorized patch.
    yielder = None
    for decl in (decl_a, decl_b):
        if decl.authorized_to_yield:
            yielder = decl.patch_name
            break
    reason = ("mutually exclusive fixed-edge constraints"
              if yielder is None else
              f"mutually exclusive; concession placed on authorized patch {yielder}")
    return SeamNegotiation(
        compatible=yielder is not None, reason=reason,
        yielding_patch=yielder, position_gap=gap,
        tangent_mismatch_deg=tangent_mismatch,
        tangent_envelope={decl_a.patch_name: (lo_a, hi_a),
                          decl_b.patch_name: (lo_b, hi_b)})


@dataclass
class StitchedMesh:
    vertices: np.ndarray
    faces: np.ndarray
    seam_indices_a: np.ndarray
    seam_indices_b: np.ndarray
    seam_gap: float
    contour_error_a: float
    contour_error_b: float


def build_stitched_mesh(patch_a: BSplineSurface, edge_a: str,
                        patch_b: BSplineSurface, edge_b: str,
                        n_along: int = 32, n_across: int = 16) -> StitchedMesh:
    """Build an actual triangle mesh stitching two patches along a common edge.

    The edge is assumed to run in opposite parameter directions on the two
    patches, so patch B's edge samples are reversed before merging.
    """
    grids = []
    seam_idx = []
    running = 0
    for patch, edge, reverse in ((patch_a, edge_a, False),
                                 (patch_b, edge_b, True)):
        us, vs, pts = patch.sample_grid(n_along, n_across)
        nu, nv, _ = pts.shape
        # reorder so the shared edge is the last column for A, first for B,
        # then flatten consistently
        if edge in ("u0", "u1"):
            mat = pts if edge == "u0" else pts[::-1, :, :]
        else:
            mat = pts if edge == "v0" else pts[:, ::-1, :]
        if reverse:
            mat = mat[::-1, ::-1, :] if edge in ("u0", "u1") else mat[:, ::-1, :]
        flat = mat.reshape(-1, 3)
        edge_pts = patch.edge_samples(edge, n_along)
        if reverse:
            edge_pts = edge_pts[::-1]
        grids.append((flat, nu, nv, edge_pts))
        seam_idx.append(np.arange(running, running + n_along))
        running += flat.shape[0]

    (fa, nua, nva, edge_a_pts), (fb, nub, nvb, edge_b_pts) = grids
    faces = []
    offset = 0
    for flat, nu, nv in ((fa, nua, nva), (fb, nub, nvb)):
        for i in range(nu - 1):
            for j in range(nv - 1):
                a0 = offset + i * nv + j
                a1 = a0 + 1
                b0 = offset + (i + 1) * nv + j
                b1 = b0 + 1
                faces.append([a0, b0, a1])
                faces.append([a1, b0, b1])
        offset += flat.shape[0]

    vertices = np.vstack([fa, fb])
    faces = np.asarray(faces)

    # seam gap: compare aligned edge samples rather than raw merged vertices
    seam_gap = float(np.max(np.linalg.norm(edge_a_pts - edge_b_pts, axis=1)))
    contour_a = _contour_error(patch_a, edge_a, n_along)
    contour_b = _contour_error(patch_b, edge_b, n_along)
    return StitchedMesh(
        vertices=vertices, faces=faces,
        seam_indices_a=seam_idx[0], seam_indices_b=seam_idx[1],
        seam_gap=seam_gap,
        contour_error_a=contour_a, contour_error_b=contour_b)


def _contour_error(patch: BSplineSurface, edge: str, n: int) -> float:
    """Deviation of the actual boundary curve from its chord (edge waviness)."""
    pts = patch.edge_samples(edge, n)
    chord = np.linspace(pts[0], pts[-1], n)
    return float(np.max(np.linalg.norm(pts - chord, axis=1)))
