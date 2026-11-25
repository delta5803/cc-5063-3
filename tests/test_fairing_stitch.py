import unittest

import numpy as np

from surface_fairing.core.benchmark import mean_filter
from surface_fairing.core.constraints import (
    HardConstraints, make_ring_soft_lock)
from surface_fairing.core.curvature import curvature_field
from surface_fairing.core.fairing import fair_surface
from surface_fairing.core.solver import InfeasibleConstraints, solve_qp
from surface_fairing.io_.artifacts import (
    conflicting_constraint_case, over_fairing_case, rippled_curved,
    rippled_plane)
from surface_fairing.io_.stitch import (
    build_stitched_mesh, declare_edge, negotiate)


def _edge_locked(surf):
    hard = HardConstraints(3 * surf.n_control)
    for edge in ("u0", "u1", "v0", "v1"):
        hard.pin_edge(surf, edge)
    return hard


def _smooth_dome(surf):
    """The ripple-free dome sharing the artifact's grid and parameters."""
    from surface_fairing.core.bspline import greville_abscissae
    gu = greville_abscissae(surf.nu, surf.degree_u)
    gv = greville_abscissae(surf.nv, surf.degree_v)
    U, V = np.meshgrid(gu, gv, indexing="ij")
    z = 0.25 * np.sin(np.pi * U) * np.sin(np.pi * V)
    from surface_fairing.core.bspline import BSplineSurface
    return BSplineSurface(np.stack([U, V, z], axis=-1))


def _rms_distance(a, b, grid=40):
    ts = np.linspace(0.0, 1.0, grid)
    d = [np.linalg.norm(a.evaluate(u, v) - b.evaluate(u, v))
         for u in ts for v in ts]
    return float(np.sqrt(np.mean(np.square(d))))


def _high_frequency_rms(surf, grid=48, cutoff=3):
    """RMS of the spectral high-frequency part of the sampled z-field."""
    ts = np.linspace(0.0, 1.0, grid)
    Z = np.array([[surf.evaluate(u, v)[2] for v in ts] for u in ts])
    F = np.fft.fft2(Z - Z.mean())
    k = np.minimum(np.arange(grid), grid - np.arange(grid))
    KU, KV = np.meshgrid(k, k, indexing="ij")
    mask = (np.abs(KU) >= cutoff) | (np.abs(KV) >= cutoff)
    return float(np.sqrt(np.mean(np.real(np.fft.ifft2(F * mask)) ** 2)))


class TestFairing(unittest.TestCase):
    def test_plane_ripple_is_faired_with_locked_boundary(self):
        art = rippled_plane()
        surf = art.surface
        hard = _edge_locked(surf)
        res = fair_surface(surf, hard, attachment_weight=0.4,
                           tangent_edges=("u0",))
        self.assertLess(res.hard_constraint_error, 1e-9)
        self.assertLess(res.direct_after, res.direct_before)
        for edge in ("u0", "u1", "v0", "v1"):
            gap = np.linalg.norm(
                res.original.edge_samples(edge)
                - res.faired.edge_samples(edge), axis=1).max()
            self.assertLess(gap, 1e-9)
        # curvature oscillation amplitude must actually decrease
        _, _, before = curvature_field(surf, 21, 21)
        _, _, after = curvature_field(res.faired, 21, 21)
        self.assertLess(np.abs(after["mean"]).max(),
                        np.abs(before["mean"]).max())
        # tangential range is returned for the stitcher
        self.assertIn("u0", res.tangential_ranges)

    def test_curved_patch_keeps_base_form(self):
        art = rippled_curved()
        surf = art.surface
        hard = _edge_locked(surf)
        res = fair_surface(surf, hard, attachment_weight=0.2)
        self.assertLess(res.hard_constraint_error, 1e-9)
        self.assertLess(res.direct_after, res.direct_before)
        # the smooth dome must survive: center height stays near the design
        z = res.faired.evaluate(0.5, 0.5)[2]
        self.assertAlmostEqual(z, art.expected["base_dome_z_at_center"],
                               delta=0.01)
        self.assertLess(res.position_residual, 0.02)
        # against the known smooth base the faired surface is no farther than
        # the rippled input (it removes ripple rather than inventing shape),
        # and its high-frequency content drops below the input's.
        base = _smooth_dome(art.surface)
        self.assertLessEqual(_rms_distance(res.faired, base),
                             _rms_distance(surf, base) + 1e-4)
        self.assertLess(_high_frequency_rms(res.faired),
                        _high_frequency_rms(surf))

    def test_key_point_lock_preserves_feature(self):
        art = over_fairing_case()
        surf = art.surface
        hard = _edge_locked(surf)
        hard.pin_surface_point(surf, 0.5, 0.5)
        res = fair_surface(surf, hard, attachment_weight=0.3)
        kept = res.faired.evaluate(0.5, 0.5)[2]
        self.assertAlmostEqual(kept, art.expected["key_height"], places=6)

    def test_over_fairing_without_lock_flattens_feature(self):
        # Retained failure/over-fairing case. With the stiffness matrix
        # normalized, a small data-attachment weight lets the curvature-
        # variation objective move the surface freely: the un-locked bump
        # shrinks well below its design height. The key-point-lock test above
        # is the correct way to prevent this.
        art = over_fairing_case()
        surf = art.surface
        hard = _edge_locked(surf)
        res = fair_surface(surf, hard, attachment_weight=0.02)
        self.assertLess(res.faired.evaluate(0.5, 0.5)[2],
                        art.expected["key_height"] - 0.01)

    def test_inconsistent_hard_constraints_raise(self):
        n = 9 * 9 * 3
        hard = HardConstraints(n)
        hard.pin_dof(0, 0, 1.0)
        hard.pin_dof(0, 0, 2.0)
        H = np.eye(n)
        with self.assertRaises(InfeasibleConstraints):
            solve_qp(H, np.zeros(n), hard)

    def test_soft_lock_relaxation_is_recorded(self):
        art = rippled_curved()
        surf = art.surface
        hard = _edge_locked(surf)
        soft = [make_ring_soft_lock(surf, "u0", weight=1000.0)]
        res = fair_surface(surf, hard, soft=soft, attachment_weight=0.5)
        # either the soft lock holds or its relaxation is fully reported
        if res.soft_events:
            event = res.soft_events[0]
            self.assertIn("ring_indices", event)
            self.assertIn("max_deviation", event)
            self.assertGreaterEqual(event["max_deviation"], 0.0)
        self.assertLess(res.hard_constraint_error, 1e-8)


class TestMeanFilterIsDifferent(unittest.TestCase):
    def test_mean_filter_is_not_the_optimizer(self):
        art = rippled_plane()
        surf = art.surface
        filtered = mean_filter(surf, iterations=3)
        hard = _edge_locked(surf)
        res = fair_surface(surf, hard, attachment_weight=0.4)
        # both reduce ripple but produce different control points
        self.assertFalse(np.allclose(filtered.P, res.faired.P))
        # the energy optimizer must reach lower bending energy
        from surface_fairing.core.energy import direct_energy
        self.assertLess(direct_energy(res.faired),
                        direct_energy(filtered))


class TestStitching(unittest.TestCase):
    def test_stitched_mesh_gap_and_contour(self):
        a, b = conflicting_constraint_case()
        mesh = build_stitched_mesh(a, "u0", b, "u0", n_along=24, n_across=10)
        self.assertEqual(mesh.faces.shape[1], 3)
        self.assertGreater(mesh.vertices.shape[0], 0)
        # the deliberately offset edge must be detected as a gap
        self.assertGreater(mesh.seam_gap, 0.01)
        self.assertGreaterEqual(mesh.contour_error_a, 0.0)

    def test_negotiation_requires_authorized_yield(self):
        # Two patches share a geometrically coincident edge, declared with
        # opposite numbering semantics. B's cross-edge tangent is perturbed
        # beyond the narrow achievable envelope while both positions stay
        # fixed, so the only resolution is a concession on authorized B.
        art = rippled_plane()
        a = art.surface
        a.name = "patch_A"
        b = a.clone()
        b.name = "patch_B"
        da = declare_edge(a, "u0", True, True, authorized_to_yield=False)
        db = declare_edge(b, "u0", True, True, authorized_to_yield=True)
        from surface_fairing.core.fairing import TangentialRange
        t = da.tangents
        narrow_a = TangentialRange(
            "u0", np.linspace(0, 1, t.shape[0]), t, t, t, t)
        tb = t.copy()
        tb[:, 2] += 0.8
        # reference stays the original tangent; achieved carries the
        # perturbation, so B's own envelope reveals the deviation
        narrow_b = TangentialRange(
            "u0", np.linspace(0, 1, t.shape[0]), tb - 0.3, tb + 0.3,
            tb, t)
        neg = negotiate(da, narrow_a, db, narrow_b)
        # the conflict is only resolvable because B is authorized to yield
        self.assertTrue(neg.compatible)
        self.assertEqual(neg.yielding_patch, "patch_B")
        self.assertGreater(neg.tangent_mismatch_deg,
                           narrow_a.angle_envelope_deg()[1])

        # without an authorized patch the same constraints are unresolvable
        da_locked = declare_edge(a, "u0", True, True,
                                 authorized_to_yield=False)
        db_locked = declare_edge(b, "u0", True, True,
                                 authorized_to_yield=False)
        neg_locked = negotiate(da_locked, narrow_a, db_locked, narrow_b)
        self.assertFalse(neg_locked.compatible)
        self.assertIsNone(neg_locked.yielding_patch)

    def test_compatible_edges_negotiate_cleanly(self):
        art = rippled_plane()
        a = art.surface
        b = a.clone()
        da = declare_edge(a, "u0", True, True)
        db = declare_edge(b, "u0", True, True)
        from surface_fairing.core.fairing import TangentialRange
        t = da.tangents
        n = t.shape[0]
        env = TangentialRange("u0", np.linspace(0, 1, n), t, t, t, t)
        neg = negotiate(da, env, db, env)
        self.assertTrue(neg.compatible)
        self.assertLess(neg.position_gap, 1e-10)


if __name__ == "__main__":
    unittest.main()
