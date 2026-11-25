import unittest

import numpy as np

from surface_fairing.core.bspline import (
    BSplineSurface, basis_table, clamped_knots, greville_abscissae)
from surface_fairing.core.curvature import (
    curvature_change, curvature_field, isophote_field)
from surface_fairing.core.energy import assemble_energy, direct_energy


class TestBasis(unittest.TestCase):
    def test_partition_of_unity(self):
        for degree in (2, 3, 4):
            knots = clamped_knots(8, degree)
            for t in np.linspace(0.0, 1.0, 25):
                N, dN, ddN, dddN = basis_table(t, knots, degree)
                self.assertAlmostEqual(N.sum(), 1.0, places=10)
                self.assertAlmostEqual(dN.sum(), 0.0, places=8)
                self.assertAlmostEqual(ddN.sum(), 0.0, places=8)
                self.assertAlmostEqual(dddN.sum(), 0.0, places=6)

    def test_endpoint_interpolation(self):
        knots = clamped_knots(6, 3)
        N0, _, _, _ = basis_table(0.0, knots, 3)
        N1, _, _, _ = basis_table(1.0, knots, 3)
        self.assertAlmostEqual(N0[0], 1.0)
        self.assertAlmostEqual(N1[-1], 1.0)

    def test_derivatives_match_finite_difference(self):
        knots = clamped_knots(7, 3)
        h = 1e-6
        for t in np.linspace(0.05, 0.95, 40):
            Np, _, _, _ = basis_table(t + h, knots, 3)
            Nm, _, _, _ = basis_table(t - h, knots, 3)
            N, dN, ddN, dddN = basis_table(t, knots, 3)
            self.assertLess(np.max(np.abs((Np - Nm) / (2 * h) - dN)), 1e-5)
            self.assertLess(
                np.max(np.abs((Np - 2 * N + Nm) / h ** 2 - ddN)), 1e-2)
            # third derivative against a one-sided second-difference of ddN
            _, _, ddP, _ = basis_table(t + h, knots, 3)
            _, _, ddM, _ = basis_table(t - h, knots, 3)
            self.assertLess(
                np.max(np.abs((ddP - ddM) / (2 * h) - dddN)), 5e-2)

    def test_greville_plane_is_exact(self):
        nu, nv = 6, 7
        gu = greville_abscissae(nu, 3)
        gv = greville_abscissae(nv, 3)
        G = np.meshgrid(gu, gv, indexing="ij")
        P = np.stack([G[0], G[1], np.zeros((nu, nv))], axis=-1)
        surf = BSplineSurface(P)
        for u in (0.0, 0.37, 1.0):
            for v in (0.0, 0.61, 1.0):
                d = surf.partials(u, v)
                np.testing.assert_allclose(d["S"], [u, v, 0], atol=1e-11)
                np.testing.assert_allclose(d["n"], [0, 0, 1], atol=1e-11)


class TestEnergy(unittest.TestCase):
    def _curved(self):
        gu = greville_abscissae(8, 3)
        gv = greville_abscissae(9, 3)
        G = np.meshgrid(gu, gv, indexing="ij")
        z = 0.2 * np.sin(np.pi * G[0]) * np.sin(np.pi * G[1])
        return BSplineSurface(np.stack([G[0], G[1], z], axis=-1))

    def test_flat_patch_has_zero_bending(self):
        gu = greville_abscissae(6, 3)
        gv = greville_abscissae(7, 3)
        G = np.meshgrid(gu, gv, indexing="ij")
        flat = BSplineSurface(
            np.stack([G[0], G[1], np.zeros((6, 7))], axis=-1))
        from surface_fairing.core.energy import BEND_MODES
        for mode in BEND_MODES:
            self.assertLess(direct_energy(flat, bend_mode=mode), 1e-18)
            model = assemble_energy(flat, bend_mode=mode)
            self.assertLess(
                model.absolute_energy(flat.flatten().reshape(-1)), 1e-10)

    def test_quadratic_matches_direct_integration(self):
        from surface_fairing.core.energy import BEND_MODES
        surf = self._curved()
        for mode in BEND_MODES:
            model = assemble_energy(surf, bend_mode=mode)
            direct = direct_energy(surf, bend_mode=mode)
            quad = model.absolute_energy(surf.flatten().reshape(-1))
            self.assertLess(abs(direct - quad) / direct, 1e-9)

    def test_normalization_keeps_attachment_commensurate(self):
        # after normalization the stiffness diagonal is order one, so the
        # dimensionless attachment weight is no longer swamped by raw
        # third-derivative magnitudes (which are O(1e5) before scaling).
        model = assemble_energy(self._curved(), bend_mode="jerk")
        self.assertAlmostEqual(float(np.mean(np.diag(model.H))), 1.0, places=8)
        self.assertGreater(model.energy_scale, 1.0)

    def test_metric_scales_change_energy(self):
        surf = self._curved()
        e_default = direct_energy(surf)
        surf.metric.scale_u = 2.0
        e_scaled = direct_energy(surf)
        self.assertNotAlmostEqual(e_default, e_scaled)


class TestCurvature(unittest.TestCase):
    def test_plane_has_zero_curvature(self):
        gu = greville_abscissae(6, 3)
        gv = greville_abscissae(7, 3)
        G = np.meshgrid(gu, gv, indexing="ij")
        flat = BSplineSurface(
            np.stack([G[0], G[1], np.zeros((6, 7))], axis=-1))
        _, _, f = curvature_field(flat, 9, 9)
        self.assertLess(np.abs(f["mean"]).max(), 1e-10)
        self.assertLess(np.abs(f["gaussian"]).max(), 1e-10)

    def test_rippled_curvature_oscillates(self):
        from surface_fairing.io_.artifacts import rippled_plane
        surf = rippled_plane().surface
        _, _, f = curvature_field(surf, 25, 25)
        self.assertGreater(f["mean"].max(), 0.0)
        self.assertLess(f["mean"].min(), 0.0)

    def test_isophote_uses_analytic_normal(self):
        from surface_fairing.io_.artifacts import rippled_plane
        surf = rippled_plane().surface
        us, vs, val = isophote_field(surf, np.array([0, 0, 1]), 12, 12)
        self.assertEqual(val.shape, (12, 12))
        self.assertTrue(np.all(np.abs(val) <= 1.0 + 1e-12))

    def test_curvature_change_field_shape(self):
        from surface_fairing.io_.artifacts import rippled_plane
        a = rippled_plane().surface
        b = a.clone()
        out = curvature_change(a, b, 8, 8)
        self.assertEqual(out["mean_delta"].shape, (8, 8))


if __name__ == "__main__":
    unittest.main()
