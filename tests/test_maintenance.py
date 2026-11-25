import unittest

import numpy as np

from surface_fairing.core.constraints import HardConstraints, make_ring_soft_lock
from surface_fairing.core.maintenance import (
    ConstraintUnit, analyze, build_conflict_graph, find_minimal_unsatisfiable,
    recompute_after_removal, units_from_hard)
from surface_fairing.io_.artifacts import rippled_curved, rippled_plane


def _locked(surf):
    hard = HardConstraints(3 * surf.n_control)
    for edge in ("u0", "u1", "v0", "v1"):
        hard.pin_edge(surf, edge)
    return hard


class TestMinimalConflict(unittest.TestCase):
    def test_consistent_set_has_no_conflict(self):
        s = rippled_plane().surface
        hard = _locked(s)
        self.assertIsNone(find_minimal_unsatisfiable(units_from_hard(hard)))

    def test_edge_lock_vs_key_point_is_minimal(self):
        # S(0, 0.5) lies on the locked u0 edge; demanding a different z there
        # contradicts the edge lock, so the minimal conflict is exactly those
        # two units (not the other three edges).
        s = rippled_plane().surface
        hard = _locked(s)
        hard.pin_surface_point(s, 0.0, 0.5,
                               position=np.array([0.0, 0.5, 0.5]))
        units = units_from_hard(hard)
        minimal = find_minimal_unsatisfiable(units)
        self.assertEqual(
            minimal, ["hard edge lock u0", "key point S(0.000,0.500)"])
        # minimality: dropping either member leaves a consistent set
        for label in minimal:
            rest = [u for u in units if u.label != label]
            self.assertIsNone(find_minimal_unsatisfiable(rest))


class TestPruneVsRelax(unittest.TestCase):
    def test_feasible_hard_set_has_no_prune_edge(self):
        s = rippled_curved().surface
        hard = _locked(s)
        rep = build_conflict_graph(
            units_from_hard(hard), surf=s,
            relaxation_weights=(10.0, 100.0, 1000.0))
        # feasible hard set -> no prune conflict, no _hard_infeasible flag
        self.assertIsNone(rep.minimal_conflict)
        self.assertNotIn("_hard_infeasible", rep.relaxation_boundaries)
        self.assertTrue(all(e.kind != "prune" for e in rep.edges))

    def test_infeasible_hard_blocks_relaxation_sweep(self):
        s = rippled_plane().surface
        hard = _locked(s)
        hard.pin_surface_point(s, 0.0, 0.5,
                               position=np.array([0.0, 0.5, 0.5]))
        soft = [make_ring_soft_lock(s, "u0", weight=100.0)]
        rep = analyze(s, hard, soft)
        # a hard-hard conflict is reported and the soft sweep is skipped
        self.assertIsNotNone(rep.minimal_conflict)
        self.assertIn("_hard_infeasible", rep.relaxation_boundaries)
        self.assertTrue(all(e.kind == "prune" for e in rep.edges))
        self.assertFalse(rep.relaxation_boundaries[
            "_hard_infeasible"].feasible_at_tolerance)

    def test_relaxation_boundary_shrinks_with_weight(self):
        s = rippled_curved().surface
        hard = _locked(s)
        soft = [make_ring_soft_lock(s, "u0", weight=100.0)]
        rep = analyze(s, hard, soft,
                      relaxation_weights=(10.0, 100.0, 1000.0))
        boundary = rep.relaxation_boundaries["soft ridge/ring lock u0"]
        self.assertEqual(boundary.weights, [10.0, 100.0, 1000.0])
        # heavier penalty -> smaller geometric deviation (monotone here)
        self.assertGreater(boundary.deviations[0], boundary.deviations[1])
        self.assertGreater(boundary.deviations[1], boundary.deviations[2])
        self.assertAlmostEqual(boundary.min_deviation,
                               min(boundary.deviations))
        self.assertTrue(boundary.feasible_at_tolerance)


class TestReactivation(unittest.TestCase):
    def test_removed_conflicting_key_point_reports_new_target(self):
        s = rippled_plane().surface
        hard = _locked(s)
        hard.pin_surface_point(s, 0.0, 0.5,
                               position=np.array([0.0, 0.5, 0.5]))
        report = recompute_after_removal(s, hard, "key point S(0.000,0.500)")
        self.assertIsNone(report.error)
        self.assertIsNotNone(report.fairing)
        # remaining hard set must now be feasible
        self.assertLess(report.fairing.hard_constraint_error, 1e-9)
        # the removed constraint can only reactivate at the achieved value
        self.assertGreater(report.reactivation_gap, 0.1)
        # achieved target is z=0 (the locked edge plane), not the demanded 0.5
        self.assertAlmostEqual(float(report.achieved_target[2]), 0.0, places=6)
        self.assertAlmostEqual(float(report.reimposed_target[2]), 0.5)
        self.assertIn("reactivates only if its target is changed",
                      report.condition)

    def test_removing_non_conflicting_unit_keeps_other_locks(self):
        s = rippled_plane().surface
        hard = _locked(s)
        report = recompute_after_removal(s, hard, "hard edge lock u1")
        self.assertIsNone(report.error)
        self.assertLess(report.fairing.hard_constraint_error, 1e-9)
        # u1 is now free to move, so its original position is not reproduced
        self.assertGreater(report.reactivation_gap, 1e-6)


if __name__ == "__main__":
    unittest.main()
