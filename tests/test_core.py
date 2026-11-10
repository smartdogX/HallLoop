"""Synthetic regression checks for Hall-loop branch pairing."""

import json
import unittest

import numpy as np

from hallloop.core import analyze


class AnalyzeTests(unittest.TestCase):
    def test_removes_channel_mixing_and_preserves_hysteretic_branches(self):
        field_up = np.linspace(-2, 2, 33)
        field_down = field_up[::-1]
        field = np.concatenate((field_up, field_down))
        direction = np.repeat([1, -1], field_up.size)
        hall = np.tanh((field - 0.4 * direction) / 0.15)
        longitudinal = 0.3 + 0.11 * field**2 + 0.07 * field * direction

        result = analyze(field, longitudinal + 0.2 * hall, hall + 0.5 * longitudinal)

        np.testing.assert_array_equal(result.field, -result.field[::-1])
        np.testing.assert_allclose(result.longitudinal_even_up, longitudinal[:field_up.size])
        np.testing.assert_allclose(result.longitudinal_even_down, longitudinal[field_up.size:][::-1])
        np.testing.assert_allclose(result.transverse_odd_up, hall[:field_up.size])
        np.testing.assert_allclose(result.transverse_odd_down, hall[field_up.size:][::-1])
        center = np.flatnonzero(result.field == 0)[0]
        self.assertLess(result.transverse_odd_up[center], -0.9)
        self.assertGreater(result.transverse_odd_down[center], 0.9)
        self.assertEqual(result.naive_odd_up[center], 0)
        self.assertEqual(result.naive_odd_down[center], 0)
        json.dumps({"branches": result.branches, "candidates": result.candidates})

    def test_uses_measured_and_reflected_knots_without_extrapolation(self):
        field = np.array([-2, -1.25, 0.5, 2, 2, 0.75, -0.5, -1.9375])
        longitudinal = 3 + 2 * field
        transverse = 2 * field

        with self.assertRaisesRegex(ValueError, "No candidate pairs"):
            analyze(field, longitudinal, transverse)
        result = analyze(field, longitudinal, transverse, field_tolerance=0.0625)

        np.testing.assert_array_equal(
            result.field, [-1.9375, -1.25, -0.75, -0.5, 0.5, 0.75, 1.25, 1.9375],
        )
        np.testing.assert_allclose(result.longitudinal_up, 3 + 2 * result.field)
        np.testing.assert_allclose(result.longitudinal_down, 3 + 2 * result.field)
        np.testing.assert_allclose(result.longitudinal_even_up, 3)
        np.testing.assert_allclose(result.transverse_odd_down, 2 * result.field)
        self.assertEqual(result.candidates[0]["measured_overlap"], [-1.9375, 2.0])
        self.assertEqual(result.candidates[0]["common_field_range"], [-1.9375, 1.9375])

    def test_keeps_turning_dwell_records_separate_and_averages_inner_duplicates(self):
        field = [-2, -1, -1, 0, 2, 2, 2, 1, 0, -2]
        signal = [1, 4, 6, 0, 15, 900, 30, 10, 7, 2]

        result = analyze(field, signal, signal)

        up, down = result.branches
        self.assertEqual((up["start_row"], up["end_row"]), (1, 5))
        self.assertEqual((down["start_row"], down["end_row"]), (7, 10))
        self.assertEqual((up["duplicate_count"], down["duplicate_count"]), (1, 0))
        self.assertEqual(result.longitudinal_up[-1], 15)
        self.assertEqual(result.longitudinal_down[-1], 30)
        self.assertEqual(result.longitudinal_up[result.field == -1].item(), 5)

    def test_tolerance_absorbs_small_field_reversals(self):
        field = [-2, -1, -1.02, 0, 1, 2, 2, 1, 0, -0.5, -0.47, -1, -2]

        with self.assertRaisesRegex(ValueError, "No candidate pairs"):
            analyze(field, field, field)
        result = analyze(field, field, field, field_tolerance=0.05)

        self.assertEqual([branch["direction"] for branch in result.branches], ["up", "down"])
        np.testing.assert_allclose(result.transverse_odd_up, result.field)

    def test_rejects_minor_loop_between_full_sweeps(self):
        field = [-2, 0, 2, 0, 0.5, 0, -2]
        with self.assertRaisesRegex(ValueError, "No candidate pairs"):
            analyze(field, field, field)

    def test_ambiguous_pairs_are_reported_and_can_be_selected(self):
        field = [-2, 0, 2, 0, -2, 0, 2]

        with self.assertRaisesRegex(
            ValueError, r"Candidates: 1 \(up rows 1-3, down rows 3-5\), 2",
        ):
            analyze(field, field, field)
        with self.assertRaisesRegex(ValueError, r"Pair 3 is unavailable.*Candidates:"):
            analyze(field, field, field, pair=3)

        result = analyze(field, field, field, pair=np.int64(2))
        self.assertEqual([branch["direction"] for branch in result.branches], ["down", "up"])
        self.assertEqual([candidate["selected"] for candidate in result.candidates], [False, True])
        self.assertEqual(result.candidates[1]["branch_ids"], [2, 3])

    def test_rejects_invalid_inputs(self):
        for field in ([-1, np.nan, 1], [-1, np.inf, 1], [-1j, 0j, 1j], [[-1, 0, 1]]):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "finite real"):
                analyze(field, [0, 0, 0], [0, 0, 0])
        with self.assertRaisesRegex(ValueError, "same length"):
            analyze([-1, 1, -1], [0, 0], [0, 0, 0])
        with self.assertRaisesRegex(ValueError, "both positive and negative"):
            analyze([0, 1, 0], [0, 0, 0], [0, 0, 0])
        for tolerance in (-1, np.nan, np.inf, [0.1]):
            with self.subTest(tolerance=tolerance), self.assertRaisesRegex(ValueError, "field_tolerance"):
                analyze([-1, 1, -1], [0, 0, 0], [0, 0, 0], field_tolerance=tolerance)
        for pair in (True, 0, -1, 1.0):
            with self.subTest(pair=pair), self.assertRaisesRegex(ValueError, "positive 1-based"):
                analyze([-1, 1, -1], [0, 0, 0], [0, 0, 0], pair=pair)

    def test_linear_interpolation_can_span_large_finite_values(self):
        field = [-1e308, 1e308, 0.0, -1e308]

        result = analyze(field, field, field)

        np.testing.assert_array_equal(result.longitudinal_up, result.field)
        np.testing.assert_array_equal(result.transverse_odd_up, result.field)


if __name__ == "__main__":
    unittest.main()
