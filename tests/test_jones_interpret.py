"""Branch-focused tests for :func:`pypolar.jones.interpret`."""

import io
import unittest
from contextlib import redirect_stdout

import numpy as np
from pypolar import jones


class TestJonesInterpret(unittest.TestCase):
    """Exercise all reachable branches in jones.interpret()."""

    def setUp(self):
        """Use a fixed internal convention for branch-focused raw-vector tests."""
        self._prev = jones.alternate_sign_convention
        jones.use_alternate_convention(True)

    def tearDown(self):
        """Restore global Jones convention state."""
        jones.use_alternate_convention(self._prev)

    def test_interpret_rejects_wrong_length(self):
        """Non-2-element inputs should return a diagnostic string."""
        result = jones.interpret(np.array([1, 2, 3]))
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Malformed input: Jones vector must have exactly two elements")

    def test_interpret_rejects_nonnumeric_and_nonfinite_inputs(self):
        """Malformed value types and non-finite values should be reported."""
        result_bad_type = jones.interpret(["a", "b"])
        self.assertEqual(result_bad_type, "Malformed input: Jones vector must contain two numeric elements")

        result_nonfinite = jones.interpret(np.array([1.0, np.inf]))
        self.assertEqual(result_nonfinite, "Malformed input: Jones vector contains NaN or infinite values")

    def test_interpret_rejects_zero_intensity_vector(self):
        """Zero-field Jones vectors should be reported as unphysical/degenerate."""
        result = jones.interpret(np.array([0.0 + 0.0j, 0.0 + 0.0j]))
        self.assertEqual(result, "Unphysical Jones vector: zero intensity (polarization state is undefined)")

    def test_interpret_linear_branch(self):
        """Linear states should take the early linear-return path."""
        result = jones.interpret(np.array([1.0, 1.0]))
        self.assertIn("Linear polarization at 45.000000 degrees CCW from x - axis", result)

    def test_interpret_equal_magnitudes_right_circular_branch(self):
        """Equal magnitudes with +pi/2 phase difference should report right circular."""
        result = jones.interpret(np.array([1.0 + 0.0j, -1.0j]))
        self.assertIn("Right circular polarization", result)
        self.assertNotIn("Left circular polarization", result)

    def test_interpret_equal_magnitudes_left_circular_branch(self):
        """Equal magnitudes with -pi/2 phase difference should report left circular."""
        result = jones.interpret(np.array([1.0 + 0.0j, 1.0j]))
        self.assertIn("Left circular polarization", result)
        self.assertNotIn("Right circular polarization", result)

    def test_interpret_equal_magnitudes_right_elliptical_branch(self):
        """Equal magnitudes with p1>p2 (not circular) should report right elliptical."""
        result = jones.interpret(np.array([np.exp(0.4j), 1.0 + 0.0j]))
        self.assertIn("Right elliptical polarization", result)
        self.assertIn("rotated", result)
        self.assertIn("ellipticity angle =", result)
        self.assertIn("ellipticity (b/a) =", result)

    def test_interpret_equal_magnitudes_left_elliptical_branch(self):
        """Equal magnitudes with p1<p2 (not circular) should report left elliptical."""
        result = jones.interpret(np.array([1.0 + 0.0j, np.exp(0.4j)]))
        self.assertIn("Left elliptical polarization", result)
        self.assertIn("rotated", result)
        self.assertIn("ellipticity angle =", result)
        self.assertIn("ellipticity (b/a) =", result)

    def test_interpret_unequal_magnitudes_right_non_rotated_branch(self):
        """Unequal magnitudes with exact +pi/2 phase difference should hit right non-rotated branch."""
        result = jones.interpret(np.array([2.0 + 0.0j, -1.0j]))
        self.assertIn("Right elliptical polarization, non - rotated", result)
        self.assertIn("ellipticity angle =", result)
        self.assertIn("ellipticity (b/a) =", result)

    def test_interpret_unequal_magnitudes_right_elliptical_branch(self):
        """Unequal magnitudes with p1>p2 (not non-rotated) should report right elliptical."""
        result = jones.interpret(np.array([2.0 * np.exp(0.3j), 1.0 + 0.0j]))
        self.assertIn("Right elliptical polarization", result)
        self.assertIn("rotated", result)

    def test_interpret_unequal_magnitudes_left_non_rotated_branch(self):
        """Unequal magnitudes with exact -pi/2 phase difference should hit left non-rotated branch."""
        result = jones.interpret(np.array([1.0 + 0.0j, 2.0j]))
        self.assertIn("Left elliptical polarization, non - rotated", result)
        self.assertIn("ellipticity angle =", result)
        self.assertIn("ellipticity (b/a) =", result)

    def test_interpret_unequal_magnitudes_left_elliptical_branch(self):
        """Unequal magnitudes with p1<p2 (not non-rotated) should report left elliptical."""
        result = jones.interpret(np.array([1.0 + 0.0j, 2.0 * np.exp(0.3j)]))
        self.assertIn("Left elliptical polarization", result)
        self.assertIn("rotated", result)

    def test_interpret_has_no_stdout_side_effects(self):
        """Interpret should return text without printing it."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            _ = jones.interpret(np.array([1.0, 1.0]))
        self.assertEqual(buf.getvalue(), "")

    def test_interpret_matches_named_circular_helpers_in_both_conventions(self):
        """Right/left helper constructors should be interpreted consistently."""
        for state in (False, True):
            jones.use_alternate_convention(state)
            self.assertIn("Right circular polarization", jones.interpret(jones.field_right_circular()))
            self.assertIn("Left circular polarization", jones.interpret(jones.field_left_circular()))

    def test_interpret_matches_elliptical_helpers_in_both_conventions(self):
        """Elliptical helper outputs should be interpreted consistently in both conventions."""
        azimuth = np.radians(20)
        ellipticity_angle = np.radians(10)

        for state in (False, True):
            jones.use_alternate_convention(state)
            right = jones.interpret(jones.field_elliptical(azimuth, ellipticity_angle))
            left = jones.interpret(jones.field_elliptical(azimuth, -ellipticity_angle))

            self.assertIn("Right elliptical polarization", right)
            self.assertIn("ellipticity angle =", right)
            self.assertIn("ellipticity (b/a) =", right)

            self.assertIn("Left elliptical polarization", left)
            self.assertIn("ellipticity angle =", left)
            self.assertIn("ellipticity (b/a) =", left)

    def test_interpret_phase_wrapping_and_global_phase_invariance(self):
        """Phase-equivalent circular states should classify identically."""
        jones.use_alternate_convention(True)
        base = jones.interpret(np.array([1.0 + 0.0j, 1.0j]))
        plus_2pi = jones.interpret(np.array([1.0 + 0.0j, np.exp(1j * (np.pi / 2 + 2 * np.pi))]))
        branch_cut = jones.interpret(np.array([-1.0 + 0.0j, -1.0j]))

        self.assertIn("Left circular polarization", base)
        self.assertIn("Left circular polarization", plus_2pi)
        self.assertIn("Left circular polarization", branch_cut)

    def test_interpret_elliptical_invariant_under_global_amplitude_scaling(self):
        """Elliptical classification/metrics should not change under global scaling."""
        jones.use_alternate_convention(True)
        J = np.array([1.0 + 0.0j, np.exp(0.3j)])
        base = jones.interpret(J)

        label = None
        for candidate in ("Right elliptical polarization", "Left elliptical polarization"):
            if candidate in base:
                label = candidate
                break
        self.assertIsNotNone(label)

        base_ellipticity = [line for line in base.splitlines() if "ellipticity (b/a) =" in line][0]
        base_ellipticity_angle = [line for line in base.splitlines() if "ellipticity angle =" in line][0]

        for scale in (0.2, 2.0, 10.0, 2.0 * np.exp(1j * 0.7)):
            summary = jones.interpret(scale * J)
            self.assertIn(label, summary)
            self.assertIn(base_ellipticity, summary)
            self.assertIn(base_ellipticity_angle, summary)


if __name__ == "__main__":
    unittest.main()
