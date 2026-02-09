"""Branch-focused tests for :func:`pypolar.jones.interpret`."""

import io
import unittest
from contextlib import redirect_stdout

import numpy as np
from pypolar import jones


class TestJonesInterpret(unittest.TestCase):
    """Exercise all reachable branches in jones.interpret()."""

    def test_interpret_rejects_wrong_length(self):
        """Non-2-element inputs should return a diagnostic string and print it."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = jones.interpret(np.array([1, 2, 3]))
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Malformed input: Jones vector must have exactly two elements")
        self.assertIn("Malformed input: Jones vector must have exactly two elements", buf.getvalue())

    def test_interpret_rejects_nonnumeric_and_nonfinite_inputs(self):
        """Malformed value types and non-finite values should be reported."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            result_bad_type = jones.interpret(["a", "b"])
        self.assertEqual(result_bad_type, "Malformed input: Jones vector must contain two numeric elements")
        self.assertIn("Malformed input: Jones vector must contain two numeric elements", buf.getvalue())

        buf = io.StringIO()
        with redirect_stdout(buf):
            result_nonfinite = jones.interpret(np.array([1.0, np.inf]))
        self.assertEqual(result_nonfinite, "Malformed input: Jones vector contains NaN or infinite values")
        self.assertIn("Malformed input: Jones vector contains NaN or infinite values", buf.getvalue())

    def test_interpret_rejects_zero_intensity_vector(self):
        """Zero-field Jones vectors should be reported as unphysical/degenerate."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = jones.interpret(np.array([0.0 + 0.0j, 0.0 + 0.0j]))
        self.assertEqual(result, "Unphysical Jones vector: zero intensity (polarization state is undefined)")
        self.assertIn("Unphysical Jones vector: zero intensity (polarization state is undefined)", buf.getvalue())

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

    def test_interpret_equal_magnitudes_left_elliptical_branch(self):
        """Equal magnitudes with p1<p2 (not circular) should report left elliptical."""
        result = jones.interpret(np.array([1.0 + 0.0j, np.exp(0.4j)]))
        self.assertIn("Left elliptical polarization", result)
        self.assertIn("rotated", result)

    def test_interpret_unequal_magnitudes_right_non_rotated_branch(self):
        """Unequal magnitudes with exact +pi/2 phase difference should hit right non-rotated branch."""
        result = jones.interpret(np.array([2.0 + 0.0j, -1.0j]))
        self.assertIn("Right elliptical polarization, non - rotated", result)

    def test_interpret_unequal_magnitudes_right_elliptical_branch(self):
        """Unequal magnitudes with p1>p2 (not non-rotated) should report right elliptical."""
        result = jones.interpret(np.array([2.0 * np.exp(0.3j), 1.0 + 0.0j]))
        self.assertIn("Right elliptical polarization", result)
        self.assertIn("rotated", result)

    def test_interpret_unequal_magnitudes_left_non_rotated_branch(self):
        """Unequal magnitudes with exact -pi/2 phase difference should hit left non-rotated branch."""
        result = jones.interpret(np.array([1.0 + 0.0j, 2.0j]))
        self.assertIn("Left circular polarization, non - rotated", result)

    def test_interpret_unequal_magnitudes_left_elliptical_branch(self):
        """Unequal magnitudes with p1<p2 (not non-rotated) should report left elliptical."""
        result = jones.interpret(np.array([1.0 + 0.0j, 2.0 * np.exp(0.3j)]))
        self.assertIn("Left elliptical polarization", result)
        self.assertIn("rotated", result)


if __name__ == "__main__":
    unittest.main()
