"""Unit tests for Jones vector operations and conversions."""

import unittest
import numpy as np
from pypolar import jones
from pypolar import mueller


class TestBasic(unittest.TestCase):
    """Test basic Jones vector creation, intensity, phase, and Stokes conversion."""

    def test_field_linear_H(self):
        """Test that horizontal polarization matches linear at 0 degrees."""
        H = jones.field_horizontal()
        J = jones.field_linear(0)
        for pair in zip(H, J):
            self.assertAlmostEqual(pair[0], pair[1])

    def test_field_linear_V(self):
        """Test that vertical polarization matches linear at 90 degrees."""
        V = jones.field_vertical()
        J = jones.field_linear(np.pi / 2)
        for pair in zip(V, J):
            self.assertAlmostEqual(pair[0], pair[1])

    def test_field_linear_multi(self):
        """Test that linear polarization works with multiple angles."""
        N = 3
        angles = np.linspace(0, np.pi / 2, N)
        #        H = jones.field_horizontal()
        #        V = jones.field_vertical()
        J = jones.field_linear(angles)
        self.assertEqual(len(J), N)

    def test_intensity_scalar(self):
        """Test intensity calculation for scalar Jones vectors."""
        J = jones.field_right_circular()
        II = jones.intensity(J)
        self.assertAlmostEqual(II, 1)
        J = jones.field_horizontal()
        II = jones.intensity(J)
        self.assertAlmostEqual(II, 1)

    def test_intensity_array(self):
        """Test intensity calculation for array of Jones vectors."""
        N = 3
        angles = np.linspace(0, np.pi / 2, N)
        J = jones.field_linear(angles)
        II = jones.intensity(J)
        self.assertEqual(len(II), N)
        for intensity in II:
            self.assertAlmostEqual(intensity, 1)

    def test_phase_scalar(self):
        """Test phase calculation for scalar Jones vectors."""
        J = jones.field_left_circular()
        phi = jones.phase(J)
        self.assertAlmostEqual(phi, -np.pi / 2)
        J = jones.field_right_circular()
        phi = jones.phase(J)
        self.assertAlmostEqual(phi, np.pi / 2)
        J = jones.field_horizontal()
        phi = jones.phase(J)
        self.assertEqual(phi, 0)

    def test_phase_array(self):
        """Test phase calculation for array of Jones vectors."""
        N = 3
        angles = np.linspace(0, np.pi / 2, N)
        J = jones.field_linear(angles)
        phi = jones.phase(J)
        self.assertEqual(len(phi), N)
        for p in phi:
            self.assertAlmostEqual(p, 0)

    def test_to_stokes_scalar(self):
        """Test Jones to Stokes conversion for scalar input."""
        J = jones.field_left_circular()
        S = mueller.stokes_left_circular()
        SS = jones.jones_to_stokes(J)
        for pair in zip(S, SS):
            self.assertAlmostEqual(pair[0], pair[1])

    def test_to_stokes_array(self):
        """Test Jones to Stokes conversion for array input."""
        N = 3
        angles = np.linspace(0, np.pi / 2, N)
        J = jones.field_linear(angles)
        S = jones.jones_to_stokes(J)
        n, m = S.shape
        self.assertEqual(n, N)
        self.assertEqual(m, 4)


if __name__ == "__main__":
    unittest.main()
