"""Unit tests for Jones vector operations and conversions."""

import unittest
import numpy as np
from pypolar import jones
from pypolar import mueller
from pypolar import fresnel


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

    def test_phase_global_phase_invariance(self):
        """Test phase invariance under global phase changes and signed-zero representations."""
        a = np.array([1.0, 1j])
        b = np.array([-1.0, -1j])

        self.assertAlmostEqual(jones.phase(a), np.pi / 2)
        self.assertAlmostEqual(jones.phase(-a), np.pi / 2)
        self.assertAlmostEqual(jones.phase(b), np.pi / 2)

    def test_ellipticity_angle_signed_zero_representation(self):
        """Test ellipticity invariance for equivalent vectors with different signed zeros."""
        a = np.array([1.0, 1j])
        b = np.array([-1.0, -1j])

        for J in (a, -a, b):
            self.assertAlmostEqual(jones.ellipticity_angle(J), np.pi / 4, places=12)
            self.assertAlmostEqual(jones.ellipticity(J), 1.0, places=12)

    def test_ellipticity_angle_consistent_for_c_and_d(self):
        """Test that nearby equivalent vectors produce the same ellipticity sign."""
        c = np.array([-1.0 + 0.1j, -1j])
        d = np.array([-1.0 - 0.1j, -1j])

        self.assertAlmostEqual(jones.ellipticity_angle(c), jones.ellipticity_angle(d), places=12)
        self.assertAlmostEqual(jones.ellipticity(c), jones.ellipticity(d), places=12)
        self.assertGreater(jones.ellipticity_angle(c), 0)

    def test_ellipticity_global_phase_invariance(self):
        """Test ellipticity invariance under arbitrary global phase shifts."""
        J = np.array([-1.0 + 0.1j, -1j])
        expected_angle = jones.ellipticity_angle(J)
        expected_ellipticity = jones.ellipticity(J)

        for theta in np.linspace(-np.pi, np.pi, 9):
            JJ = J * np.exp(1j * theta)
            self.assertAlmostEqual(jones.ellipticity_angle(JJ), expected_angle, places=12)
            self.assertAlmostEqual(jones.ellipticity(JJ), expected_ellipticity, places=12)

    def test_fresnel_transmission_operator_matches_field_amplitudes(self):
        """Jones Fresnel transmission operator should return Fresnel field amplitudes."""
        m = 1.5
        theta = np.radians(45)
        J = jones.op_fresnel_transmission(m, theta)
        self.assertAlmostEqual(J[0, 0], fresnel.t_par_amplitude(m, theta), places=12)
        self.assertAlmostEqual(J[1, 1], fresnel.t_per_amplitude(m, theta), places=12)

    def test_fresnel_transmission_operator_matches_field_amplitudes_under_tir(self):
        """Jones Fresnel transmission operator should preserve complex TIR amplitudes."""
        m = 0.8
        theta = np.radians(70)
        J = jones.op_fresnel_transmission(m, theta)
        self.assertAlmostEqual(J[0, 0], fresnel.t_par_amplitude(m, theta), places=12)
        self.assertAlmostEqual(J[1, 1], fresnel.t_per_amplitude(m, theta), places=12)
        self.assertGreater(abs(J[0, 0]), 0)
        self.assertGreater(abs(J[1, 1]), 0)

    def test_fresnel_operators_with_nonunity_incident_index(self):
        """Jones Fresnel operators should support n_i != 1."""
        m = 1.0
        n_i = 1.5
        theta = np.radians(30)

        R = jones.op_fresnel_reflection(m, theta, n_i=n_i)
        T = jones.op_fresnel_transmission(m, theta, n_i=n_i)

        self.assertAlmostEqual(R[0, 0], fresnel.r_par_amplitude(m, theta, n_i=n_i), places=12)
        self.assertAlmostEqual(R[1, 1], fresnel.r_per_amplitude(m, theta, n_i=n_i), places=12)
        self.assertAlmostEqual(T[0, 0], fresnel.t_par_amplitude(m, theta, n_i=n_i), places=12)
        self.assertAlmostEqual(T[1, 1], fresnel.t_per_amplitude(m, theta, n_i=n_i), places=12)


if __name__ == "__main__":
    unittest.main()
