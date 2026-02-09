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

    def test_use_alternate_convention_for_circular_fields(self):
        """Alternate convention should conjugate circular basis fields."""
        previous = jones.alternate_sign_convention
        try:
            jones.use_alternate_convention(False)
            right_default = jones.field_right_circular()
            left_default = jones.field_left_circular()

            jones.use_alternate_convention(True)
            self.assertTrue(np.allclose(jones.field_right_circular(), np.conjugate(right_default)))
            self.assertTrue(np.allclose(jones.field_left_circular(), np.conjugate(left_default)))
        finally:
            jones.use_alternate_convention(previous)

    def test_linear_polarizer_and_rotation_operators(self):
        """Linear polarizer should project and rotation should be invertible."""
        H = jones.field_horizontal()
        V = jones.field_vertical()
        P0 = jones.op_linear_polarizer(0)
        P90 = jones.op_linear_polarizer(np.pi / 2)
        self.assertTrue(np.allclose(P0 @ H, H))
        self.assertTrue(np.allclose(P0 @ V, np.array([0, 0])))
        self.assertTrue(np.allclose(P90 @ V, V))
        self.assertTrue(np.allclose(P90 @ H, np.array([0, 0])))

        theta = 0.37
        R = jones.op_rotation(theta)
        self.assertTrue(np.allclose(R @ jones.op_rotation(-theta), np.eye(2)))

    def test_basic_component_operators(self):
        """Attenuator, mirror, and wave-plate wrappers should behave as defined."""
        A = jones.op_attenuator(0.25)
        self.assertTrue(np.allclose(A, np.array([[0.5, 0.0], [0.0, 0.5]])))

        t_nd = 0.01
        self.assertTrue(np.allclose(jones.op_neutral_density(t_nd), jones.op_attenuator(t_nd)))
        self.assertTrue(np.allclose(jones.op_neutral_density_filter(t_nd), jones.op_attenuator(t_nd)))
        self.assertTrue(np.allclose(jones.op_neutral_density_filter(t_nd), np.array([[0.1, 0.0], [0.0, 0.1]])))

        M = jones.op_mirror()
        self.assertTrue(np.allclose(M @ jones.field_horizontal(), jones.field_horizontal()))
        self.assertTrue(np.allclose(M @ jones.field_vertical(), -jones.field_vertical()))

        theta = 0.22
        self.assertTrue(np.allclose(jones.op_quarter_wave_plate(theta), jones.op_retarder(theta, np.pi / 2)))
        self.assertTrue(np.allclose(jones.op_half_wave_plate(theta), jones.op_retarder(theta, np.pi)))

    def test_field_ellipsometry_parameterization(self):
        """Ellipsometry constructor should preserve tanpsi and Delta convention."""
        tanpsi = 2.0
        Delta = 0.35
        J = jones.field_ellipsometry(tanpsi, Delta)
        self.assertAlmostEqual(abs(J[0] / J[1]), tanpsi, places=12)
        self.assertAlmostEqual(jones.phase(J), np.angle(np.exp(-1j * Delta)), places=12)

    def test_field_elliptical_roundtrip(self):
        """Elliptical constructor should preserve azimuth, ellipticity, phase, and intensity."""
        azimuth = np.radians(20)
        ell = np.radians(10)
        phi_x = 0.7
        E_0 = 2.3
        J = jones.field_elliptical(azimuth, ell, phi_x=phi_x, E_0=E_0)

        self.assertAlmostEqual(jones.intensity(J), E_0**2, places=12)
        self.assertAlmostEqual(jones.ellipse_azimuth(J), azimuth, places=12)
        self.assertAlmostEqual(jones.ellipticity_angle(J), ell, places=12)
        self.assertAlmostEqual(np.angle(J[0]), phi_x, places=12)

    def test_field_components_constructor_and_convention(self):
        """Raw-component constructor should preserve values and follow sign convention toggles."""
        Ex = 1 + 2j
        Ey = 3 - 4j
        prev = jones.alternate_sign_convention
        try:
            jones.use_alternate_convention(False)
            J = jones.field_components(Ex, Ey)
            self.assertTrue(np.allclose(J, np.array([Ex, Ey])))

            jones.use_alternate_convention(True)
            J_alt = jones.field_components(Ex, Ey)
            self.assertTrue(np.allclose(J_alt, np.conjugate(np.array([Ex, Ey]))))
        finally:
            jones.use_alternate_convention(prev)

    def test_normalize_and_ratio_utilities(self):
        """Normalization and ratio helpers should handle edge and nominal cases."""
        z = np.array([0 + 0j, 0 + 0j])
        self.assertTrue(np.array_equal(jones.normalize_vector(z), z))

        J = np.array([3 + 4j, 0])
        JJ = jones.normalize_vector(J)
        self.assertAlmostEqual(np.linalg.norm(JJ), 1.0, places=12)

        H = jones.field_horizontal()
        V = jones.field_vertical()
        self.assertAlmostEqual(jones.amplitude_ratio(H), 0.0, places=12)
        self.assertTrue(np.isinf(jones.amplitude_ratio(V)))
        self.assertAlmostEqual(jones.amplitude_ratio_angle(H), 0.0, places=12)
        self.assertAlmostEqual(jones.amplitude_ratio_angle(V), np.pi / 2, places=12)

    def test_polarization_variable_and_poincare_point(self):
        """Polarization variable and Poincare helper should match their definitions."""
        J = np.array([2 + 0j, 0 + 2j])
        self.assertAlmostEqual(jones.polarization_variable(J), 1j, places=12)

        latitude, longitude = jones.poincare_point(J)
        self.assertAlmostEqual(longitude, 2 * jones.ellipse_azimuth(J), places=12)
        a, b = jones.ellipse_axes(J)
        self.assertAlmostEqual(latitude, 2 * np.arctan2(b, a), places=12)

    def test_ellipse_azimuth2_for_known_states(self):
        """Secondary azimuth helper should match expected values for simple states."""
        self.assertAlmostEqual(jones.ellipse_azimuth2(jones.field_linear(np.pi / 6)), 0.0, places=12)
        self.assertAlmostEqual(jones.ellipse_azimuth2(jones.field_right_circular()), np.pi / 4, places=12)

    def test_canonical_ellipse_aliases(self):
        """Canonical ellipse naming aliases should match legacy Jones helper outputs."""
        J = jones.field_elliptical(np.radians(20), np.radians(10))
        self.assertAlmostEqual(jones.ellipse_orientation(J), jones.ellipse_azimuth(J), places=12)
        self.assertAlmostEqual(jones.ellipse_ellipticity(J), jones.ellipticity_angle(J), places=12)

    def test_interpret_and_normalize_passthrough(self):
        """Interpret should classify a simple linear state and normalize() should be pass-through."""
        description = jones.interpret(np.array([1.0, 1.0]))
        self.assertIn("Linear polarization", description)

        J = np.array([1 + 2j, 3 - 4j])
        self.assertTrue(np.allclose(jones.normalize(J), J))

    def test_jones_op_to_mueller_matches_known_operators(self):
        """Jones-to-Mueller conversion should match direct Mueller operators."""
        theta = 0.31
        M_lp = jones.jones_op_to_mueller_op(jones.op_linear_polarizer(theta))
        self.assertTrue(np.allclose(M_lp, mueller.op_linear_polarizer(theta)))

        M_qwp = jones.jones_op_to_mueller_op(jones.op_quarter_wave_plate(theta))
        self.assertTrue(np.allclose(M_qwp, mueller.op_quarter_wave_plate(theta)))

        M_hwp = jones.jones_op_to_mueller_op(jones.op_half_wave_plate(theta))
        self.assertTrue(np.allclose(M_hwp, mueller.op_half_wave_plate(theta)))

    def test_jones_to_stokes_wrong_shape_returns_none(self):
        """Jones-to-Stokes should return None for inputs that are not n x 2."""
        bad = np.ones((2, 3))
        self.assertIsNone(jones.jones_to_stokes(bad))


if __name__ == "__main__":
    unittest.main()
