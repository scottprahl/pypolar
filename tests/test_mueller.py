"""Additional unit tests for numeric Mueller/Stokes operators and edge cases."""

import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from pypolar import fresnel
from pypolar import jones
from pypolar import mueller


class TestMuellerOperators(unittest.TestCase):
    """Test Mueller matrix operators beyond the basic stokes tests."""

    def test_basic_component_operators(self):
        """Attenuator, mirror, rotation, and wave-plate wrapper behavior."""
        t = 0.23
        self.assertTrue(np.allclose(mueller.op_attenuator(t), t * np.eye(4)))
        self.assertTrue(np.allclose(mueller.op_attenuator(0.01), 0.01 * np.eye(4)))
        self.assertTrue(np.allclose(mueller.op_mirror(), np.diag([1, 1, -1, -1])))

        theta = 0.31
        rot = mueller.op_rotation(theta)
        self.assertTrue(np.allclose(rot @ mueller.op_rotation(-theta), np.eye(4)))

        self.assertTrue(np.allclose(mueller.op_quarter_wave_plate(theta), mueller.op_retarder(theta, np.pi / 2)))
        self.assertTrue(np.allclose(mueller.op_half_wave_plate(theta), mueller.op_retarder(theta, np.pi)))

    def test_linear_polarizer_action(self):
        """Linear polarizer should pass aligned states and block orthogonal states."""
        H = mueller.stokes_horizontal()
        V = mueller.stokes_vertical()
        P0 = mueller.op_linear_polarizer(0)
        P90 = mueller.op_linear_polarizer(np.pi / 2)
        self.assertTrue(np.allclose(P0 @ H, H))
        self.assertTrue(np.allclose(P0 @ V, np.zeros(4)))
        self.assertTrue(np.allclose(P90 @ V, V))
        self.assertTrue(np.allclose(P90 @ H, np.zeros(4)))

    def test_fresnel_operators_at_normal_incidence(self):
        """Fresnel Mueller operators should match expected normal-incidence coefficients."""
        m = 1.5
        theta = 0.0
        R = fresnel.R_par(m, theta)

        M_ref = mueller.op_fresnel_reflection(m, theta)
        self.assertTrue(np.allclose(M_ref, np.diag([R, R, -R, -R])))

        M_trn = mueller.op_fresnel_transmission(m, theta)
        M_trn_expected = jones.jones_op_to_mueller_op(jones.op_fresnel_transmission(m, theta))
        self.assertTrue(np.allclose(M_trn, M_trn_expected))

    def test_fresnel_transmission_complex_index_is_real_matrix(self):
        """Transmission Mueller matrix entries should remain real-valued."""
        M = mueller.op_fresnel_transmission(1.5 - 0.1j, np.radians(30))
        self.assertTrue(np.isrealobj(M))

    def test_fresnel_operators_match_jones_conversion(self):
        """Numeric Fresnel Mueller operators should match Jones-to-Mueller conversion."""
        cases = [
            (1.5, np.radians(30), 1.0),
            (1.5 - 0.1j, np.radians(30), 1.0),
            (1.5, np.radians(60), 1.0),
            (1.33, np.radians(45), 1.5),
        ]
        for m, theta, n_i in cases:
            self.assertTrue(
                np.allclose(
                    mueller.op_fresnel_reflection(m, theta, n_i=n_i),
                    jones.jones_op_to_mueller_op(jones.op_fresnel_reflection(m, theta, n_i=n_i)),
                )
            )
            self.assertTrue(
                np.allclose(
                    mueller.op_fresnel_transmission(m, theta, n_i=n_i),
                    jones.jones_op_to_mueller_op(jones.op_fresnel_transmission(m, theta, n_i=n_i)),
                )
            )


class TestMuellerStokesConstructors(unittest.TestCase):
    """Test additional stokes constructors and edge paths."""

    def test_stokes_components_scalar_and_array(self):
        """Explicit component constructor should support scalar and broadcasted array inputs."""
        S = mueller.stokes_components(1.0, 0.2, -0.3, 0.4)
        self.assertTrue(np.allclose(S, np.array([1.0, 0.2, -0.3, 0.4])))

        I = np.array([1.0, 2.0])  # noqa: E741
        Q = np.array([0.0, 0.1])
        U = 0.5
        V = np.array([0.0, -0.2])
        SS = mueller.stokes_components(I, Q, U, V)
        expected = np.array([[1.0, 0.0, 0.5, 0.0], [2.0, 0.1, 0.5, -0.2]])
        self.assertEqual(SS.shape, (2, 4))
        self.assertTrue(np.allclose(SS, expected))

    def test_stokes_ellipsometry_scalar_and_array(self):
        """Ellipsometry constructor should support scalar and vector inputs."""
        S = mueller.stokes_ellipsometry(1.0, 0.0)
        self.assertTrue(np.allclose(S, np.array([1.0, 0.0, 1.0, 0.0])))

        tanpsi = np.array([1.0, 2.0, 0.5])
        Delta = np.array([0.0, 0.3, -0.7])
        SS = mueller.stokes_ellipsometry(tanpsi, Delta)
        self.assertEqual(SS.shape, (3, 4))
        self.assertTrue(np.allclose(SS[:, 0], np.ones(3)))

    def test_stokes_elliptical_scalar_and_array(self):
        """Elliptical constructor should support scalar and array DOP paths."""
        self.assertTrue(np.allclose(mueller.stokes_elliptical(0, 0, 0), mueller.stokes_unpolarized()))
        self.assertTrue(np.allclose(mueller.stokes_elliptical(1, 0, 0), mueller.stokes_horizontal()))

        dop = np.array([0.0, 1.0])
        S = mueller.stokes_elliptical(dop, 0.0, 0.0)
        self.assertEqual(S.shape, (2, 4))
        self.assertTrue(np.allclose(S[0], mueller.stokes_unpolarized()))
        self.assertTrue(np.allclose(S[1], mueller.stokes_horizontal()))

    def test_degree_of_polarization_zero_intensity_edge(self):
        """Zero-intensity stokes vectors should return zero DOP."""
        self.assertEqual(mueller.degree_of_polarization(np.array([0.0, 1.0, 2.0, 3.0])), 0)
        SS = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 0.0, 0.0]])
        dop = mueller.degree_of_polarization(SS)
        self.assertTrue(np.allclose(dop, np.array([0.0, 1.0])))

    def test_ellipse_helpers_known_states(self):
        """Orientation/ellipticity/axes helpers should match known canonical states."""
        H = mueller.stokes_horizontal()
        self.assertAlmostEqual(mueller.ellipse_orientation(H), 0.0)
        self.assertAlmostEqual(mueller.ellipticity_angle(H), 0.0)
        A, B = mueller.ellipse_axes(H)
        self.assertAlmostEqual(A, 1.0)
        self.assertAlmostEqual(B, 0.0)

        V = mueller.stokes_vertical()
        self.assertAlmostEqual(mueller.ellipse_orientation(V), np.pi / 2)

        R = mueller.stokes_right_circular()
        self.assertAlmostEqual(mueller.ellipticity_angle(R), np.pi / 4)
        A, B = mueller.ellipse_axes(R)
        self.assertAlmostEqual(A, np.sqrt(0.5))
        self.assertAlmostEqual(B, np.sqrt(0.5))


class TestMuellerConversions(unittest.TestCase):
    """Test conversion helpers and edge handling."""

    def test_stokes_to_jones_edges(self):
        """Cover scalar edge paths, shape validation, and alternate-sign branch."""
        self.assertTrue(np.allclose(mueller.stokes_to_jones(np.array([0.0, 0.0, 0.0, 0.0])), np.array([0.0, 0.0])))
        self.assertTrue(np.allclose(mueller.stokes_to_jones(mueller.stokes_vertical()), np.array([0.0, 1.0])))

        bad = np.ones((2, 3))
        self.assertIsNone(mueller.stokes_to_jones(bad))

        prev = jones.alternate_sign_convention
        S = mueller.stokes_right_circular()
        try:
            jones.use_alternate_convention(False)
            J1 = mueller.stokes_to_jones(S)
            jones.use_alternate_convention(True)
            J2 = mueller.stokes_to_jones(S)
            self.assertTrue(np.allclose(J2, np.conjugate(J1)))
        finally:
            jones.use_alternate_convention(prev)

    def test_mueller_to_jones_known_matrices(self):
        """Known Mueller matrices should map to expected Jones matrices."""
        JJ = mueller.mueller_to_jones(np.eye(4))
        self.assertTrue(np.allclose(JJ, np.eye(2)))

        MM = np.diag([1, 1, -1, -1])
        JJ = mueller.mueller_to_jones(MM)
        self.assertAlmostEqual(JJ[0, 0], 1.0)
        self.assertAlmostEqual(JJ[0, 1], 0.0)
        self.assertAlmostEqual(JJ[1, 0], 0.0)
        self.assertAlmostEqual(JJ[1, 1], -1.0)

    def test_interpret_descriptive_output_and_error_path(self):
        """Interpret should return a useful summary string and report malformed inputs."""
        s = mueller.interpret(mueller.stokes_horizontal())
        self.assertIsInstance(s, str)
        self.assertIn("I = 1.000", s)
        self.assertIn("Q = 1.000", s)
        self.assertIn("Degree of polarization = 1.000", s)
        self.assertIn("Fully polarized light", s)
        self.assertIn("Linear polarization at 0.0 degrees CCW from x-axis", s)

        bad = mueller.interpret(np.array([1.0, 2.0, 3.0]))
        self.assertIsInstance(bad, str)
        self.assertIn("Malformed input:", bad)

    def test_interpret_classification_branches(self):
        """Interpret should classify unpolarized, circular, elliptical, and dark states."""
        s_un = mueller.interpret(mueller.stokes_unpolarized())
        s_rcp = mueller.interpret(mueller.stokes_right_circular())
        s_lcp = mueller.interpret(mueller.stokes_left_circular())
        s_part = mueller.interpret(mueller.stokes_elliptical(0.6, 0.3, 0.2))
        s_dark = mueller.interpret(np.zeros(4))

        self.assertIn("Unpolarized light", s_un)
        self.assertIn("Right circular polarization", s_rcp)
        self.assertIn("Left circular polarization", s_lcp)
        self.assertIn("Partially polarized:", s_part)
        self.assertIn("Right elliptical polarization", s_part)
        self.assertIn("ellipticity angle =", s_part)
        self.assertIn("No light (zero intensity)", s_dark)

    def test_interpret_rejects_unphysical_stokes_vectors(self):
        """Interpret should explicitly report physically impossible Stokes vectors."""
        s_bad = mueller.interpret(np.array([1.0, 2.0, 0.0, 0.0]))
        self.assertIsInstance(s_bad, str)
        self.assertIn("Physically impossible Stokes vector", s_bad)
        self.assertIn("exceeds I", s_bad)

        bad_complex = mueller.interpret(np.array([1.0 + 1j, 0.0, 0.0, 0.0]))
        self.assertIsInstance(bad_complex, str)
        self.assertIn("Malformed input:", bad_complex)
        self.assertIn("non-zero imaginary parts", bad_complex)

    def test_interpret_mueller_matrix_admissibility_checks(self):
        """Interpret should run basic admissibility diagnostics for 4x4 Mueller input."""
        s_ok = mueller.interpret(np.eye(4))
        self.assertIn("Detected 4x4 Mueller matrix input.", s_ok)
        self.assertIn("No violations found in necessary checks.", s_ok)

        M_bad = np.eye(4)
        M_bad[0, 1] = 2.0
        s_bad = mueller.interpret(M_bad)
        self.assertIn("WARNING: matrix is not physically admissible", s_bad)
        self.assertIn("Diattenuation > 1", s_bad)

    def test_interpret_has_no_stdout_side_effects(self):
        """Interpret should return text without printing it."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            _ = mueller.interpret(mueller.stokes_horizontal())
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
