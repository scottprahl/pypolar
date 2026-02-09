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
        cases = [(1.5, np.radians(30)), (1.5 - 0.1j, np.radians(30)), (1.5, np.radians(60))]
        for m, theta in cases:
            self.assertTrue(
                np.allclose(
                    mueller.op_fresnel_reflection(m, theta),
                    jones.jones_op_to_mueller_op(jones.op_fresnel_reflection(m, theta)),
                )
            )
            self.assertTrue(
                np.allclose(
                    mueller.op_fresnel_transmission(m, theta),
                    jones.jones_op_to_mueller_op(jones.op_fresnel_transmission(m, theta)),
                )
            )


class TestMuellerStokesConstructors(unittest.TestCase):
    """Test additional stokes constructors and edge paths."""

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
        self.assertAlmostEqual(mueller.ellipse_ellipticity(H), 0.0)
        A, B = mueller.ellipse_axes(H)
        self.assertAlmostEqual(A, 1.0)
        self.assertAlmostEqual(B, 0.0)

        V = mueller.stokes_vertical()
        self.assertAlmostEqual(mueller.ellipse_orientation(V), np.pi / 2)

        R = mueller.stokes_right_circular()
        self.assertAlmostEqual(mueller.ellipse_ellipticity(R), np.pi / 4)
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

    def test_interpret_placeholder_and_error_path(self):
        """Interpret currently returns placeholder text and reports malformed inputs."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            s = mueller.interpret(mueller.stokes_horizontal())
        self.assertEqual(s, "not implemented yet")
        out = buf.getvalue()
        self.assertIn("I =", out)
        self.assertIn("Q =", out)
        self.assertIn("U =", out)
        self.assertIn("V =", out)

        buf = io.StringIO()
        with redirect_stdout(buf):
            bad = mueller.interpret(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(bad, 0)
        self.assertIn("Stokes vector must have four real elements", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
