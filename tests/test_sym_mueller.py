"""Unit tests for symbolic Mueller/Stokes vector operations."""

import unittest

import numpy as np
import sympy
from pypolar import jones
from pypolar import sym_mueller


class TestSymMuellerConstructors(unittest.TestCase):
    """Test symbolic Stokes constructors and interpretation helpers."""

    def test_stokes_constructors(self):
        """Canonical symbolic constructors should match expected vectors."""
        self.assertEqual(sym_mueller.stokes_horizontal(), sympy.Matrix([1, 1, 0, 0]))
        self.assertEqual(sym_mueller.stokes_vertical(), sympy.Matrix([1, -1, 0, 0]))
        self.assertEqual(sym_mueller.stokes_right_circular(), sympy.Matrix([1, 0, 0, 1]))
        self.assertEqual(sym_mueller.stokes_left_circular(), sympy.Matrix([1, 0, 0, -1]))
        self.assertEqual(sym_mueller.stokes_unpolarized(), sympy.Matrix([1, 0, 0, 0]))
        self.assertEqual(sym_mueller.stokes_linear(sympy.pi / 4), sympy.Matrix([1, 0, 1, 0]))

        S_ellips = sym_mueller.stokes_ellipsometry(1, 0)
        self.assertEqual(sympy.simplify(S_ellips), sympy.Matrix([1, 0, 1, 0]))

        S_ellip = sym_mueller.stokes_elliptical(1, 0, 0)
        self.assertEqual(sympy.simplify(S_ellip), sym_mueller.stokes_horizontal())

    def test_intensity_and_dop(self):
        """Intensity and DOP should evaluate correctly for common states."""
        S = sym_mueller.stokes_left_circular()
        self.assertEqual(sym_mueller.intensity(S), 1)
        self.assertEqual(sympy.simplify(sym_mueller.degree_of_polarization(S)), 1)
        self.assertEqual(sympy.simplify(sym_mueller.degree_of_polarization(sym_mueller.stokes_unpolarized())), 0)

        S = sympy.Matrix([1, sympy.Rational(1, 2), 0, 0])
        self.assertEqual(sympy.simplify(sym_mueller.degree_of_polarization(S)), sympy.Rational(1, 2))

    def test_ellipse_helpers_known_states(self):
        """Orientation, ellipticity, and axes should match canonical states."""
        H = sym_mueller.stokes_horizontal()
        self.assertEqual(sympy.simplify(sym_mueller.ellipse_orientation(H)), 0)
        self.assertEqual(sympy.simplify(sym_mueller.ellipse_ellipticity(H)), 0)
        A, B = sym_mueller.ellipse_axes(H)
        self.assertEqual(sympy.simplify(A), 1)
        self.assertEqual(sympy.simplify(B), 0)

        V = sym_mueller.stokes_vertical()
        self.assertEqual(sympy.simplify(sym_mueller.ellipse_orientation(V) - sympy.pi / 2), 0)

        R = sym_mueller.stokes_right_circular()
        self.assertEqual(sympy.simplify(sym_mueller.ellipse_ellipticity(R) - sympy.pi / 4), 0)


class TestSymMuellerOperators(unittest.TestCase):
    """Test symbolic Mueller matrix operators."""

    def test_basic_operator_identities(self):
        """Mirror, attenuator, rotation, and waveplate wrappers should match definitions."""
        self.assertEqual(
            sym_mueller.op_mirror(),
            sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
        )
        self.assertEqual(sym_mueller.op_attenuator(sympy.Rational(1, 3)), sympy.Rational(1, 3) * sympy.eye(4))

        theta = sympy.Symbol("theta", real=True)
        self.assertEqual(sympy.simplify(sym_mueller.op_rotation(theta) * sym_mueller.op_rotation(-theta)), sympy.eye(4))

        theta2 = sympy.Symbol("theta2", real=True)
        self.assertEqual(
            sympy.simplify(sym_mueller.op_quarter_wave_plate(theta2) - sym_mueller.op_retarder(theta2, sympy.pi / 2)),
            sympy.zeros(4),
        )
        self.assertEqual(
            sympy.simplify(sym_mueller.op_half_wave_plate(theta2) - sym_mueller.op_retarder(theta2, sympy.pi)),
            sympy.zeros(4),
        )

    def test_linear_polarizer_at_zero(self):
        """A zero-angle polarizer should transmit horizontal Stokes state."""
        P0 = sym_mueller.op_linear_polarizer(0)
        self.assertEqual(
            sympy.simplify(P0 * sym_mueller.stokes_horizontal() - sym_mueller.stokes_horizontal()), sympy.zeros(4, 1)
        )

    def test_fresnel_operators_match_numeric(self):
        """Symbolic Fresnel operators should match numeric Mueller operators on evaluation."""
        m = 1.5 - 0.1j
        theta = np.radians(30)

        R_sym = np.array(sym_mueller.op_fresnel_reflection(m, theta).evalf(), dtype=complex)
        R_num = jones.jones_op_to_mueller_op(jones.op_fresnel_reflection(m, theta))
        self.assertTrue(np.allclose(R_sym.imag, 0, atol=1e-12))
        self.assertTrue(np.allclose(R_sym.real, R_num))

        T_sym = np.array(sym_mueller.op_fresnel_transmission(m, theta).evalf(), dtype=complex)
        T_num = jones.jones_op_to_mueller_op(jones.op_fresnel_transmission(m, theta))
        self.assertTrue(np.allclose(T_sym.imag, 0, atol=1e-12))
        self.assertTrue(np.allclose(T_sym.real, T_num))

    def test_fresnel_operators_with_nonunity_incident_index(self):
        """Symbolic Fresnel Mueller operators should support n_i != 1."""
        m = 1.33
        theta = np.radians(45)
        n_i = 1.5

        R_sym = np.array(sym_mueller.op_fresnel_reflection(m, theta, n_i=n_i).evalf(), dtype=complex)
        R_num = jones.jones_op_to_mueller_op(jones.op_fresnel_reflection(m, theta, n_i=n_i))
        self.assertTrue(np.allclose(R_sym.imag, 0, atol=1e-12))
        self.assertTrue(np.allclose(R_sym.real, R_num))

        T_sym = np.array(sym_mueller.op_fresnel_transmission(m, theta, n_i=n_i).evalf(), dtype=complex)
        T_num = jones.jones_op_to_mueller_op(jones.op_fresnel_transmission(m, theta, n_i=n_i))
        self.assertTrue(np.allclose(T_sym.imag, 0, atol=1e-12))
        self.assertTrue(np.allclose(T_sym.real, T_num))

    def test_fresnel_reflection_accepts_symbolic_inputs(self):
        """Reflection operator should support symbolic refractive index and angle."""
        m = sympy.Symbol("m")
        theta = sympy.Symbol("theta")
        R = sym_mueller.op_fresnel_reflection(m, theta)
        self.assertEqual(R.shape, (4, 4))
        self.assertIn(m, R.free_symbols)
        self.assertIn(theta, R.free_symbols)


class TestSymMuellerConversions(unittest.TestCase):
    """Test symbolic Stokes/Mueller conversion helpers."""

    def test_stokes_to_jones_known_states(self):
        """Symbolic Stokes-to-Jones conversion should match canonical states."""
        H = sym_mueller.stokes_to_jones(sym_mueller.stokes_horizontal())
        self.assertEqual(sympy.simplify(H), sympy.Matrix([1, 0]))

        V = sym_mueller.stokes_to_jones(sym_mueller.stokes_vertical())
        self.assertEqual(sympy.simplify(V), sympy.Matrix([0, 1]))

        R = sym_mueller.stokes_to_jones(sym_mueller.stokes_right_circular())
        self.assertEqual(sympy.simplify(R), sympy.Matrix([sympy.sqrt(2) / 2, -sympy.I * sympy.sqrt(2) / 2]))

        U = sym_mueller.stokes_to_jones(sym_mueller.stokes_unpolarized())
        self.assertEqual(sympy.simplify(U), sympy.Matrix([0, 0]))

    def test_mueller_to_jones_known_matrices(self):
        """Symbolic Mueller-to-Jones conversion should match known matrices."""
        JI = sym_mueller.mueller_to_jones(sympy.eye(4))
        self.assertEqual(sympy.simplify(JI), sympy.eye(2))

        MM = sympy.diag(1, 1, -1, -1)
        JM = sym_mueller.mueller_to_jones(MM)
        self.assertEqual(sympy.simplify(JM[0, 0]), 1)
        self.assertEqual(sympy.simplify(JM[0, 1]), 0)
        self.assertEqual(sympy.simplify(JM[1, 0]), 0)
        self.assertEqual(sympy.simplify(JM[1, 1]), -1)


if __name__ == "__main__":
    unittest.main()
