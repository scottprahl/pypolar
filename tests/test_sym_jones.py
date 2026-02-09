"""Unit tests for symbolic Jones vector operations."""

import unittest

import sympy
from pypolar import sym_jones
from pypolar import fresnel
from pypolar import sym_fresnel


class TestSymJones(unittest.TestCase):
    """Test symbolic Jones helpers."""

    def test_ellipse_orientation_linear(self):
        """Test that orientation for linear polarization is computed as a scalar expression."""
        theta = sympy.pi / 6
        J = sym_jones.field_linear(theta)
        psi = sym_jones.ellipse_orientation(J)
        self.assertEqual(sympy.simplify(psi - theta), 0)

    def test_ellipse_axes_linear(self):
        """Test that ellipse axes are returned without unpack errors."""
        theta = sympy.pi / 6
        J = sym_jones.field_linear(theta)
        a, b = sym_jones.ellipse_axes(J)
        self.assertEqual(sympy.simplify(a), 1)
        self.assertEqual(sympy.simplify(b), 0)

    def test_fresnel_transmission_operator_matches_symbolic_field_amplitudes(self):
        """Symbolic Jones Fresnel transmission should return Fresnel field amplitudes."""
        m = sympy.Rational(3, 2)
        theta = sympy.pi / 4
        J = sym_jones.op_fresnel_transmission(m, theta)
        self.assertEqual(sympy.simplify(J[0, 0] - sym_fresnel.t_par_amplitude(m, theta)), 0)
        self.assertEqual(sympy.simplify(J[1, 1] - sym_fresnel.t_per_amplitude(m, theta)), 0)

    def test_fresnel_transmission_operator_matches_numeric_field_amplitudes_under_tir(self):
        """Symbolic Jones Fresnel transmission should preserve complex TIR amplitudes."""
        m = sympy.Rational(4, 5)
        theta = 70 * sympy.pi / 180
        J = sym_jones.op_fresnel_transmission(m, theta)
        tpar = fresnel.t_par_amplitude(float(m), float(sympy.N(theta)))
        tper = fresnel.t_per_amplitude(float(m), float(sympy.N(theta)))
        self.assertAlmostEqual(complex(sympy.N(J[0, 0])), tpar, places=12)
        self.assertAlmostEqual(complex(sympy.N(J[1, 1])), tper, places=12)
        self.assertGreater(abs(complex(sympy.N(J[0, 0]))), 0)
        self.assertGreater(abs(complex(sympy.N(J[1, 1]))), 0)

    def test_fresnel_operators_with_nonunity_incident_index(self):
        """Symbolic Jones Fresnel operators should support n_i != 1."""
        m = sympy.Integer(1)
        n_i = sympy.Rational(3, 2)
        theta = sympy.pi / 6

        R = sym_jones.op_fresnel_reflection(m, theta, n_i=n_i)
        T = sym_jones.op_fresnel_transmission(m, theta, n_i=n_i)

        theta_num = float(sympy.N(theta))
        m_num = float(sympy.N(m))
        n_i_num = float(sympy.N(n_i))
        self.assertAlmostEqual(
            complex(sympy.N(R[0, 0])), fresnel.r_par_amplitude(m_num, theta_num, n_i=n_i_num), places=12
        )
        self.assertAlmostEqual(
            complex(sympy.N(R[1, 1])), fresnel.r_per_amplitude(m_num, theta_num, n_i=n_i_num), places=12
        )
        self.assertAlmostEqual(
            complex(sympy.N(T[0, 0])), fresnel.t_par_amplitude(m_num, theta_num, n_i=n_i_num), places=12
        )
        self.assertAlmostEqual(
            complex(sympy.N(T[1, 1])), fresnel.t_per_amplitude(m_num, theta_num, n_i=n_i_num), places=12
        )

    def test_field_constructors_and_phase(self):
        """Symbolic field constructors and phase helper should match conventions."""
        self.assertEqual(sym_jones.field_horizontal(), sym_jones.field_linear(0))
        self.assertEqual(sym_jones.field_vertical(), sym_jones.field_linear(sympy.pi / 2))

        phase_r = sym_jones.phase(sym_jones.field_right_circular())
        phase_l = sym_jones.phase(sym_jones.field_left_circular())
        self.assertEqual(sympy.simplify(phase_r + sympy.pi / 2), 0)
        self.assertEqual(sympy.simplify(phase_l - sympy.pi / 2), 0)

    def test_symbolic_operator_basics(self):
        """Symbolic Jones operators should satisfy basic projection and inversion identities."""
        H = sym_jones.field_horizontal()
        V = sym_jones.field_vertical()
        P0 = sym_jones.op_linear_polarizer(0)
        self.assertEqual(P0 * H, H)
        self.assertEqual(P0 * V, sympy.Matrix([0, 0]))

        theta = sympy.Symbol("theta", real=True)
        R = sym_jones.op_rotation(theta)
        self.assertEqual(sympy.simplify(R * sym_jones.op_rotation(-theta)), sympy.eye(2))

        self.assertEqual(sym_jones.op_mirror(), sympy.Matrix([[1, 0], [0, -1]]))

        self.assertEqual(
            sym_jones.op_attenuator(sympy.Rational(1, 4)),
            sympy.Matrix([[sympy.Rational(1, 2), 0], [0, sympy.Rational(1, 2)]]),
        )

        theta2 = sympy.Symbol("theta2", real=True)
        self.assertEqual(
            sympy.simplify(sym_jones.op_quarter_wave_plate(theta2) - sym_jones.op_retarder(theta2, sympy.pi / 2)),
            sympy.zeros(2),
        )
        self.assertEqual(
            sympy.simplify(sym_jones.op_half_wave_plate(theta2) - sym_jones.op_retarder(theta2, sympy.pi)),
            sympy.zeros(2),
        )

        nd = sympy.Integer(2)
        self.assertEqual(
            sym_jones.op_neutral_density_filter(nd),
            sympy.Matrix([[sympy.Rational(1, 200), 0], [0, sympy.Rational(1, 200)]]),
        )

    def test_intensity_and_elliptical_passthrough(self):
        """Intensity and elliptical passthrough should return expected symbolic results."""
        J = sym_jones.field_right_circular()
        intensity = sym_jones.intensity(J)
        self.assertEqual(sympy.simplify(intensity[0] - 1), 0)

        A = 1 + sympy.I
        B = 2 - sympy.I
        J2 = sym_jones.field_elliptical(A, B)
        self.assertEqual(J2, sympy.Matrix([A, B]))
        self.assertEqual(sympy.simplify(sym_jones.ellipse_ellipticity(sym_jones.field_linear(sympy.pi / 6))), 0)

    def test_symbolic_fresnel_reflection_matches_symbolic_amplitudes(self):
        """Symbolic reflection operator should map directly to symbolic Fresnel amplitudes."""
        m = sympy.Rational(3, 2)
        n_i = sympy.Rational(6, 5)
        theta = sympy.pi / 7
        m_rel = m / n_i
        R = sym_jones.op_fresnel_reflection(m, theta, n_i=n_i)
        self.assertEqual(sympy.simplify(R[0, 0] - sym_fresnel.r_par_amplitude(m_rel, theta)), 0)
        self.assertEqual(sympy.simplify(R[1, 1] - sym_fresnel.r_per_amplitude(m_rel, theta)), 0)

    def test_symbolic_jones_to_stokes_known_states(self):
        """Jones-to-Stokes conversion should match canonical symbolic circular states."""
        self.assertEqual(
            sympy.simplify(sym_jones.jones_to_stokes(sym_jones.field_right_circular())),
            sympy.Matrix([1, 0, 0, 1]),
        )
        self.assertEqual(
            sympy.simplify(sym_jones.jones_to_stokes(sym_jones.field_left_circular())),
            sympy.Matrix([1, 0, 0, -1]),
        )

    def test_symbolic_jones_op_to_mueller_op_known_matrices(self):
        """Jones-operator conversion should reproduce known Mueller operators."""
        self.assertEqual(sympy.simplify(sym_jones.jones_op_to_mueller_op(sympy.eye(2))), sympy.eye(4))
        mirror_j = sympy.Matrix([[1, 0], [0, -1]])
        self.assertEqual(
            sympy.simplify(sym_jones.jones_op_to_mueller_op(mirror_j)),
            sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
        )


if __name__ == "__main__":
    unittest.main()
