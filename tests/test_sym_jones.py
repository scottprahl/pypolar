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


if __name__ == "__main__":
    unittest.main()
