"""Unit tests for symbolic Mueller/Stokes vector operations."""

import unittest

import sympy
from pypolar import sym_mueller


class TestSymMueller(unittest.TestCase):
    """Test symbolic Mueller helpers."""

    def test_dop_fully_polarized(self):
        """Degree of polarization should be one for fully polarized light."""
        S = sym_mueller.stokes_left_circular()
        dop = sym_mueller.degree_of_polarization(S)
        self.assertEqual(sympy.simplify(dop), 1)

    def test_dop_unpolarized(self):
        """Degree of polarization should be zero for unpolarized light."""
        S = sym_mueller.stokes_unpolarized()
        dop = sym_mueller.degree_of_polarization(S)
        self.assertEqual(sympy.simplify(dop), 0)

    def test_dop_partially_polarized(self):
        """Degree of polarization should match normalized polarized magnitude."""
        S = sympy.Matrix([1, sympy.Rational(1, 2), 0, 0])
        dop = sym_mueller.degree_of_polarization(S)
        self.assertEqual(sympy.simplify(dop), sympy.Rational(1, 2))


if __name__ == "__main__":
    unittest.main()
