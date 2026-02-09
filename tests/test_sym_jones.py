"""Unit tests for symbolic Jones vector operations."""

import unittest

import sympy
from pypolar import sym_jones


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


if __name__ == "__main__":
    unittest.main()
