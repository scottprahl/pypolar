"""Unit tests for visualization helpers and edge cases."""

import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
from pypolar import jones  # noqa: E402  pylint: disable=wrong-import-position
from pypolar import visualization  # noqa: E402  pylint: disable=wrong-import-position


class TestVisualization(unittest.TestCase):
    """Test visualization edge cases that should avoid NaNs."""

    def tearDown(self):
        """Close figures created during tests."""
        plt.close("all")

    def test_draw_stokes_poincare_unpolarized_maps_to_origin(self):
        """Unpolarized light should map to the sphere center."""
        S = np.array([1.0, 0.0, 0.0, 0.0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        visualization.draw_stokes_poincare(S, ax=ax)
        x, y, z = ax.lines[-1].get_data_3d()
        self.assertAlmostEqual(x[0], 0.0, places=12)
        self.assertAlmostEqual(y[0], 0.0, places=12)
        self.assertAlmostEqual(z[0], 0.0, places=12)

    def test_join_stokes_poincare_unpolarized_endpoint_is_finite(self):
        """Joining from/to the origin should produce a finite interior segment."""
        S1 = np.array([1.0, 0.0, 0.0, 0.0])
        S2 = np.array([1.0, 1.0, 0.0, 0.0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        visualization.join_stokes_poincare(S1, S2, ax=ax)
        x, y, z = ax.lines[-1].get_data_3d()
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.isfinite(y).all())
        self.assertTrue(np.isfinite(z).all())
        self.assertAlmostEqual(x[0], 0.0, places=12)
        self.assertAlmostEqual(y[0], 0.0, places=12)
        self.assertAlmostEqual(z[0], 0.0, places=12)
        self.assertAlmostEqual(x[-1], 1.0, places=12)
        self.assertAlmostEqual(y[-1], 0.0, places=12)
        self.assertAlmostEqual(z[-1], 0.0, places=12)

    def test_draw_stokes_poincare_valid_state_has_finite_point(self):
        """Polarized vectors should still plot finite coordinates."""
        S = np.array([1.0, 1.0, 0.0, 0.0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        visualization.draw_stokes_poincare(S, ax=ax)
        x, y, z = ax.lines[-1].get_data_3d()
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.isfinite(y).all())
        self.assertTrue(np.isfinite(z).all())

    def test_draw_stokes_poincare_partial_polarization_is_inside_sphere(self):
        """Partially polarized states should plot inside the unit sphere."""
        S = np.array([2.0, 1.0, 0.0, 0.0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        visualization.draw_stokes_poincare(S, ax=ax)
        x, y, z = ax.lines[-1].get_data_3d()
        self.assertAlmostEqual(x[0], 0.5, places=12)
        self.assertAlmostEqual(y[0], 0.0, places=12)
        self.assertAlmostEqual(z[0], 0.0, places=12)

    def test_great_circle_points_nearly_identical_is_finite(self):
        """Near-identical endpoints should not produce NaNs from arccos round-off."""
        x, y, z = visualization.great_circle_points(1.0, 0.0, 0.0, 1.0 + 1e-15, 0.0, 0.0)
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.isfinite(y).all())
        self.assertTrue(np.isfinite(z).all())
        self.assertAlmostEqual(x[0], 1.0, places=12)
        self.assertAlmostEqual(y[0], 0.0, places=12)
        self.assertAlmostEqual(z[0], 0.0, places=12)
        self.assertAlmostEqual(x[-1], 1.0, places=12)
        self.assertAlmostEqual(y[-1], 0.0, places=12)
        self.assertAlmostEqual(z[-1], 0.0, places=12)

    def test_great_circle_points_antipodal_stays_on_unit_sphere(self):
        """Antipodal endpoints should generate a finite deterministic half-circle."""
        x, y, z = visualization.great_circle_points(1.0, 0.0, 0.0, -1.0, 0.0, 0.0)
        r = np.sqrt(x * x + y * y + z * z)
        self.assertTrue(np.isfinite(r).all())
        self.assertTrue(np.allclose(r, 1.0, atol=1e-12))
        self.assertAlmostEqual(x[0], 1.0, places=12)
        self.assertAlmostEqual(y[0], 0.0, places=12)
        self.assertAlmostEqual(z[0], 0.0, places=12)
        self.assertAlmostEqual(x[-1], -1.0, places=12)
        self.assertAlmostEqual(y[-1], 0.0, places=12)
        self.assertAlmostEqual(z[-1], 0.0, places=12)

    def test_draw_jones_field_matches_between_sign_conventions(self):
        """Jones-field plotting should be consistent for equivalent physical states."""
        try:
            jones.use_alternate_convention(False)
            J_default = jones.field_right_circular()
            visualization.draw_jones_field(J_default, offset=0)
            ax_default = plt.gcf().axes[0]
            _, y_def, z_def = ax_default.lines[5].get_data_3d()

            jones.use_alternate_convention(True)
            J_alternate = jones.field_right_circular()
            visualization.draw_jones_field(J_alternate, offset=0)
            ax_alt = plt.gcf().axes[0]
            _, y_alt, z_alt = ax_alt.lines[5].get_data_3d()

            self.assertTrue(np.allclose(y_def, y_alt))
            self.assertTrue(np.allclose(z_def, z_alt))
        finally:
            jones.use_alternate_convention(False)

    def test_draw_jones_poincare_matches_between_sign_conventions(self):
        """Jones-Poincare plotting should be consistent for equivalent physical states."""
        try:
            jones.use_alternate_convention(False)
            J_default = jones.field_right_circular()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            visualization.draw_jones_poincare(J_default, ax=ax)
            x_def, y_def, z_def = ax.lines[-1].get_data_3d()

            jones.use_alternate_convention(True)
            J_alternate = jones.field_right_circular()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            visualization.draw_jones_poincare(J_alternate, ax=ax)
            x_alt, y_alt, z_alt = ax.lines[-1].get_data_3d()

            self.assertTrue(np.allclose(x_def, x_alt))
            self.assertTrue(np.allclose(y_def, y_alt))
            self.assertTrue(np.allclose(z_def, z_alt))
        finally:
            jones.use_alternate_convention(False)
