"""Unit tests for visualization helpers and edge cases."""

import importlib
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

    def test_draw_empty_sphere_uses_passed_axis(self):
        """Sphere guide lines should be drawn on the provided axes object."""
        fig = plt.figure()
        ax_target = fig.add_subplot(121, projection="3d")
        ax_other = fig.add_subplot(122, projection="3d")
        plt.sca(ax_other)

        visualization.draw_empty_sphere(ax_target)

        self.assertGreater(len(ax_target.lines), 0)
        self.assertEqual(len(ax_other.lines), 0)

    def test_draw_stokes_poincare_unpolarized_behavior(self):
        """Unpolarized light should map to the sphere center with finite coordinates."""
        S = np.array([1.0, 0.0, 0.0, 0.0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        visualization.draw_stokes_poincare(S, ax=ax)
        x, y, z = ax.lines[-1].get_data_3d()
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.isfinite(y).all())
        self.assertTrue(np.isfinite(z).all())
        self.assertAlmostEqual(x[0], 0.0, places=12)
        self.assertAlmostEqual(y[0], 0.0, places=12)
        self.assertAlmostEqual(z[0], 0.0, places=12)

    def test_join_stokes_poincare_unpolarized_behavior(self):
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

    def test_partial_polarization_radius(self):
        """Point radius should equal DOP when reduced Stokes coordinates are used."""
        S = np.array([2.0, 1.0, 1.0, 0.0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        visualization.draw_stokes_poincare(S, ax=ax)
        x, y, z = ax.lines[-1].get_data_3d()
        r = np.sqrt(x[0] * x[0] + y[0] * y[0] + z[0] * z[0])
        dop = np.sqrt(S[1] ** 2 + S[2] ** 2 + S[3] ** 2) / S[0]
        self.assertAlmostEqual(r, dop, places=12)

    def test_draw_stokes_poincare_normalize_unit_projects_to_sphere(self):
        """normalize='unit' should project partially polarized states onto the unit sphere."""
        S = np.array([2.0, 1.0, 1.0, 0.0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        visualization.draw_stokes_poincare(S, ax=ax, normalize="unit")
        x, y, z = ax.lines[-1].get_data_3d()
        r = np.sqrt(x[0] * x[0] + y[0] * y[0] + z[0] * z[0])
        self.assertAlmostEqual(r, 1.0, places=12)

    def test_draw_stokes_poincare_normalize_unit_unpolarized_raises(self):
        """normalize='unit' should reject unpolarized states (undefined direction)."""
        S = np.array([1.0, 0.0, 0.0, 0.0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        with self.assertRaises(ValueError):
            visualization.draw_stokes_poincare(S, ax=ax, normalize="unit")

    def test_join_stokes_poincare_normalize_unit_stays_on_sphere(self):
        """normalize='unit' join should trace a finite arc on the unit sphere."""
        S1 = np.array([2.0, 1.0, 0.0, 0.0])
        S2 = np.array([2.0, 0.0, 1.0, 0.0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        visualization.join_stokes_poincare(S1, S2, ax=ax, normalize="unit")
        x, y, z = ax.lines[-1].get_data_3d()
        r = np.sqrt(x * x + y * y + z * z)
        self.assertTrue(np.isfinite(r).all())
        self.assertTrue(np.allclose(r, 1.0, atol=1e-12))

    def test_draw_stokes_poincare_invalid_normalize_raises(self):
        """Unknown normalize modes should raise a clear ValueError."""
        S = np.array([1.0, 1.0, 0.0, 0.0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        with self.assertRaises(ValueError) as exc:
            visualization.draw_stokes_poincare(S, ax=ax, normalize="bad-mode")
        self.assertIn("normalize must be", str(exc.exception))

    def test_great_circle_points_clips_dot_product(self):
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

    def test_great_circle_points_antipodal_policy(self):
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

    def test_draw_stokes_poincare_style_kwargs(self):
        """Line width aliases should work and unsupported kwargs should fail clearly."""
        S = np.array([1.0, 1.0, 0.0, 0.0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        visualization.draw_stokes_poincare(S, ax=ax, linewidth=2.5)
        self.assertAlmostEqual(ax.lines[-1].get_linewidth(), 2.5, places=12)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        visualization.draw_stokes_poincare(S, ax=ax, lw=1.5)
        self.assertAlmostEqual(ax.lines[-1].get_linewidth(), 1.5, places=12)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        with self.assertRaises(TypeError) as exc:
            visualization.draw_stokes_poincare(S, ax=ax, unknown_kw=1)
        self.assertIn("Unsupported keyword", str(exc.exception))

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

    def test_sign_convention_consistency_between_field_and_animated(self):
        """Field and animated plots should agree for the same convention and phase offset."""
        try:
            jones.use_alternate_convention(True)
            J = jones.field_right_circular()
            offset = 0.0

            visualization.draw_jones_field(J, offset=offset)
            ax_field = plt.gcf().axes[0]
            _, y_field, z_field = ax_field.lines[5].get_data_3d()

            ani = visualization.draw_jones_animated(J, nframes=8)
            ax_anim = ani._args[1]
            ani._func(offset, *ani._args)
            _, y_anim, z_anim = ax_anim.lines[5].get_data_3d()
            ani._draw_was_started = True

            self.assertTrue(np.allclose(y_field, y_anim))
            self.assertTrue(np.allclose(z_field, z_anim))
        finally:
            jones.use_alternate_convention(False)

    def test_reload_does_not_mutate_animation_rcparam(self):
        """Importing visualization should not change global Matplotlib animation settings."""
        previous = plt.rcParams["animation.html"]
        try:
            plt.rcParams["animation.html"] = "none"
            importlib.reload(visualization)
            self.assertEqual(plt.rcParams["animation.html"], "none")
        finally:
            plt.rcParams["animation.html"] = previous
            importlib.reload(visualization)
