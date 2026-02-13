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

    def test_draw_empty_sphere_returns_handles(self):
        """draw_empty_sphere should return figure, axis, and created artist handles."""
        fig, ax, artists = visualization.draw_empty_sphere()
        self.assertIs(fig, ax.figure)
        self.assertIn("surface", artists)
        self.assertIn("lines", artists)
        self.assertIn("texts", artists)
        self.assertGreater(len(artists["lines"]), 0)

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
        """Line width aliases should work and unsupported kwargs should fail via Matplotlib."""
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
        with self.assertRaises((TypeError, AttributeError)) as exc:
            visualization.draw_stokes_poincare(S, ax=ax, unknown_kw=1)
        self.assertIn("unknown_kw", str(exc.exception))

    def test_draw_jones_poincare_legacy_label_kwargs(self):
        """Legacy label text kwargs should be applied to text, not sent to plot()."""
        J = np.array([1.0, 1.0j]) / np.sqrt(2)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        _, _, artists = visualization.draw_jones_poincare(
            J, ax=ax, label="P", color="red", va="center", ha="right", fontsize=9
        )

        self.assertIsNotNone(artists["label"])
        self.assertEqual(artists["label"].get_va(), "center")
        self.assertEqual(artists["label"].get_ha(), "right")
        self.assertAlmostEqual(artists["label"].get_fontsize(), 9.0, places=12)

    def test_draw_stokes_poincare_text_kwargs_override_legacy_label_kwargs(self):
        """Explicit text_kwargs should override compatibility-mapped label kwargs."""
        S = np.array([1.0, 1.0, 0.0, 0.0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        _, _, artists = visualization.draw_stokes_poincare(S, ax=ax, label="P", va="bottom", text_kwargs={"va": "top"})

        self.assertEqual(artists["label"].get_va(), "top")

    def test_draw_empty_sphere_style_kwargs(self):
        """Sphere line styling should honor Matplotlib kwargs and reject unknown ones."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        visualization.draw_empty_sphere(ax=ax, linewidth=2.0)
        self.assertTrue(all(np.isclose(line.get_linewidth(), 2.0) for line in ax.lines))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        with self.assertRaises((TypeError, AttributeError)) as exc:
            visualization.draw_empty_sphere(ax=ax, unknown_kw=1)
        self.assertIn("unknown_kw", str(exc.exception))

    def test_draw_jones_ellipse_style_kwargs(self):
        """Ellipse drawing should pass Matplotlib kwargs through to line artists."""
        J = np.array([1.0, 1.0j]) / np.sqrt(2)
        _, _, artists = visualization.draw_jones_ellipse(J, simple=True, linewidth=2.5)
        self.assertTrue(all(np.isclose(line.get_linewidth(), 2.5) for line in artists["lines"]))

        with self.assertRaises((TypeError, AttributeError)) as exc:
            visualization.draw_jones_ellipse(J, simple=True, unknown_kw=1)
        self.assertIn("unknown_kw", str(exc.exception))

    def test_draw_stokes_ellipse_style_kwargs(self):
        """Stokes ellipse wrapper should forward kwargs to Jones ellipse drawing."""
        S = np.array([1.0, 1.0, 0.0, 0.0])
        _, _, artists = visualization.draw_stokes_ellipse(S, simple=True, linewidth=1.75)
        self.assertTrue(all(np.isclose(line.get_linewidth(), 1.75) for line in artists["lines"]))

    def test_draw_jones_field_style_kwargs(self):
        """Field plots should honor Matplotlib kwargs on generated line artists."""
        J = np.array([1.0, 1.0j]) / np.sqrt(2)
        _, axes, _ = visualization.draw_jones_field(J, linewidth=2.25)
        self.assertGreater(len(axes[0].lines), 0)
        self.assertAlmostEqual(axes[0].lines[0].get_linewidth(), 2.25, places=12)
        self.assertGreater(len(axes[1].lines), 0)
        self.assertAlmostEqual(axes[1].lines[0].get_linewidth(), 2.25, places=12)

        with self.assertRaises((TypeError, AttributeError)) as exc:
            visualization.draw_jones_field(J, unknown_kw=1)
        self.assertIn("unknown_kw", str(exc.exception))

    def test_draw_stokes_field_style_kwargs(self):
        """Stokes field wrapper should forward kwargs to Jones field drawing."""
        S = np.array([1.0, 1.0, 0.0, 0.0])
        _, axes, _ = visualization.draw_stokes_field(S, linewidth=1.6)
        self.assertGreater(len(axes[0].lines), 0)
        self.assertAlmostEqual(axes[0].lines[0].get_linewidth(), 1.6, places=12)

    def test_draw_jones_animated_style_kwargs(self):
        """Animated field plotting should honor kwargs on per-frame line artists."""
        J = np.array([1.0, 1.0j]) / np.sqrt(2)
        ani = visualization.draw_jones_animated(J, nframes=8, linewidth=1.4)
        ax_anim = ani._args[1]
        ani._func(0.0, *ani._args)
        self.assertGreater(len(ax_anim.lines), 0)
        self.assertAlmostEqual(ax_anim.lines[0].get_linewidth(), 1.4, places=12)
        ani._draw_was_started = True

    def test_draw_stokes_animated_style_kwargs(self):
        """Stokes animated wrapper should forward kwargs to Jones animated drawing."""
        S = np.array([1.0, 1.0, 0.0, 0.0])
        ani = visualization.draw_stokes_animated(S, nframes=8, linewidth=1.2)
        ax_anim = ani._args[1]
        ani._func(0.0, *ani._args)
        self.assertGreater(len(ax_anim.lines), 0)
        self.assertAlmostEqual(ax_anim.lines[0].get_linewidth(), 1.2, places=12)
        ani._draw_was_started = True

    def test_draw_stokes_poincare_returns_handles(self):
        """Point plotting should return figure, axis, and point/label artists."""
        S = np.array([1.0, 1.0, 0.0, 0.0])
        fig, ax, artists = visualization.draw_stokes_poincare(S, label="P")
        self.assertIs(fig, ax.figure)
        self.assertIn("point", artists)
        self.assertIn("label", artists)
        self.assertIsNotNone(artists["point"])
        self.assertIsNotNone(artists["label"])

    def test_join_stokes_poincare_returns_handles(self):
        """Arc plotting should return figure, axis, and line handle."""
        S1 = np.array([1.0, 1.0, 0.0, 0.0])
        S2 = np.array([1.0, 0.0, 1.0, 0.0])
        fig, ax, line = visualization.join_stokes_poincare(S1, S2)
        self.assertIs(fig, ax.figure)
        self.assertTrue(hasattr(line, "get_data_3d"))

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

    def test_draw_jones_field_returns_handles(self):
        """Field drawing should return figure, both axes, and grouped artists."""
        J = np.array([1.0, 1.0j]) / np.sqrt(2)
        fig, axes, artists = visualization.draw_jones_field(J, offset=0)
        self.assertEqual(len(axes), 2)
        self.assertIs(fig, axes[0].figure)
        self.assertIn("ax3d_lines", artists)
        self.assertIn("ax2d_lines", artists)
        self.assertGreater(len(artists["ax3d_lines"]), 0)
        self.assertGreater(len(artists["ax2d_lines"]), 0)

    def test_draw_jones_ellipse_returns_handles(self):
        """Ellipse drawing should return figure/axes and created artists in both modes."""
        J = np.array([1.0, 1.0j]) / np.sqrt(2)

        fig_simple, ax_simple, artists_simple = visualization.draw_jones_ellipse(J, simple=True)
        self.assertIs(fig_simple, ax_simple.figure)
        self.assertIn("lines", artists_simple)
        self.assertGreater(len(artists_simple["lines"]), 0)

        fig_panel, axes_panel, artists_panel = visualization.draw_jones_ellipse(J, simple=False)
        self.assertEqual(len(axes_panel), 2)
        self.assertIs(fig_panel, axes_panel[0].figure)
        self.assertIn("ax1_lines", artists_panel)
        self.assertIn("ax2_lines", artists_panel)

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
