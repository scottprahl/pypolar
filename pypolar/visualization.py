"""
A set of basic routines for visualizing polarization.

Functions for drawing the polarization ellipse (sectional pattern)::

   * draw_jones_ellipse(J, simple=False, **kwargs)
   * draw_stokes_ellipse(S, **kwargs)

Functions for drawing 2D and 3D representations::

    * draw_jones_field(J, offset=0, **kwargs)
    * draw_stokes_field(S, offset=0, **kwargs)

Functions for drawing animated 2D and 3D representations::

   * draw_jones_animated(J, nframes=64, **kwargs)
   * draw_stokes_animated(S, **kwargs)

Functions for drawing PoincarÃ© representations::
   * draw_empty_sphere(ax=None, **kwargs)
   * draw_jones_poincare(J, ax=None, label=None, normalize="s0", text_kwargs=None, **kwargs)
   * draw_stokes_poincare(S, ax=None, label=None, normalize="s0", text_kwargs=None, **kwargs)
   * join_jones_poincare(J1, J2, ax=None, normalize="s0", **kwargs)
   * join_stokes_poincare(S1, S2, ax=None, normalize="s0", **kwargs)

PoincarÃ© coordinates use reduced Stokes values (S1/S0, S2/S0, S3/S0),
so partially polarized states lie inside the unit sphere.

Jones-vector plots follow the package-wide sign convention set by
`pypolar.jones.use_alternate_convention(...)`.

Set `normalize="unit"` to project states onto the unit sphere using
`(S1,S2,S3) / sqrt(S1^2+S2^2+S3^2)`.

Example: PoincarÃ© sphere plot of a Jones vector::

    J = pypolar.jones.field_linear(np.pi / 6)
    pypolar.visualization.draw_jones_poincare(J)

Example: PoincarÃ© sphere plot of two Stokes vectors::

    S1 = pypolar.mueller.stokes_left_circular()
    S2 = pypolar.mueller.stokes_linear(np.radians(15))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    pypolar.visualization.draw_empty_sphere(ax)
    pypolar.visualization.draw_stokes_poincare(S1, ax, label='  S1')
    pypolar.visualization.draw_stokes_poincare(S2, ax, label='  S2')
    pypolar.visualization.join_stokes_poincare(S1, S2, ax, lw=2, ls=':', color='orange')
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation

import pypolar.fresnel
import pypolar.mueller
import pypolar.jones
import pypolar.poincare as _poincare

draw_empty_sphere = _poincare.draw_empty_sphere
draw_jones_poincare = _poincare.draw_jones_poincare
draw_stokes_poincare = _poincare.draw_stokes_poincare
join_jones_poincare = _poincare.join_jones_poincare
join_stokes_poincare = _poincare.join_stokes_poincare
great_circle_points = _poincare.great_circle_points
spherical_angles = _poincare.spherical_angles

__all__ = (
    "draw_jones_field",
    "draw_jones_animated",
    "draw_jones_ellipse",
    "draw_stokes_ellipse",
    "draw_stokes_field",
    "draw_stokes_animated",
    "draw_empty_sphere",
    "draw_jones_poincare",
    "draw_stokes_poincare",
    "join_jones_poincare",
    "join_stokes_poincare",
)


def _jones_for_visualization(J):
    """Return a Jones vector in the plotting convention used by this module."""
    if pypolar.jones.alternate_sign_convention:
        return np.conjugate(J)
    return J


def _draw_optical_axis_3d(J, ax, last=4 * np.pi, **kwargs):
    """
    Draw the optical axis in a 3D plot.

    Args:
        J:    Jones vector
        ax:   matplotlib axis to use
        last: length of optical axis
        **kwargs: style arguments passed to line artists.
    """
    h_amp, v_amp = abs(J)
    the_max = max(h_amp, v_amp) * 1.1

    ax.plot([0, last * 1.15], [0, 0], [0, 0], "k", **kwargs)
    ax.plot([0, 0], [-the_max, the_max], [0, 0], "g", **kwargs)
    ax.plot([0, 0], [0, 0], [-the_max, the_max], "b", **kwargs)
    ax.text(0, 0, the_max*1.1, "y", ha="center", va="bottom")
    ax.text(0, the_max*1.1, 0, "x", va="center")
    ax.text(last * 1.2, 0, 0, "z", va="center")


def _draw_h_field_3d(J, ax, offset, last=4 * np.pi, **kwargs):
    """
    Draw the horizontal electric field in a 3D plot.

    Args:
        J:      Jones vector
        ax:     matplotlib axis to use
        offset: starting point
        last:   length of optical axis
        **kwargs: style arguments passed to line artists.
    """
    t = np.linspace(0, last, 100)
    x = t
    y = np.abs(J[0]) * np.cos(t + offset - np.angle(J[0]))
    z = 0
    ax.plot(x, y, z, ":g", **kwargs)


def _draw_v_field_3d(J, ax, offset, last=4 * np.pi, **kwargs):
    """
    Draw the vertical electric field in a 3D plot.

    Args:
        J:      Jones vector
        ax:     matplotlib axis to use
        offset: starting point
        last:   length of optical axis
        **kwargs: style arguments passed to line artists.
    """
    t = np.linspace(0, last, 100)
    x = t
    y = 0 * t
    z = np.abs(J[1]) * np.cos(t + offset - np.angle(J[1]))
    ax.plot(x, y, z, ":b", **kwargs)


def _draw_total_field_3d(J, ax, offset, last=4 * np.pi, **kwargs):
    """
    Draw the total electric field in a 3D plot.

    Args:
        J:      Jones vector
        ax:     matplotlib axis to use
        offset: starting point
        last:   length of optical axis
        **kwargs: style arguments passed to line artists.
    """
    t = np.linspace(0, last, 100)
    x = t
    y = np.abs(J[0]) * np.cos(t + offset - np.angle(J[0]))
    z = np.abs(J[1]) * np.cos(t + offset - np.angle(J[1]))
    ax.plot(x, y, z, "r", **kwargs)


def _draw_projected_vector_3d(J, ax, offset, **kwargs):
    """
    Draw the projection vector of the polarization field in 3D.

    Args:
        J:      Jones vector
        ax:     matplotlib axis to use
        offset: starting point
        **kwargs: style arguments passed to line artists.
    """
    y = np.abs(J[0]) * np.cos(offset - np.angle(J[0]))
    z = np.abs(J[1]) * np.cos(offset - np.angle(J[1]))

    x1, y1, z1 = 0, y, 0
    x2, y2, z2 = 0, y, z
    ax.plot([x1, x2], [y1, y2], [z1, z2], "g--", **kwargs)

    x1, y1, z1 = 0, 0, z
    ax.plot([x1, x2], [y1, y2], [z1, z2], "b--", **kwargs)

    x1, y1, z1 = 0, 0, 0
    ax.plot([x1, x2], [y1, y2], [z1, z2], "r", **kwargs)
    ax.plot([0], [y], [z], "ro", **kwargs)


def _draw_3D_field(J, ax, offset, **kwargs):
    """
    Draw a representation of the polarization fields in 3D.

    Args:
        J:      Jones vector
        ax:     matplotlib axis to use
        offset: starting point
        **kwargs: style arguments passed to line artists.
    """
    _draw_optical_axis_3d(J, ax, **kwargs)
    _draw_h_field_3d(J, ax, offset, **kwargs)
    _draw_v_field_3d(J, ax, offset, **kwargs)
    _draw_total_field_3d(J, ax, offset, **kwargs)
    _draw_projected_vector_3d(J, ax, offset, **kwargs)

    ax.grid(False)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect(None, zoom=1.3)


def _draw_2D_field(J, ax, offset, **kwargs):
    """
    Draw a simple 2D representation of the projected field.

    Also called a sectional pattern.

    Args:
        J:      Jones vector
        ax:     matplotlib axis to use
        offset: starting point
        **kwargs: style arguments passed to line artists.
    """
    h_amp, v_amp = np.abs(J)
    h_phi, v_phi = np.angle(J)
    the_max = max(h_amp, v_amp) * 1.2

    ax.plot([-the_max, the_max], [0, 0], "g", **kwargs)
    ax.plot([0, 0], [-the_max, the_max], "b", **kwargs)

    t = np.linspace(0, 2 * np.pi, 100)
    x = h_amp * np.cos(t + offset - h_phi)
    y = v_amp * np.cos(t + offset - v_phi)
    ax.plot(x, y, "k", **kwargs)

    x = h_amp * np.cos(offset - h_phi)
    y = v_amp * np.cos(offset - v_phi)
    ax.plot(x, y, "ro", **kwargs)
    ax.plot([x, x], [0, y], "g--", **kwargs)
    ax.plot([0, x], [y, y], "b--", **kwargs)
    ax.plot([0, x], [0, y], "r", **kwargs)

    ax.set_xlim(-the_max, the_max)
    ax.set_ylim(-the_max, the_max)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")
    ax.text(0, the_max, "y", ha="center", va="bottom")
    ax.text(the_max, 0, "x", va="center")


def _animation_update(offset, J, ax1, ax2, plot_kwargs):
    """
    Draw the next animation frame.

    Args:
        offset: starting phase for drawings
        J:      Jones vector
        ax1:    matplotlib axis for 3D plot
        ax2:    matplotlib axis for 2D plot
        plot_kwargs: style arguments passed to line artists.
    """
    ax1.clear()
    ax2.clear()
    _draw_3D_field(J, ax1, offset, **plot_kwargs)
    _draw_2D_field(J, ax2, offset, **plot_kwargs)
    return ax1, ax2


def draw_ellipse_axes(J, ax, **kwargs):
    """
    Draw the sectional pattern with ellipse labels.

    Args:
        J:  Jones vector
        ax: plot axis
        **kwargs: style arguments passed to line artists.
    """
    Ex0, Ey0 = np.abs(J)
    phix, phiy = np.angle(J)

    alpha = pypolar.jones.ellipse_azimuth(J)
    a, b = pypolar.jones.ellipse_axes(J)

    t = np.linspace(0, 2 * np.pi, 100)
    xx = Ex0 * np.cos(t + phix)
    yy = Ey0 * np.cos(t + phiy)

    the_max = max(Ex0, Ey0) * 1.2

    ax.set_aspect("equal")
    ax.plot(xx, yy, "b", **kwargs)

    # semi-major diameter
    dx = a * np.cos(alpha)
    dy = a * np.sin(alpha)
    ax.plot([0, dx], [0, dy], "r", **kwargs)
    ax.text(dx / 2, dy / 2, "  a", color="red")
    ax.text(dx / 5, dy / 10, r"$\alpha$", va="center", ha="center")
    s = r"a=%.2f, b=%.2f, $\alpha$=%.2fÂ°" % (a, b, np.degrees(alpha))
    ax.text(0, -1.15 * the_max, s, ha="center")

    # semi-minor diameter
    alpha += np.pi / 2
    dx = b * np.cos(alpha)
    dy = b * np.sin(alpha)
    ax.plot([0, dx], [0, dy], "g", **kwargs)
    ax.text(dx / 2, dy / 2, "  b", color="green")
    s = r"b / a=%.2f, " % (b / a)
    s += r"$\tan^{-1}(b / a)$=%.2fÂ°" % np.degrees(pypolar.jones.ellipticity_angle(J))
    ax.text(0, -1.30 * the_max, s, ha="center")

    # draw x and y axes
    ax.plot([0, 0], [-the_max, the_max], "k", **kwargs)
    ax.plot([-the_max, the_max], [0, 0], "k", **kwargs)
    ax.set_xlim(-the_max, the_max)
    ax.set_ylim(-the_max, the_max)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_ellipse_Ex_Ey(J, ax, **kwargs):
    """
    Draw the sectional pattern with field labels.

    Args:
        J:  Jones vector
        ax: plot axis
        **kwargs: style arguments passed to line artists.
    """
    Ex0, Ey0 = np.abs(J)
    phix, phiy = np.angle(J)

    t = np.linspace(0, 2 * np.pi, 100)
    xx = Ex0 * np.cos(t + phix)
    yy = Ey0 * np.cos(t + phiy)

    the_max = max(Ex0, Ey0) * 1.2
    ax.set_aspect("equal")
    ax.plot(xx, yy, "b", **kwargs)
    ax.plot([-Ex0, -Ex0, Ex0, Ex0, -Ex0], [-Ey0, Ey0, Ey0, -Ey0, -Ey0], ":g", **kwargs)
    ax.plot([-Ex0, Ex0], [-Ey0, Ey0], ":r", **kwargs)
    ax.plot([0, 0], [-the_max, the_max], "k", **kwargs)
    ax.plot([-the_max, the_max], [0, 0], "k", **kwargs)
    ax.text(Ex0, 0, r" $E_{x0}$", va="bottom", ha="left")
    ax.text(-Ex0, 0, r"$-E_{x0} $", va="bottom", ha="right")
    ax.text(0, Ey0, r"$E_{y0}$", va="bottom", ha="left")
    ax.text(0, -Ey0, r"$-E_{y0}$", va="top", ha="left")
    ax.text(0, Ey0 / 5, r" $\psi$", va="bottom", ha="left")
    ax.set_xlim(-the_max, the_max)
    ax.set_ylim(-the_max, the_max)
    ax.set_xticks([])
    ax.set_yticks([])
    psi = np.degrees(np.arctan2(Ex0, Ey0))
    s = r"$E_{0x}$=%.2f, $E_{0y}$=%.2f, $\psi$=%.2fÂ°" % (Ex0, Ey0, psi)
    ax.text(0, -1.15 * the_max, s, ha="center")
    s = r"$\phi_x$=%.2fÂ°, " % np.degrees(phix)
    s += r"$\phi_y$=%.2fÂ°, " % np.degrees(phiy)
    s += r"$\phi_y-\phi_x$=%.2fÂ°" % np.degrees(phiy - phix)
    ax.text(0, -1.30 * the_max, s, ha="center")


def draw_jones_ellipse(J, simple=False, **kwargs):
    """
    Draw a 2D sectional pattern for a Jones vector.

    Args:
        J:      Jones vector
        simple: if True then just draw a simple ellipse plot
        **kwargs: style arguments passed to line artists.

    Returns:
        tuple: `(fig, ax_or_axes, artists)` where `ax_or_axes` is one axis for
        `simple=True` and `(ax1, ax2)` for `simple=False`.
    """
    JJ = _jones_for_visualization(J)

    if simple:
        Ex0, Ey0 = np.abs(JJ)
        phix, phiy = np.angle(JJ)
        the_max = max(Ex0, Ey0) * 1.2
        t = np.linspace(0, 2 * np.pi, 100)
        xx = Ex0 * np.cos(t + phix)
        yy = Ey0 * np.cos(t + phiy)
        ax = plt.gca()
        fig = ax.figure
        n_lines = len(ax.lines)
        n_texts = len(ax.texts)
        ax.set_xlim(-the_max, the_max)
        ax.set_ylim(-the_max, the_max)
        ax.set_aspect("equal")
        axis_kwargs = dict(kwargs)
        if "color" not in axis_kwargs and "c" not in axis_kwargs:
            axis_kwargs["color"] = "black"
        ax.axhline(0, **axis_kwargs)
        ax.axvline(0, **axis_kwargs)
        ax.plot(xx, yy, "b", **kwargs)
        ax.plot([-Ex0, Ex0], [-Ey0, Ey0], ":r", **kwargs)
        ax.axis("off")
        ax.text(0, Ey0 / 5, r" $\psi$", va="bottom", ha="left")
        artists = {"lines": list(ax.lines[n_lines:]), "texts": list(ax.texts[n_texts:])}
        return fig, ax, artists

    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax1 = plt.subplot(gs[0])
    draw_ellipse_axes(JJ, ax1, **kwargs)
    ax2 = plt.subplot(gs[1])
    draw_ellipse_Ex_Ey(JJ, ax2, **kwargs)
    artists = {
        "ax1_lines": list(ax1.lines),
        "ax1_texts": list(ax1.texts),
        "ax2_lines": list(ax2.lines),
        "ax2_texts": list(ax2.texts),
    }
    return fig, (ax1, ax2), artists


def draw_stokes_ellipse(S, **kwargs):
    """
    Draw polarization ellipse panels from a Stokes vector.

    Args:
        S:      Stokes vector
        **kwargs: style arguments passed to `draw_jones_ellipse`.

    Returns:
        tuple: `(fig, ax_or_axes, artists)` as returned by
        :func:`draw_jones_ellipse`.
    """
    J = pypolar.mueller.stokes_to_jones(S)
    return draw_jones_ellipse(J, **kwargs)


def draw_jones_field(J, offset=0, **kwargs):
    """
    Draw 3D and 2D representations of a Jones vector.

    Args:
        J:      Jones vector
        offset: starting point
        **kwargs: style arguments passed to line artists.

    Returns:
        tuple: `(fig, (ax3d, ax2d), artists)` where `artists` includes line and
        text handles for each axis.

    Example:
        Plot a static field representation::

            import matplotlib.pyplot as plt
            import pypolar.jones as jones
            import pypolar.visualization as vis

            J = jones.field_left_circular()
            vis.draw_jones_field(J)
            plt.show()
    """
    JJ = _jones_for_visualization(J)

    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 3])
    ax1 = fig.add_subplot(gs[0], projection="3d")
    ax2 = fig.add_subplot(gs[1])

    _draw_3D_field(JJ, ax1, offset, **kwargs)

    _draw_2D_field(JJ, ax2, offset, **kwargs)
    artists = {
        "ax3d_lines": list(ax1.lines),
        "ax3d_collections": list(ax1.collections),
        "ax3d_texts": list(ax1.texts),
        "ax2d_lines": list(ax2.lines),
        "ax2d_texts": list(ax2.texts),
    }
    return fig, (ax1, ax2), artists


def draw_stokes_field(S, offset=0, **kwargs):
    """
    Draw 3D and 2D field representations of a Stokes vector.

    Args:
        S:      Stokes vector
        offset: starting point
        **kwargs: style arguments passed to `draw_jones_field`.

    Returns:
        tuple: `(fig, (ax3d, ax2d), artists)` as returned by
        :func:`draw_jones_field`.

    Example:
        Plot a static field representation from a Stokes vector::

            import matplotlib.pyplot as plt
            import pypolar.mueller as mueller
            import pypolar.visualization as vis

            S = mueller.stokes_linear(0)
            vis.draw_stokes_field(S)
            plt.show()

        For animated output in notebooks, use :func:`draw_stokes_animated`.
    """
    J = pypolar.mueller.stokes_to_jones(S)
    return draw_jones_field(J, offset, **kwargs)


def draw_jones_animated(J, nframes=64, **kwargs):
    """
    Animate 3D and 2D representations of the polarization field.

    Args:
        J:      Jones vector
        nframes: number of frames to create
        **kwargs: style arguments passed to line artists in each frame.

    Returns:
        matplotlib.animation.FuncAnimation: animation handle. The associated
        figure and axes are available via `ani._fig` and `ani._args[1:]`.

    Example:
        Display an animation in Jupyter notebooks::

            import matplotlib.pyplot as plt
            import pypolar.jones as jones
            import pypolar.visualization as vis

            plt.rcParams["animation.html"] = "jshtml"
            J = jones.field_linear(0)
            ani = vis.draw_jones_animated(J, nframes=32)
            ani

        In scripts, save the animation to a file::

            ani = vis.draw_jones_animated(J, nframes=32)
            ani.save("jones_field.mp4")
    """
    JJ = _jones_for_visualization(J)

    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 3])
    ax1 = fig.add_subplot(gs[0], projection="3d")
    ax2 = fig.add_subplot(gs[1])

    ani = animation.FuncAnimation(
        fig, _animation_update, frames=np.linspace(0, -2 * np.pi, nframes), fargs=(JJ, ax1, ax2, kwargs)
    )
    ani.axes = (ax1, ax2)
    plt.close()
    return ani


def draw_stokes_animated(S, **kwargs):
    """
    Draw animated 3D and 2D field representations from a Stokes vector.

    Args:
        S:      Stokes vector
        **kwargs: style arguments passed to `draw_jones_animated`.

    Returns:
        matplotlib.animation.FuncAnimation: animation handle as returned by
        :func:`draw_jones_animated`.

    Example:
        Display an animation in Jupyter notebooks::

            import matplotlib.pyplot as plt
            import pypolar.mueller as mueller
            import pypolar.visualization as vis

            plt.rcParams["animation.html"] = "jshtml"
            S = mueller.stokes_right_circular()
            ani = vis.draw_stokes_animated(S, nframes=32)
            ani
    """
    J = pypolar.mueller.stokes_to_jones(S)
    return draw_jones_animated(J, **kwargs)
