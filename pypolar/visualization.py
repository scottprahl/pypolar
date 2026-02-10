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

Functions for drawing Poincaré representations::
   * draw_empty_sphere(ax=None, **kwargs)
   * draw_jones_poincare(J, ax=None, label=None, normalize="s0", text_kwargs=None, **kwargs)
   * draw_stokes_poincare(S, ax=None, label=None, normalize="s0", text_kwargs=None, **kwargs)
   * join_jones_poincare(J1, J2, ax=None, normalize="s0", **kwargs)
   * join_stokes_poincare(S1, S2, ax=None, normalize="s0", **kwargs)

Poincaré coordinates use reduced Stokes values (S1/S0, S2/S0, S3/S0),
so partially polarized states lie inside the unit sphere.

Jones-vector plots follow the package-wide sign convention set by
`pypolar.jones.use_alternate_convention(...)`.

Set `normalize="unit"` to project states onto the unit sphere using
`(S1,S2,S3) / sqrt(S1^2+S2^2+S3^2)`.

Example: Poincaré sphere plot of a Jones vector::

    J = pypolar.jones.field_linear(np.pi / 6)
    pypolar.visualization.draw_jones_poincare(J)

Example: Poincaré sphere plot of two Stokes vectors::

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

    ax.plot([0, last], [0, 0], [0, 0], "k", **kwargs)
    ax.plot([0, 0], [-the_max, the_max], [0, 0], "g", **kwargs)
    ax.plot([0, 0], [0, 0], [-the_max, the_max], "b", **kwargs)
    ax.text(0, 0, 1, "y", ha="center")
    ax.text(0, 1, 0, "x", va="center")
    ax.text(last * 1.05, 0, 0, "z", va="center")


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
    the_max = max(h_amp, v_amp) * 1.1

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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0, 1, "y", ha="center")
    ax.text(1, 0, "x", va="center")


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
    s = r"a=%.2f, b=%.2f, $\alpha$=%.2f°" % (a, b, np.degrees(alpha))
    ax.text(0, -1.15 * the_max, s, ha="center")

    # semi-minor diameter
    alpha += np.pi / 2
    dx = b * np.cos(alpha)
    dy = b * np.sin(alpha)
    ax.plot([0, dx], [0, dy], "g", **kwargs)
    ax.text(dx / 2, dy / 2, "  b", color="green")
    s = r"b / a=%.2f, " % (b / a)
    s += r"$\tan^{-1}(b / a)$=%.2f°" % np.degrees(pypolar.jones.ellipticity_angle(J))
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
    s = r"$E_{0x}$=%.2f, $E_{0y}$=%.2f, $\psi$=%.2f°" % (Ex0, Ey0, psi)
    ax.text(0, -1.15 * the_max, s, ha="center")
    s = r"$\phi_x$=%.2f°, " % np.degrees(phix)
    s += r"$\phi_y$=%.2f°, " % np.degrees(phiy)
    s += r"$\phi_y-\phi_x$=%.2f°" % np.degrees(phiy - phix)
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
    Draw 3D and 2D representations of the polarization field.

    Args:
        J:      Jones vector
        offset: starting point
        **kwargs: style arguments passed to line artists.

    Returns:
        tuple: `(fig, (ax3d, ax2d), artists)` where `artists` includes line and
        text handles for each axis.
    """
    JJ = _jones_for_visualization(J)

    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    ax1 = plt.subplot(gs[0], projection="3d")
    _draw_3D_field(JJ, ax1, offset, **kwargs)

    ax2 = plt.subplot(gs[1])
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
    Draw 3D and 2D field representations from a Stokes vector.

    Args:
        S:      Stokes vector
        offset: starting point
        **kwargs: style arguments passed to `draw_jones_field`.

    Returns:
        tuple: `(fig, (ax3d, ax2d), artists)` as returned by
        :func:`draw_jones_field`.
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
    """
    JJ = _jones_for_visualization(J)

    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0], projection="3d")
    ax2 = plt.subplot(gs[1])

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
    """
    J = pypolar.mueller.stokes_to_jones(S)
    return draw_jones_animated(J, **kwargs)


def draw_empty_sphere(ax=None, **kwargs):
    """
    Plot an empty Poincare sphere.

    Args:
        ax: pyplot axis
        **kwargs: style arguments passed to guide line artists.

    Returns:
        tuple: `(fig, ax, artists)` with surface, line, and text handles.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.view_init(elev=30, azim=45)

    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        try:
            ax.set_aspect("equal")
        except NotImplementedError:
            pass

    u = np.radians(np.linspace(0, 360, 90))
    v = np.radians(np.linspace(0, 180, 90))
    zz = np.zeros_like(u)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    surface = ax.plot_surface(x, y, z, alpha=0.1, color="blue")

    # draw circumferences
    lines = []
    circle_kwargs = dict(kwargs)
    if "lw" not in circle_kwargs and "linewidth" not in circle_kwargs:
        circle_kwargs["lw"] = 0.5
    lines.append(ax.plot(np.sin(u), np.cos(u), zz, "k", **circle_kwargs)[0])
    lines.append(ax.plot(np.sin(u), zz, np.cos(u), "k", **circle_kwargs)[0])
    lines.append(ax.plot(zz, np.sin(u), np.cos(u), "k", **circle_kwargs)[0])

    # draw x,y,z axes
    axis_kwargs = dict(kwargs)
    if "lw" not in axis_kwargs and "linewidth" not in axis_kwargs:
        axis_kwargs["lw"] = 1
    if "alpha" not in axis_kwargs:
        axis_kwargs["alpha"] = 0.5
    lines.append(ax.plot([-1, 1], [0, 0], [0, 0], "k--", **axis_kwargs)[0])
    lines.append(ax.plot([0, 0], [-1, 1], [0, 0], "k--", **axis_kwargs)[0])
    lines.append(ax.plot([0, 0], [0, 0], [-1, 1], "k--", **axis_kwargs)[0])

    # label directions
    texts = []
    texts.append(ax.text(1.15, 0, 0, "0°", fontsize=12, color="black", ha="center"))
    texts.append(ax.text(0, 1.25, 0, "45°", fontsize=12, color="black", ha="center"))
    texts.append(ax.text(0, 0, 1.15, "RCP", fontsize=12, color="black", ha="center"))
    texts.append(ax.text(0, 0, -1.15, "LCP", fontsize=12, color="black", ha="center"))
    texts.append(ax.text(-1.15, 0, 0, "90°", fontsize=12, color="black", ha="center"))

    # Stokes parameters
    ax.set_xlabel("S₁", fontsize=14, labelpad=-10)
    ax.set_ylabel("S₂", fontsize=14, labelpad=-10)
    ax.set_zlabel("S₃", fontsize=14, labelpad=-10)

    # Hide grid and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    artists = {"surface": surface, "lines": lines, "texts": texts}
    return fig, ax, artists


def great_circle_points(ax, ay, az, bx, by, bz):
    """
    Create a list of points along the great circle between a and b.

    The great circle is assumed to lie on the unit sphere with center at (0,0,0)

    The points a=(ax,ay,az) and b=(bx,by,bz) are the beginning and end of the arc.

    Algorithm is from https://www.physicsforums.com / threads / 571535
    """
    a = np.array([ax, ay, az], dtype=float)
    b = np.array([bx, by, bz], dtype=float)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if np.isclose(na, 0.0) or np.isclose(nb, 0.0):
        raise ValueError("Great-circle endpoints must be non-zero vectors.")

    a /= na
    b /= nb

    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    delta = np.arccos(dot)
    psi = np.linspace(0.0, delta)

    # Identical points: the arc degenerates to a single point.
    if np.isclose(delta, 0.0):
        p = np.repeat(a[np.newaxis, :], psi.size, axis=0)
        return p[:, 0], p[:, 1], p[:, 2]

    # Antipodal points: choose a deterministic orthogonal direction.
    if np.isclose(delta, np.pi):
        i = int(np.argmin(np.abs(a)))
        ref = np.eye(3)[i]
        u = np.cross(a, ref)
        u /= np.linalg.norm(u)
        p = np.cos(psi)[:, np.newaxis] * a + np.sin(psi)[:, np.newaxis] * u
        return p[:, 0], p[:, 1], p[:, 2]

    # Spherical linear interpolation (SLERP) on the unit sphere.
    sindelta = np.sin(delta)
    w1 = np.sin(delta - psi) / sindelta
    w2 = np.sin(psi) / sindelta
    p = w1[:, np.newaxis] * a + w2[:, np.newaxis] * b
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p[:, 0], p[:, 1], p[:, 2]


def spherical_angles(x, y, z):
    """Azimuth and elevation for a point on a sphere."""
    phi = np.arctan2(y, x)
    theta = np.arctan2(np.sqrt(x * x + y * y), z)
    return phi, theta


def _stokes_xyz_for_poincare(S, normalize="s0"):
    """Return Stokes coordinates for Poincaré plotting."""
    SS = np.asarray(S, dtype=float)
    if SS.shape != (4,):
        raise ValueError("Stokes vector must have shape (4,).")

    if normalize == "s0":
        s0 = SS[0]
        if np.isclose(s0, 0.0):
            raise ValueError("Stokes vector with S0=0 cannot be mapped onto the Poincare sphere.")
        return SS[1] / s0, SS[2] / s0, SS[3] / s0

    if normalize == "unit":
        sp = np.sqrt(SS[1] ** 2 + SS[2] ** 2 + SS[3] ** 2)
        if np.isclose(sp, 0.0):
            raise ValueError("Unpolarized Stokes vector cannot be projected onto the unit Poincare sphere.")
        return SS[1] / sp, SS[2] / sp, SS[3] / sp

    raise ValueError("normalize must be either 's0' or 'unit'.")


_LEGACY_POINCARE_TEXT_KWARGS = (
    "ha",
    "horizontalalignment",
    "va",
    "verticalalignment",
    "fontsize",
    "fontfamily",
    "fontname",
    "fontstyle",
    "fontvariant",
    "fontweight",
    "fontstretch",
    "fontproperties",
    "multialignment",
    "rotation_mode",
    "linespacing",
    "bbox",
)


def draw_stokes_poincare(S, ax=None, label=None, normalize="s0", text_kwargs=None, **kwargs):
    """
    Plot one Stokes state on or inside the Poincaré sphere.

    Coordinates are controlled by `normalize`:
    * `normalize="s0"` uses reduced Stokes values `(S1/S0, S2/S0, S3/S0)`.
    * `normalize="unit"` uses pure-state projection
      `(S1,S2,S3) / sqrt(S1^2+S2^2+S3^2)`.

    Any keyword arguments for point styling should use standard Matplotlib names
    (for example `linewidth`, `lw`, `color`, `linestyle`, `markersize`).
    Label styling should use `text_kwargs`; legacy text keys like `ha`, `va`,
    and `fontsize` in `**kwargs` are still accepted when `label` is provided.

    Args:
        S: Stokes vector with shape `(4,)`
        ax: optional matplotlib 3D axis
        label: optional text label
        normalize: either `"s0"` or `"unit"`
        text_kwargs: optional style args for label text
        **kwargs: style arguments passed directly to `matplotlib.axes.Axes.plot`
    Returns:
        tuple: `(fig, ax, artists)` with point and optional label handles.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        draw_empty_sphere(ax)
    else:
        fig = ax.figure

    x, y, z = _stokes_xyz_for_poincare(S, normalize=normalize)

    plot_kwargs = dict(kwargs)
    resolved_text_kwargs = text_kwargs
    if label is not None:
        if resolved_text_kwargs is None:
            resolved_text_kwargs = {}
        else:
            resolved_text_kwargs = dict(resolved_text_kwargs)

        # Backward compatibility: route legacy text kwargs to the label artist.
        for key in _LEGACY_POINCARE_TEXT_KWARGS:
            if key in plot_kwargs:
                if key not in resolved_text_kwargs:
                    resolved_text_kwargs[key] = plot_kwargs[key]
                plot_kwargs.pop(key)

        if "color" in plot_kwargs and "color" not in resolved_text_kwargs:
            resolved_text_kwargs["color"] = plot_kwargs["color"]

    point = ax.plot([x], [y], [z], "o", **plot_kwargs)[0]
    label_artist = None

    if label is not None:
        label_artist = ax.text(x, y, z, label, **resolved_text_kwargs)
    return fig, ax, {"point": point, "label": label_artist}


def draw_jones_poincare(J, ax=None, label=None, normalize="s0", text_kwargs=None, **kwargs):
    """
    Plot one Jones state on or inside the Poincaré sphere.

    Args:
        J: Jones vector with shape `(2,)`
        ax: optional matplotlib 3D axis
        label: optional text label
        normalize: either `"s0"` or `"unit"`
        text_kwargs: optional style args for label text
        **kwargs: style arguments passed to `draw_stokes_poincare`
    Returns:
        tuple: `(fig, ax, artists)` as returned by
        :func:`draw_stokes_poincare`.
    """
    JJ = _jones_for_visualization(J)
    S = pypolar.jones.jones_to_stokes(JJ)
    return draw_stokes_poincare(S, ax=ax, label=label, normalize=normalize, text_kwargs=text_kwargs, **kwargs)


def join_stokes_poincare(S1, S2, ax=None, normalize="s0", **kwargs):
    """
    Plot a connection between two Stokes vectors on or inside the Poincaré sphere.

    The direction follows a great-circle path for non-zero-radius endpoints and
    uses linear interpolation when an endpoint is at the origin.

    Args:
        S1: first Stokes vector with shape `(4,)`
        S2: second Stokes vector with shape `(4,)`
        ax: optional matplotlib 3D axis
        normalize: either `"s0"` or `"unit"`
        **kwargs: style arguments passed to `matplotlib.axes.Axes.plot`
    Returns:
        tuple: `(fig, ax, line)` where `line` is the connecting arc/segment.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        draw_empty_sphere(ax)
    else:
        fig = ax.figure

    p1 = np.array(_stokes_xyz_for_poincare(S1, normalize=normalize), dtype=float)
    p2 = np.array(_stokes_xyz_for_poincare(S2, normalize=normalize), dtype=float)
    r1 = np.linalg.norm(p1)
    r2 = np.linalg.norm(p2)

    # If either endpoint is at the origin, connect points with a straight segment.
    if np.isclose(r1, 0.0) or np.isclose(r2, 0.0):
        t = np.linspace(0.0, 1.0, 50)
        p = (1.0 - t)[:, np.newaxis] * p1 + t[:, np.newaxis] * p2
        line = ax.plot(p[:, 0], p[:, 1], p[:, 2], **kwargs)[0]
        return fig, ax, line

    u1 = p1 / r1
    u2 = p2 / r2
    ux, uy, uz = great_circle_points(u1[0], u1[1], u1[2], u2[0], u2[1], u2[2])
    u = np.column_stack((ux, uy, uz))

    # On the sphere, this is a great-circle arc; inside the sphere, scale radius between endpoints.
    radii = np.linspace(r1, r2, u.shape[0])
    p = u * radii[:, np.newaxis]
    line = ax.plot(p[:, 0], p[:, 1], p[:, 2], **kwargs)[0]
    return fig, ax, line


def join_jones_poincare(J1, J2, ax=None, normalize="s0", **kwargs):
    """
    Plot a connection between two Jones vectors on or inside the Poincaré sphere.

    Args:
        J1: first Jones vector with shape `(2,)`
        J2: second Jones vector with shape `(2,)`
        ax: optional matplotlib 3D axis
        normalize: either `"s0"` or `"unit"`
        **kwargs: style arguments passed to `join_stokes_poincare`
    Returns:
        tuple: `(fig, ax, line)` as returned by
        :func:`join_stokes_poincare`.
    """
    JJ1 = _jones_for_visualization(J1)
    JJ2 = _jones_for_visualization(J2)
    S1 = pypolar.jones.jones_to_stokes(JJ1)
    S2 = pypolar.jones.jones_to_stokes(JJ2)
    return join_stokes_poincare(S1, S2, ax=ax, normalize=normalize, **kwargs)
