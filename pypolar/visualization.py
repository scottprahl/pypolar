"""
A set of basic routines for visualizing polarization.

Functions for drawing the polarization ellipse (sectional pattern)::

   * draw_jones_ellipse(J, simple=False)
   * draw_stokes_ellipse(S)

Functions for drawing 2D and 3D representations::

    * draw_jones_field(J, offset=0)
    * draw_stokes_field(S, offset=0)

Functions for drawing animated 2D and 3D representations::

   * draw_jones_animated(J, nframes=64)
   * draw_stokes_animated(S)

Functions for drawing Poincaré representations::
   * draw_empty_sphere(ax=None)
   * draw_jones_poincare(J, ax=None, label=None, normalize="s0", **kwargs)
   * draw_stokes_poincare(S, ax=None, label=None, normalize="s0", **kwargs)
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


def _draw_optical_axis_3d(J, ax, last=4 * np.pi):
    """
    Draw the optical axis in a 3D plot.

    Args:
        J:    Jones vector
        ax:   matplotlib axis to use
        last: length of optical axis
    """
    h_amp, v_amp = abs(J)
    the_max = max(h_amp, v_amp) * 1.1

    ax.plot([0, last], [0, 0], [0, 0], "k")
    ax.plot([0, 0], [-the_max, the_max], [0, 0], "g")
    ax.plot([0, 0], [0, 0], [-the_max, the_max], "b")
    ax.text(0, 0, 1, "y", ha="center")
    ax.text(0, 1, 0, "x", va="center")
    ax.text(last * 1.05, 0, 0, "z", va="center")


def _draw_h_field_3d(J, ax, offset, last=4 * np.pi):
    """
    Draw the horizontal electric field in a 3D plot.

    Args:
        J:      Jones vector
        ax:     matplotlib axis to use
        offset: starting point
        last:   length of optical axis
    """
    t = np.linspace(0, last, 100)
    x = t
    y = np.abs(J[0]) * np.cos(t + offset - np.angle(J[0]))
    z = 0
    ax.plot(x, y, z, ":g")


def _draw_v_field_3d(J, ax, offset, last=4 * np.pi):
    """
    Draw the vertical electric field in a 3D plot.

    Args:
        J:      Jones vector
        ax:     matplotlib axis to use
        offset: starting point
        last:   length of optical axis
    """
    t = np.linspace(0, last, 100)
    x = t
    y = 0 * t
    z = np.abs(J[1]) * np.cos(t + offset - np.angle(J[1]))
    ax.plot(x, y, z, ":b")


def _draw_total_field_3d(J, ax, offset, last=4 * np.pi):
    """
    Draw the total electric field in a 3D plot.

    Args:
        J:      Jones vector
        ax:     matplotlib axis to use
        offset: starting point
        last:   length of optical axis
    """
    t = np.linspace(0, last, 100)
    x = t
    y = np.abs(J[0]) * np.cos(t + offset - np.angle(J[0]))
    z = np.abs(J[1]) * np.cos(t + offset - np.angle(J[1]))
    ax.plot(x, y, z, "r")


def _draw_projected_vector_3d(J, ax, offset):
    """
    Draw the projection vector of the polarization field in 3D.

    Args:
        J:      Jones vector
        ax:     matplotlib axis to use
        offset: starting point
    """
    y = np.abs(J[0]) * np.cos(offset - np.angle(J[0]))
    z = np.abs(J[1]) * np.cos(offset - np.angle(J[1]))

    x1, y1, z1 = 0, y, 0
    x2, y2, z2 = 0, y, z
    ax.plot([x1, x2], [y1, y2], [z1, z2], "g--")

    x1, y1, z1 = 0, 0, z
    ax.plot([x1, x2], [y1, y2], [z1, z2], "b--")

    x1, y1, z1 = 0, 0, 0
    ax.plot([x1, x2], [y1, y2], [z1, z2], "r")
    ax.scatter([0], [y], [z], marker="o", color="red")


def _draw_3D_field(J, ax, offset):
    """
    Draw a representation of the polarization fields in 3D.

    Args:
        J:      Jones vector
        ax:     matplotlib axis to use
        offset: starting point
    """
    _draw_optical_axis_3d(J, ax)
    _draw_h_field_3d(J, ax, offset)
    _draw_v_field_3d(J, ax, offset)
    _draw_total_field_3d(J, ax, offset)
    _draw_projected_vector_3d(J, ax, offset)

    ax.grid(False)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def _draw_2D_field(J, ax, offset):
    """
    Draw a simple 2D representation of the projected field.

    Also called a sectional pattern.

    Args:
        J:      Jones vector
        ax:     matplotlib axis to use
        offset: starting point
    """
    h_amp, v_amp = np.abs(J)
    h_phi, v_phi = np.angle(J)
    the_max = max(h_amp, v_amp) * 1.1

    ax.plot([-the_max, the_max], [0, 0], "g")
    ax.plot([0, 0], [-the_max, the_max], "b")

    t = np.linspace(0, 2 * np.pi, 100)
    x = h_amp * np.cos(t + offset - h_phi)
    y = v_amp * np.cos(t + offset - v_phi)
    ax.plot(x, y, "k")

    x = h_amp * np.cos(offset - h_phi)
    y = v_amp * np.cos(offset - v_phi)
    ax.plot(x, y, "ro")
    ax.plot([x, x], [0, y], "g--")
    ax.plot([0, x], [y, y], "b--")
    ax.plot([0, x], [0, y], "r")

    ax.set_xlim(-the_max, the_max)
    ax.set_ylim(-the_max, the_max)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0, 1, "y", ha="center")
    ax.text(1, 0, "x", va="center")


def _animation_update(offset, J, ax1, ax2):
    """
    Draw the next animation frame.

    Args:
        offset: starting phase for drawings
        J:      Jones vector
        ax1:    matplotlib axis for 3D plot
        ax2:    matplotlib axis for 2D plot
    """
    ax1.clear()
    ax2.clear()
    _draw_3D_field(J, ax1, offset)
    _draw_2D_field(J, ax2, offset)
    return ax1, ax2


def draw_ellipse_axes(J, ax):
    """
    Draw the sectional pattern with ellipse labels.

    Args:
        J:  Jones vector
        ax: plot axis
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
    ax.plot(xx, yy, "b")

    # semi-major diameter
    dx = a * np.cos(alpha)
    dy = a * np.sin(alpha)
    ax.plot([0, dx], [0, dy], "r")
    ax.text(dx / 2, dy / 2, "  a", color="red")
    ax.text(dx / 5, dy / 10, r"$\alpha$", va="center", ha="center")
    s = r"a=%.2f, b=%.2f, $\alpha$=%.2f°" % (a, b, np.degrees(alpha))
    ax.text(0, -1.15 * the_max, s, ha="center")

    # semi-minor diameter
    alpha += np.pi / 2
    dx = b * np.cos(alpha)
    dy = b * np.sin(alpha)
    ax.plot([0, dx], [0, dy], "g")
    ax.text(dx / 2, dy / 2, "  b", color="green")
    s = r"b / a=%.2f, " % (b / a)
    s += r"$\tan^{-1}(b / a)$=%.2f°" % np.degrees(pypolar.jones.ellipticity_angle(J))
    ax.text(0, -1.30 * the_max, s, ha="center")

    # draw x and y axes
    ax.plot([0, 0], [-the_max, the_max], "k")
    ax.plot([-the_max, the_max], [0, 0], "k")
    ax.set_xlim(-the_max, the_max)
    ax.set_ylim(-the_max, the_max)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_ellipse_Ex_Ey(J, ax):
    """
    Draw the sectional pattern with field labels.

    Args:
        J:  Jones vector
        ax: plot axis
    """
    Ex0, Ey0 = np.abs(J)
    phix, phiy = np.angle(J)

    t = np.linspace(0, 2 * np.pi, 100)
    xx = Ex0 * np.cos(t + phix)
    yy = Ey0 * np.cos(t + phiy)

    the_max = max(Ex0, Ey0) * 1.2
    ax.set_aspect("equal")
    ax.plot(xx, yy, "b")
    ax.plot([-Ex0, -Ex0, Ex0, Ex0, -Ex0], [-Ey0, Ey0, Ey0, -Ey0, -Ey0], ":g")
    ax.plot([-Ex0, Ex0], [-Ey0, Ey0], ":r")
    ax.plot([0, 0], [-the_max, the_max], "k")
    ax.plot([-the_max, the_max], [0, 0], "k")
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


def draw_jones_ellipse(J, simple=False):
    """
    Draw a 2D sectional pattern for a Jones vector.

    Args:
        J:      Jones vector
        simple: if True then just draw a simple ellipse plot
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
        ax.set_xlim(-the_max, the_max)
        ax.set_ylim(-the_max, the_max)
        ax.set_aspect("equal")
        ax.axhline(0, color="black")
        ax.axvline(0, color="black")
        ax.plot(xx, yy, "b")
        ax.plot([-Ex0, Ex0], [-Ey0, Ey0], ":r")
        ax.axis("off")
        ax.text(0, Ey0 / 5, r" $\psi$", va="bottom", ha="left")
        return

    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax1 = plt.subplot(gs[0])
    draw_ellipse_axes(JJ, ax1)
    ax2 = plt.subplot(gs[1])
    draw_ellipse_Ex_Ey(JJ, ax2)


def draw_stokes_ellipse(S):
    """
    Draw polarization ellipse panels from a Stokes vector.

    Args:
        S:      Stokes vector
    """
    J = pypolar.mueller.stokes_to_jones(S)
    draw_jones_ellipse(J)


def draw_jones_field(J, offset=0):
    """
    Draw 3D and 2D representations of the polarization field.

    Args:
        J:      Jones vector
        offset: starting point
    """
    JJ = _jones_for_visualization(J)

    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    ax1 = plt.subplot(gs[0], projection="3d")
    _draw_3D_field(JJ, ax1, offset)

    ax2 = plt.subplot(gs[1])
    _draw_2D_field(JJ, ax2, offset)


def draw_stokes_field(S, offset=0):
    """
    Draw 3D and 2D field representations from a Stokes vector.

    Args:
        S:      Stokes vector
        offset: starting point
    """
    J = pypolar.mueller.stokes_to_jones(S)
    draw_jones_field(J, offset)


def draw_jones_animated(J, nframes=64):
    """
    Animate 3D and 2D representations of the polarization field.

    Args:
        J:      Jones vector
        nframes: number of frames to create
    """
    JJ = _jones_for_visualization(J)

    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0], projection="3d")
    ax2 = plt.subplot(gs[1])

    ani = animation.FuncAnimation(
        fig, _animation_update, frames=np.linspace(0, -2 * np.pi, nframes), fargs=(JJ, ax1, ax2)
    )
    plt.close()
    return ani


def draw_stokes_animated(S):
    """
    Draw animated 3D and 2D field representations from a Stokes vector.

    Args:
        S:      Stokes vector
    """
    J = pypolar.mueller.stokes_to_jones(S)
    ani = draw_jones_animated(J)
    return ani


def draw_empty_sphere(ax=None):
    """
    Plot an empty Poincare sphere.

    Args:
        ax: pyplot axis
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

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

    ax.plot_surface(x, y, z, alpha=0.1, color="blue")

    # draw circumferences
    ax.plot(np.sin(u), np.cos(u), zz, "k", lw=0.5)
    ax.plot(np.sin(u), zz, np.cos(u), "k", lw=0.5)
    ax.plot(zz, np.sin(u), np.cos(u), "k", lw=0.5)

    # draw x,y,z axes
    ax.plot([-1, 1], [0, 0], [0, 0], "k--", lw=1, alpha=0.5)
    ax.plot([0, 0], [-1, 1], [0, 0], "k--", lw=1, alpha=0.5)
    ax.plot([0, 0], [0, 0], [-1, 1], "k--", lw=1, alpha=0.5)

    # label directions
    ax.text(1.15, 0, 0, "0°", fontsize=12, color="black", ha="center")
    ax.text(0, 1.25, 0, "45°", fontsize=12, color="black", ha="center")
    ax.text(0, 0, 1.15, "RCP", fontsize=12, color="black", ha="center")
    ax.text(0, 0, -1.15, "LCP", fontsize=12, color="black", ha="center")
    ax.text(-1.15, 0, 0, "90°", fontsize=12, color="black", ha="center")

    # Stokes parameters
    ax.set_xlabel("S₁", fontsize=14, labelpad=-10)
    ax.set_ylabel("S₂", fontsize=14, labelpad=-10)
    ax.set_zlabel("S₃", fontsize=14, labelpad=-10)

    # Hide grid and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


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


def draw_stokes_poincare(S, ax=None, label=None, normalize="s0", **kwargs):
    """
    Plot one Stokes state on or inside the Poincaré sphere.

    Coordinates are controlled by `normalize`:
    * `normalize="s0"` uses reduced Stokes values `(S1/S0, S2/S0, S3/S0)`.
    * `normalize="unit"` uses pure-state projection
      `(S1,S2,S3) / sqrt(S1^2+S2^2+S3^2)`.

    Any keyword arguments for point styling should use standard Matplotlib names
    (for example `linewidth`, `lw`, `color`, `linestyle`, `markersize`).

    Args:
        S: Stokes vector with shape `(4,)`
        ax: optional matplotlib 3D axis
        label: optional text label
        normalize: either `"s0"` or `"unit"`
        **kwargs: style arguments for the plotted point and optional label text
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        draw_empty_sphere(ax)

    x, y, z = _stokes_xyz_for_poincare(S, normalize=normalize)

    if "lineweight" in kwargs:
        raise TypeError("`lineweight` is not supported; use `linewidth` or `lw`.")

    plot_keys = ["linewidth", "lw", "color", "linestyle", "ls", "markersize", "ms", "marker"]
    text_keys = ["fontsize", "ha", "color", "va"]
    allowed_keys = set(plot_keys + text_keys)
    unknown = sorted(k for k in kwargs if k not in allowed_keys)
    if unknown:
        raise TypeError("Unsupported keyword(s) for draw_stokes_poincare: %s" % ", ".join(unknown))

    plot_args = {}
    plot_args.update((k, kwargs[k]) for k in plot_keys if k in kwargs)
    ax.plot([x], [y], [z], "o", **plot_args)

    if label is not None:
        text_args = dict((k, kwargs[k]) for k in text_keys if k in kwargs)
        ax.text(x, y, z, label, **text_args)


def draw_jones_poincare(J, ax=None, label=None, normalize="s0", **kwargs):
    """
    Plot one Jones state on or inside the Poincaré sphere.

    Args:
        J: Jones vector with shape `(2,)`
        ax: optional matplotlib 3D axis
        label: optional text label
        normalize: either `"s0"` or `"unit"`
        **kwargs: style arguments passed to `draw_stokes_poincare`
    """
    JJ = _jones_for_visualization(J)
    S = pypolar.jones.jones_to_stokes(JJ)
    draw_stokes_poincare(S, ax=ax, label=label, normalize=normalize, **kwargs)


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
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        draw_empty_sphere(ax)

    p1 = np.array(_stokes_xyz_for_poincare(S1, normalize=normalize), dtype=float)
    p2 = np.array(_stokes_xyz_for_poincare(S2, normalize=normalize), dtype=float)
    r1 = np.linalg.norm(p1)
    r2 = np.linalg.norm(p2)

    # If either endpoint is at the origin, connect points with a straight segment.
    if np.isclose(r1, 0.0) or np.isclose(r2, 0.0):
        t = np.linspace(0.0, 1.0, 50)
        p = (1.0 - t)[:, np.newaxis] * p1 + t[:, np.newaxis] * p2
        ax.plot(p[:, 0], p[:, 1], p[:, 2], **kwargs)
        return

    u1 = p1 / r1
    u2 = p2 / r2
    ux, uy, uz = great_circle_points(u1[0], u1[1], u1[2], u2[0], u2[1], u2[2])
    u = np.column_stack((ux, uy, uz))

    # On the sphere, this is a great-circle arc; inside the sphere, scale radius between endpoints.
    radii = np.linspace(r1, r2, u.shape[0])
    p = u * radii[:, np.newaxis]
    ax.plot(p[:, 0], p[:, 1], p[:, 2], **kwargs)


def join_jones_poincare(J1, J2, ax=None, normalize="s0", **kwargs):
    """
    Plot a connection between two Jones vectors on or inside the Poincaré sphere.

    Args:
        J1: first Jones vector with shape `(2,)`
        J2: second Jones vector with shape `(2,)`
        ax: optional matplotlib 3D axis
        normalize: either `"s0"` or `"unit"`
        **kwargs: style arguments passed to `join_stokes_poincare`
    """
    JJ1 = _jones_for_visualization(J1)
    JJ2 = _jones_for_visualization(J2)
    S1 = pypolar.jones.jones_to_stokes(JJ1)
    S2 = pypolar.jones.jones_to_stokes(JJ2)
    join_stokes_poincare(S1, S2, ax=ax, normalize=normalize, **kwargs)
