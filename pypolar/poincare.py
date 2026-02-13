"""
Poincare-sphere visualization routines.

Functions for drawing Poincare representations::
   * draw_empty_sphere(ax=None, **kwargs)
   * draw_jones_poincare(J, ax=None, label=None, normalize="s0", text_kwargs=None, **kwargs)
   * draw_stokes_poincare(S, ax=None, label=None, normalize="s0", text_kwargs=None, **kwargs)
   * join_jones_poincare(J1, J2, ax=None, normalize="s0", **kwargs)
   * join_stokes_poincare(S1, S2, ax=None, normalize="s0", **kwargs)

Poincare coordinates use reduced Stokes values (S1/S0, S2/S0, S3/S0),
so partially polarized states lie inside the unit sphere.

Set `normalize="unit"` to project states onto the unit sphere using
`(S1,S2,S3) / sqrt(S1^2+S2^2+S3^2)`.
"""

import numpy as np
import matplotlib.pyplot as plt

import pypolar.jones

__all__ = (
    "draw_empty_sphere",
    "draw_jones_poincare",
    "draw_stokes_poincare",
    "join_jones_poincare",
    "join_stokes_poincare",
)


def _jones_for_visualization(J):
    """Return a Jones vector in the plotting convention used by this package."""
    if pypolar.jones.alternate_sign_convention:
        return np.conjugate(J)
    return J


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

    # Draw circumferences.
    lines = []
    circle_kwargs = dict(kwargs)
    if "lw" not in circle_kwargs and "linewidth" not in circle_kwargs:
        circle_kwargs["lw"] = 0.5
    lines.append(ax.plot(np.sin(u), np.cos(u), zz, "k", **circle_kwargs)[0])
    lines.append(ax.plot(np.sin(u), zz, np.cos(u), "k", **circle_kwargs)[0])
    lines.append(ax.plot(zz, np.sin(u), np.cos(u), "k", **circle_kwargs)[0])

    # Draw x,y,z axes.
    axis_kwargs = dict(kwargs)
    if "lw" not in axis_kwargs and "linewidth" not in axis_kwargs:
        axis_kwargs["lw"] = 1
    if "alpha" not in axis_kwargs:
        axis_kwargs["alpha"] = 0.5
    lines.append(ax.plot([-1, 1], [0, 0], [0, 0], "k--", **axis_kwargs)[0])
    lines.append(ax.plot([0, 0], [-1, 1], [0, 0], "k--", **axis_kwargs)[0])
    lines.append(ax.plot([0, 0], [0, 0], [-1, 1], "k--", **axis_kwargs)[0])

    # Label directions.
    texts = []
    texts.append(ax.text(1.15, 0, 0, "0°", fontsize=12, color="black", ha="center"))
    texts.append(ax.text(0, 1.25, 0, "45°", fontsize=12, color="black", ha="center"))
    texts.append(ax.text(0, 0, 1.15, "RCP", fontsize=12, color="black", ha="center"))
    texts.append(ax.text(0, 0, -1.15, "LCP", fontsize=12, color="black", ha="center"))
    texts.append(ax.text(-1.15, 0, 0, "90°", fontsize=12, color="black", ha="center"))

    # Stokes parameters.
    ax.set_xlabel("S₁", fontsize=14, labelpad=-10)
    ax.set_ylabel("S₂", fontsize=14, labelpad=-10)
    ax.set_zlabel("S₃", fontsize=14, labelpad=-10)

    # Hide grid and ticks.
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
    """Return Stokes coordinates for Poincare plotting."""
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
    Plot one Stokes state on or inside the Poincare sphere.

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
    Plot one Jones state on or inside the Poincare sphere.

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
    Plot a connection between two Stokes vectors on or inside the Poincare sphere.

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
    Plot a connection between two Jones vectors on or inside the Poincare sphere.

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
