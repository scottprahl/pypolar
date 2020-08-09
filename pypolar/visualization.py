# pylint: disable=invalid-name
# pylint: disable=bare-except
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=unused-import
# pylint: disable=too-many-arguments

"""
A set of basic routines for visualizing polarization.

Functions for drawing the polarization ellipse (sectional pattern)::

   draw_jones_ellipse(J)
   draw_stokes_ellipse(S)

Functions for drawing 2D and 3D representations::

    draw_jones_field(J)
    draw_stokes_field(S)

Functions for drawing an animated 2D and 3D representations::

   draw_jones_animated(J)
   draw_stokes_animated(S)

Functions for drawing a Poincaré representation::
   draw_empty_sphere()
   draw_jones_poincare(J)
   draw_stokes_poincare(S)
   join_jones_poincare(J)
   join_stokes_poincare(S)

Example: Poincaré sphere plot of a Jones vector::

    J = pypolar.jones.field_linear(np.pi/6)
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
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import pypolar.fresnel
import pypolar.mueller
import pypolar.jones

plt.rcParams["animation.html"] = "jshtml"

__all__ = ('draw_jones_field',
           'draw_jones_animated',
           'draw_jones_ellipse',
           'draw_stokes_ellipse',
           'draw_stokes_field',
           'draw_stokes_animated',
           'draw_empty_sphere',
           'draw_jones_poincare',
           'draw_stokes_poincare',
           'join_jones_poincare',
           'join_stokes_poincare'
           )

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

    ax.plot([0, last], [0, 0], [0, 0], 'k')
    ax.plot([0, 0], [-the_max, the_max], [0, 0], 'g')
    ax.plot([0, 0], [0, 0], [-the_max, the_max], 'b')
    ax.text(0, 0, 1, "y", ha="center")
    ax.text(0, 1, 0, "x", va="center")
    ax.text(last*1.05, 0, 0, "z", va="center")


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
    ax.plot(x, y, z, ':g')


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
    ax.plot(x, y, z, ':b')


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
    ax.plot(x, y, z, 'r')


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
    ax.plot([x1, x2], [y1, y2], [z1, z2], 'g--')

    x1, y1, z1 = 0, 0, z
    ax.plot([x1, x2], [y1, y2], [z1, z2], 'b--')

    x1, y1, z1 = 0, 0, 0
    ax.plot([x1, x2], [y1, y2], [z1, z2], 'r')
    ax.scatter([0], [y], [z], marker='o', color='red')


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
    ax.axis('off')
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

    ax.plot([-the_max, the_max], [0, 0], 'g')
    ax.plot([0, 0], [-the_max, the_max], 'b')

    t = np.linspace(0, 2 * np.pi, 100)
    x = h_amp * np.cos(t + offset - h_phi)
    y = v_amp * np.cos(t + offset - v_phi)
    ax.plot(x, y, 'k')

    x = h_amp * np.cos(offset - h_phi)
    y = v_amp * np.cos(offset - v_phi)
    ax.plot(x, y, 'ro')
    ax.plot([x, x], [0, y], 'g--')
    ax.plot([0, x], [y, y], 'b--')
    ax.plot([0, x], [0, y], 'r')

    ax.set_xlim(-the_max, the_max)
    ax.set_ylim(-the_max, the_max)
    ax.set_aspect('equal')
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

    ax.set_aspect('equal')
    ax.plot(xx, yy, 'b')

    # semi-major diameter
    dx = a * np.cos(alpha)
    dy = a * np.sin(alpha)
    ax.plot([0, dx], [0, dy], 'r')
    ax.text(dx/2, dy/2, '  a', color='red')
    ax.text(dx/5, dy/10, r'$\alpha$', va='center', ha='center')
    s = r'a=%.2f, b=%.2f, $\alpha$=%.2f°' % (a, b, np.degrees(alpha))
    ax.text(0, -1.15*the_max, s, ha='center')

    # semi-minor diameter
    alpha += np.pi/2
    dx = b * np.cos(alpha)
    dy = b * np.sin(alpha)
    ax.plot([0, dx], [0, dy], 'g')
    ax.text(dx/2, dy/2, '  b', color='green')
    s = r'b/a=%.2f, ' % (b/a)
    s += r'$\tan^{-1}(b/a)$=%.2f°' % np.degrees(pypolar.jones.ellipticity_angle(J))
    ax.text(0, -1.30*the_max, s, ha='center')

    # draw x and y axes
    ax.plot([0, 0], [-the_max, the_max], 'k')
    ax.plot([-the_max, the_max], [0, 0], 'k')
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
    ax.set_aspect('equal')
    ax.plot(xx, yy, 'b')
    ax.plot([-Ex0, -Ex0, Ex0, Ex0, -Ex0], [-Ey0, Ey0, Ey0, -Ey0, -Ey0], ':g')
    ax.plot([-Ex0, Ex0], [-Ey0, Ey0], ':r')
    ax.plot([0, 0], [-the_max, the_max], 'k')
    ax.plot([-the_max, the_max], [0, 0], 'k')
    ax.text(Ex0, 0, r' $E_{x0}$', va='bottom', ha='left')
    ax.text(-Ex0, 0, r'$-E_{x0} $', va='bottom', ha='right')
    ax.text(0, Ey0, r'$E_{y0}$', va='bottom', ha='left')
    ax.text(0, -Ey0, r'$-E_{y0}$', va='top', ha='left')
    ax.text(0, Ey0/5, r' $\psi$', va='bottom', ha='left')
    ax.set_xlim(-the_max, the_max)
    ax.set_ylim(-the_max, the_max)
    ax.set_xticks([])
    ax.set_yticks([])
    psi = np.degrees(np.arctan2(Ex0, Ey0))
    s = r'$E_{0x}$=%.2f, $E_{0y}$=%.2f, $\psi$=%.2f°' % (Ex0, Ey0, psi)
    ax.text(0, -1.15*the_max, s, ha='center')
    s = r'$\phi_x$=%.2f°, ' % np.degrees(phix)
    s += r'$\phi_y$=%.2f°, ' % np.degrees(phiy)
    s += r'$\phi_y-\phi_x$=%.2f°' % np.degrees(phiy-phix)
    ax.text(0, -1.30*the_max, s, ha='center')


def draw_jones_ellipse(J, simple=False):
    """
    Draw a 2D sectional pattern for a Jones vector.

    Args:
        J:      Jones vector
        simple: if True then just draw a simple ellipse plot
    """
    JJ = J
    if pypolar.jones.alternate_sign_convention:
        JJ = np.conjugate(J)

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
        ax.set_aspect('equal')
        ax.axhline(0, color='black')
        ax.axvline(0, color='black')
        ax.plot(xx, yy, 'b')
        ax.plot([-Ex0, Ex0], [-Ey0, Ey0], ':r')
        ax.axis('off')
        ax.text(0, Ey0/5, r' $\psi$', va='bottom', ha='left')
        return

    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax1 = plt.subplot(gs[0])
    draw_ellipse_axes(JJ, ax1)
    ax2 = plt.subplot(gs[1])
    draw_ellipse_Ex_Ey(JJ, ax2)


def draw_stokes_ellipse(S):
    """
    Draw a 2D and 3D representation of the polarization.

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
    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    ax1 = plt.subplot(gs[0], projection='3d')
    _draw_3D_field(J, ax1, offset)

    ax2 = plt.subplot(gs[1])
    _draw_2D_field(J, ax2, offset)


def draw_stokes_field(S, offset=0):
    """
    Draw a 2D and 3D representation of the polarization.

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
    """
    JJ = J
    if pypolar.jones.alternate_sign_convention:
        JJ = np.conjugate(J)

    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0], projection='3d')
    ax2 = plt.subplot(gs[1])

    ani = animation.FuncAnimation(fig, _animation_update,
                                  frames=np.linspace(0, -2 * np.pi, nframes),
                                  fargs=(JJ, ax1, ax2))
    plt.close()
    return ani


def draw_stokes_animated(S):
    """
    Draw animated 2D and 3D representations of the polarization.

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
        ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=30, azim=45)

    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            pass

    u = np.radians(np.linspace(0, 360, 90))
    v = np.radians(np.linspace(0, 180, 90))
    zz = np.zeros_like(u)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, alpha=0.1, color='blue')

    # draw circumferences
    plt.plot(np.sin(u), np.cos(u), 0, 'k', lw=0.5)
    plt.plot(np.sin(u), zz, np.cos(u), 'k', lw=0.5)
    plt.plot(zz, np.sin(u), np.cos(u), 'k', lw=0.5)

    # draw x,y,z axes
    plt.plot([-1, 1], [0, 0], [0, 0], 'k--', lw=1, alpha=0.5)
    plt.plot([0, 0], [-1, 1], [0, 0], 'k--', lw=1, alpha=0.5)
    plt.plot([0, 0], [0, 0], [-1, 1], 'k--', lw=1, alpha=0.5)

    # label directions
    ax.text(1.15, 0, 0, '0°', fontsize=12, color='black', ha='center')
    ax.text(0, 1.25, 0, '45°', fontsize=12, color='black', ha='center')
    ax.text(0, 0, 1.15, 'RCP', fontsize=12, color='black', ha='center')
    ax.text(0, 0, -1.15, 'LCP', fontsize=12, color='black', ha='center')
    ax.text(-1.15, 0, 0, '90°', fontsize=12, color='black', ha='center')

    # Stokes parameters
    ax.set_xlabel('S₁', fontsize=14, labelpad=-10)
    ax.set_ylabel('S₂', fontsize=14, labelpad=-10)
    ax.set_zlabel('S₃', fontsize=14, labelpad=-10)

    # Hide grid and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

def great_circle_points(ax, ay, az, bx, by, bz):
    """
    Create a list of points along the great circle between a and b.

    The great circle is assumed to lie on the unit sphere with center at (0,0,0)

    The points a=(ax,ay,az) and b=(bx,by,bz) are the beginning and end of the arc.

    Algorithm is from https://www.physicsforums.com/threads/571535
    """
    delta = np.arccos(ax*bx + ay*by + az*bz)
    psi = np.linspace(0, delta)
    sinpsi = np.sin(psi)
    cospsi = np.cos(psi)
    sindelta = np.sin(delta)

    # handle case when delta=0° or 180°
    if sindelta == 0:
        sindelta = 1e-5
    elif abs(sindelta) < 1e-5:
        sindelta = 1e-5 * np.sign(sindelta)

    x = cospsi * ax + sinpsi * ((az**2 + ay**2)*bx - (az*bz+ay*by)*ax)/sindelta
    y = cospsi * ay + sinpsi * ((az**2 + ax**2)*by - (az*bz+ax*bx)*ay)/sindelta
    z = cospsi * az + sinpsi * ((ay**2 + ax**2)*bz - (ay*by+ax*bx)*az)/sindelta
    return x, y, z

def spherical_angles(x, y, z):
    """Azimuth and elevation for a point on a sphere."""
    phi = np.arctan2(y, x)
    theta = np.arctan2(np.sqrt(x*x+y*y), z)
    return phi, theta

def draw_stokes_poincare(S, ax=None, label=None, **kwargs):
    """Plot single point on Poincaré sphere."""
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        draw_empty_sphere(ax)

    SS = np.sqrt(S[1]**2+S[2]**2+S[3]**2)
    x = S[1]/SS
    y = S[2]/SS
    z = S[3]/SS

    plot_keys = ['lineweight', 'color', 'linestyle', 'markersize']
    plot_args = dict((k, kwargs[k]) for k in plot_keys if k in kwargs)
    ax.plot([x], [y], [z], 'o', **plot_args)

    if not label is None:
        text_keys = ['fontsize', 'ha', 'color', 'va']
        text_args = dict((k, kwargs[k]) for k in text_keys if k in kwargs)
        ax.text(x, y, z, label, **text_args)

def draw_jones_poincare(J, ax=None, label=None, **kwargs):
    """Plot single point on Poincaré sphere."""
    S = pypolar.jones.jones_to_stokes(J)
    draw_stokes_poincare(S, ax=ax, label=label, **kwargs)

def join_stokes_poincare(S1, S2, ax=None, **kwargs):
    """Plot arc joining two Stokes vectors on Poincaré sphere."""
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        draw_empty_sphere(ax)

    SS1 = np.sqrt(S1[1]**2+S1[2]**2+S1[3]**2)
    SS2 = np.sqrt(S2[1]**2+S2[2]**2+S2[3]**2)
    x, y, z = great_circle_points(S1[1]/SS1, S1[2]/SS1, S1[3]/SS1, S2[1]/SS2, S2[2]/SS2, S2[3]/SS2)
    ax.plot(x, y, z, **kwargs)

def join_jones_poincare(J1, J2, ax=None, **kwargs):
    """Plot arc joining two Jones vectors on Poincaré sphere."""
    S1 = pypolar.jones.jones_to_stokes(J1)
    S2 = pypolar.jones.jones_to_stokes(J2)
    join_stokes_poincare(S1, S2, ax=ax, **kwargs)
