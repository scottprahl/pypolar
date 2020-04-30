# pylint: disable=invalid-name
# pylint: disable=bare-except
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=unused-import

"""
Useful basic routines for visualizing polarization.

To Do
    * re-orient so xyz match xyz
    *

Scott Prahl
Mar 2019
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
           'draw_stokes_animated')

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
    Helper function to draw the next animation frame.

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


def draw_jones_ellipse(J, simple_plot=False):
    """
    Draw a 2D sectional pattern for a Jones vector.

    Args:
        J:      Jones vector
    """
    JJ = J
    if pypolar.jones.alternate_sign_convention:
        JJ = np.conjugate(J)
    Ex0, Ey0 = np.abs(JJ)
    phix, phiy = np.angle(JJ)

    alpha = pypolar.jones.ellipse_azimuth(JJ)
    psi = pypolar.jones.amplitude_ratio_angle(JJ)
    a, b = pypolar.jones.ellipse_axes(JJ)

    t = np.linspace(0, 2 * np.pi, 100)
    xx = Ex0 * np.cos(t + phix)
    yy = Ey0 * np.cos(t + phiy)

    the_max = max(Ex0, Ey0) * 1.2

    if simple_plot:
        plt.plot(xx, yy, 'b')
        return

    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax1 = plt.subplot(gs[0])
    ax1.set_aspect('equal')
    ax1.plot(xx, yy, 'b')

    # semi-major diameter
    dx = a * np.cos(alpha)
    dy = a * np.sin(alpha)
    ax1.plot([0, dx], [0, dy], 'r')
    ax1.text(dx/2, dy/2, '  a', color='red')
    ax1.text(dx/5, dy/10, r'$\alpha$', va='center', ha='center')
    s = r'a=%.2f, b=%.2f, $\alpha$=%.2f°' % (a, b, np.degrees(alpha))
    ax1.text(0, -1.15*the_max, s, ha='center')

    # semi-minor diameter
    alpha += np.pi/2
    dx = b * np.cos(alpha)
    dy = b * np.sin(alpha)
    ax1.plot([0, dx], [0, dy], 'g')
    ax1.text(dx/2, dy/2, '  b', color='green')
    s = r'b/a=%.2f, ' % (b/a)
    s += r'$\tan^{-1}(b/a)$=%.2f°' % np.degrees(pypolar.jones.ellipticity_angle(JJ))
    ax1.text(0, -1.30*the_max, s, ha='center')

    # draw x and y axes
    ax1.plot([0, 0], [-the_max, the_max], 'k')
    ax1.plot([-the_max, the_max], [0, 0], 'k')
    ax1.set_xlim(-the_max, the_max)
    ax1.set_ylim(-the_max, the_max)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = plt.subplot(gs[1])
    ax2.set_aspect('equal')
    ax2.plot(xx, yy, 'b')
    ax2.plot([-Ex0, -Ex0, Ex0, Ex0, -Ex0], [-Ey0, Ey0, Ey0, -Ey0, -Ey0], ':g')
    ax2.plot([-Ex0, Ex0], [-Ey0, Ey0], ':r')
    ax2.plot([0, 0], [-the_max, the_max], 'k')
    ax2.plot([-the_max, the_max], [0, 0], 'k')
    ax2.text(Ex0, 0, r' $E_{x0}$', va='bottom', ha='left')
    ax2.text(-Ex0, 0, r'$-E_{x0} $', va='bottom', ha='right')
    ax2.text(0, Ey0, r'$E_{y0}$', va='bottom', ha='left')
    ax2.text(0, -Ey0, r'$-E_{y0}$', va='top', ha='left')
    ax2.text(Ex0/5, Ey0/10, r'$\psi$', va='center', ha='center')
    ax2.set_xlim(-the_max, the_max)
    ax2.set_ylim(-the_max, the_max)
    ax2.set_xticks([])
    ax2.set_yticks([])
    psi = np.degrees(np.arctan2(Ey0, Ex0))
    s = r'$E_{0x}$=%.2f, $E_{0y}$=%.2f, $\psi$=%.2f°' % (Ex0, Ey0, psi)
    ax2.text(0, -1.15*the_max, s, ha='center')
    s = r'$\phi_x$=%.2f°, ' % np.degrees(phix)
    s += r'$\phi_y$=%.2f°, ' % np.degrees(phiy)
    s += r'$\phi_y-\phi_x$=%.2f°' % np.degrees(phiy-phix)
    ax2.text(0, -1.30*the_max, s, ha='center')


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
