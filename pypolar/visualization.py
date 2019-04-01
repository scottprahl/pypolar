# pylint: disable=invalid-name
# pylint: disable=bare-except

"""
Useful basic routines for visualizing polarization

Todo:
    * improve documentation and testing

Scott Prahl
Mar 2019
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import pypolar.fresnel
import pypolar.mueller
import pypolar.jones

from mpl_toolkits.mplot3d import Axes3D

#from IPython.display import HTML
#import mpl_toolkits.mplot3d.axes3d as axes3d


__all__ = ['draw_jones_field',
           'draw_jones_animated',
           'draw_jones_ellipse',
           'draw_stokes_ellipse',
           'draw_stokes_field',
           'draw_stokes_animated']


def _draw_optical_axis_3d(J, ax, last=4 * np.pi):
    """
    Draw the optical axis in a 3D plot

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


def _draw_h_field_3d(J, ax, offset, last=4 * np.pi):
    """
    Draw the horizontal electric field in a 3D plot

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
    Draw the vertical electric field in a 3D plot

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
    Draw the total electric field in a 3D plot

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
    Draw the projection vector of the polarization field in 3D

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
    ax.plot([x2], [y2], [z2], 'ro')


def _draw_3D_field(J, ax, offset):
    """
    Draw a representation of the polarization fields in 3D

    Args:
        J:      Jones vector
        ax:     matplotlib axis to use
        offset: starting point
    """
    _draw_optical_axis_3d(J, ax, offset)
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
    Draw a simple 2D representation of the projected field

    Args:
        J:      Jones vector
        ax:     matplotlib axis to use
        offset: starting point
    """
    h_amp, v_amp = np.abs(J)
    h_phi, v_phi = np.angle(J)
    the_max = max(h_amp, v_amp) * 1.1

    plt.plot([-the_max, the_max], [0, 0], 'g')
    plt.plot([0, 0], [-the_max, the_max], 'b')

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


def draw_jones_ellipse(J):
    """
    Draw a simple 2D representation of the projected field

    Args:
        J:      Jones vector
    """
    Exo, Eyo = np.abs(J)
    phix, phiy = np.angle(J)
    the_max = max(Exo, Eyo) * 1.2

    the_max = the_max
    the_max = the_max
    plt.axes().set_aspect('equal')
    plt.plot([-the_max, the_max], [0, 0], 'k')
    plt.plot([0, 0], [-the_max, the_max], 'k')

    plt.plot([-Exo, -Exo, Exo, Exo, -Exo], [-Eyo, Eyo, Eyo, -Eyo, -Eyo], ':g')
    plt.annotate(r' $E_{x0}$', xy=(Exo, 0), va='bottom', ha='left')
    plt.annotate(r'$-E_{x0} $', xy=(-Exo, 0), va='bottom', ha='right')
    plt.annotate(r'$E_{y0}$', xy=(0, Eyo), va='bottom', ha='left')
    plt.annotate(r'$-E_{y0}$', xy=(0, -Eyo), va='top', ha='left')
    plt.annotate(r'  $\psi$', xy=(0, 0), va='bottom', ha='left')

    t = np.linspace(0, 2 * np.pi, 100)
    plt.plot(Exo * np.cos(t + phix), Eyo * np.cos(t + phiy), 'b')

    psi = pypolar.jones.ellipse_orientation(J)
    M = np.sqrt(pypolar.jones.intensity(J))
    plt.plot([-M * np.cos(psi), M * np.cos(psi)],
             [-M * np.sin(psi), M * np.sin(psi)])
    plt.xlim(-the_max, the_max)
    plt.ylim(-the_max, the_max)
    plt.xticks([])
    plt.yticks([])
    return plt


def draw_stokes_ellipse(S):
    """
    Draw a simple 2D representation of the projected field

    Args:
        S:      Stokes vector
    """
    Exo = np.sqrt((S[0]+S[1])/2)
    Eyo = np.sqrt((S[0]-S[1])/2)
    phix = 0
    phiy = np.arcsin(S[2]/(2*Exo*Eyo))
    the_max = max(Exo, Eyo) * 1.2

    the_max = the_max
    the_max = the_max
    plt.axes().set_aspect('equal')
    plt.plot([-the_max, the_max], [0, 0], 'k')
    plt.plot([0, 0], [-the_max, the_max], 'k')

    plt.plot([-Exo, -Exo, Exo, Exo, -Exo], [-Eyo, Eyo, Eyo, -Eyo, -Eyo], ':g')
    plt.annotate(r' $E_{x0}$', xy=(Exo, 0), va='bottom', ha='left')
    plt.annotate(r'$-E_{x0} $', xy=(-Exo, 0), va='bottom', ha='right')
    plt.annotate(r'$E_{y0}$', xy=(0, Eyo), va='bottom', ha='left')
    plt.annotate(r'$-E_{y0}$', xy=(0, -Eyo), va='top', ha='left')
    plt.annotate(r'  $\psi$', xy=(0, 0), va='bottom', ha='left')

    t = np.linspace(0, 2 * np.pi, 100)
    plt.plot(Exo * np.cos(t + phix), Eyo * np.cos(t + phiy), 'b')

    psi = pypolar.mueller.ellipse_orientation(S)
    M = np.sqrt(pypolar.mueller.intensity(S))
    plt.plot([-M * np.cos(psi), M * np.cos(psi)],
             [-M * np.sin(psi), M * np.sin(psi)])
    plt.xlim(-the_max, the_max)
    plt.ylim(-the_max, the_max)
    plt.xticks([])
    plt.yticks([])
    return plt


def draw_jones_field(J, offset=0):
    """
    Draw 3D and 2D representations of the polarization field

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
    return plt


def _animation_update(offset, J, ax1, ax2):
    """
    function to draw the next animation frame

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


def draw_jones_animated(J):
    """
    Animate 3D and 2D representations of the polarization field

    Args:
        J:      Jones vector
    """
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    ax1 = plt.subplot(gs[0], projection='3d')
    _draw_3D_field(J, ax1, 0)

    ax2 = plt.subplot(gs[1])
    _draw_2D_field(J, ax2, 0)

    ani = animation.FuncAnimation(fig, _animation_update,
                                  frames=np.linspace(0, 2 * np.pi, 64),
                                  fargs=(J, ax1, ax2))
    return ani


def draw_stokes_ellipse(S):
    """
    Draw a 2D representation of the polarization state of S

    Args:
        S:      Stokes vector
        offset: starting point
    Returns:
        a matplotlib object with the graph
    """
    J = pypolar.mueller.stokes_to_jones(S)
    aplt = draw_jones_ellipse(J)
    return aplt


def draw_stokes_field(S, offset=0):
    """
    Draw a 2D and 3D representation of the polarization

    Args:
        S:      Stokes vector
        offset: starting point
    """
    J = pypolar.mueller.stokes_to_jones(S)
    aplt = draw_jones_field(J, offset)
    return aplt


def draw_stokes_animated(S):
    """
    Draw animated 2D and 3D representations of the polarization

    Args:
        S:      Stokes vector
    """
    J = pypolar.mueller.stokes_to_jones(S)
    ani = draw_jones_animated(J)
    return ani
