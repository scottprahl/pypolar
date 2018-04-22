"""
Useful basic routines for managing polarization using the Jones calculus

Todo:
    * improve documentation of each routine
    * modify interpret when phase difference differs by more than 2pi
    * improve interpret to give angle for elliptical polarization

Scott Prahl
Apr 2018
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from pypolar.fresnel import fresnel

#from IPython.display import HTML
import mpl_toolkits.mplot3d.axes3d as axes3d


__all__ = ['op_linear_polarizer',
           'op_retarder',
           'op_attenuator',
           'op_mirror',
           'op_rotation',
           'op_quarter_wave_plate',
           'op_half_wave_plate',
           'op_fresnel_reflection',
           'op_fresnel_transmission',
           'field_linear',
           'field_left_circular',
           'field_right_circular',
           'field_horizontal',
           'field_vertical',
           'draw_field',
           'draw_field_animated',
           'draw_field_ellipse',
           'interpret',
           'intensity',
           'phase',
           'ellipse_azimuth',
           'ellipse_ellipticity',
           'ellipse_psi',
           'ellipse_axes',
           'poincare_point',
           'jones_op_to_mueller_op']


def op_linear_polarizer(theta):
    """
    Jones matrix operator for a linear polarizer rotated about a normal to
    the surface of the polarizer.
    theta: rotation angle measured from the horizontal plane [radians]
    """

    return np.matrix([[np.cos(theta)**2, np.sin(theta) * np.cos(theta)],
                      [np.sin(theta) * np.cos(theta), np.sin(theta)**2]])


def op_retarder(theta, delta):
    """
    Jones matrix operator for an optical retarder rotated about a normal to
    the surface of the retarder.
    theta: rotation angle between fast-axis and the horizontal plane [radians]
    delta: phase delay introduced between fast and slow-axes         [radians]
    """
    P = np.exp(+delta / 2 * 1j)
    Q = np.exp(-delta / 2 * 1j)
    D = np.sin(delta / 2) * 2j
    C = np.cos(theta)
    S = np.sin(theta)
    return np.array([[C * C * P + S * S * Q, C * S * D],
                     [C * S * D, C * C * Q + S * S * P]])


def op_attenuator(od):
    """
    Jones matrix operator for an optical attenuator.
    od: base ten optical density  [---]
    """

    return np.matrix([[10**-od / 2, 0], [0, 10**-od / 2]])


def op_mirror():
    """
    Jones matrix operator for a perfect mirror.
    """
    return np.matrix([[1, 0], [0, -1]])


def op_rotation(theta):
    """
    Jones matrix operator to rotate light about the optical axis.

    Args:
        theta : angle of rotation about optical axis  [radians]
    Returns:
        2x2 matrix of the rotation operator           [-]
    """
    return np.array([[np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])


def op_quarter_wave_plate(theta):
    """
    Jones matrix operator for an quarter-wave plate rotated about a normal to
    the surface of the plate.

    Args:
        theta : angle from fast-axis to horizontal plane  [radians]
    Returns:
        2x2 matrix of the quarter-wave plate operator     [-]
    """

    return op_retarder(theta, np.pi / 2)


def op_half_wave_plate(theta):
    """
    Jones matrix operator for an half-wave plate rotated about a normal to
    the surface of the plate.

    Args:
        theta : angle from fast-axis to horizontal plane  [radians]
    Returns:
        2x2 matrix of the half-wave plate operator     [-]
    """

    return op_retarder(theta, np.pi)


def op_fresnel_reflection(m, theta):
    """
    Jones matrix operator for Fresnel reflection at angle theta
    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        2x2 matrix of the Fresnel transmission operator     [-]
    """
    return np.array([[fresnel.r_par(m, theta), 0],
                     [0, fresnel.r_per(m, theta)]])


def op_fresnel_transmission(m, theta):
    """
    Jones matrix operator for Fresnel transmission at angle theta

    *** THIS IS ALMOST CERTAINLY WRONG ***
    Args:
        m :     complex index of refraction       [-]
        theta : angle from normal to surface      [radians]
    Returns:
        2x2 Fresnel transmission operator           [-]
    """
    c = np.cos(theta)
    d = np.sqrt(m * m - np.sin(theta)**2, dtype=np.complex)
    if m.imag == 0:
        d = np.conjugate(d)
    a = np.sqrt(d/c)
    return a*np.array([[fresnel.t_par(m, theta), 0],
                     [0, fresnel.t_per(m, theta)]])


def field_linear(theta):
    """Jones vector for linear polarized light at angle theta from       horizontal plane"""

    return np.array([np.cos(theta), np.sin(theta)])


def field_right_circular():
    """Jones Vector corresponding to right circular polarized light"""

    return 1 / np.sqrt(2) * np.array([1, -1j])


def field_left_circular():
    """Jones Vector corresponding to left circular polarized light"""

    return 1 / np.sqrt(2) * np.array([1, 1j])


def field_horizontal():
    """Jones Vector corresponding to horizontal polarized light"""

    return field_linear(0)


def field_vertical():
    """Jones Vector corresponding to vertical polarized light"""

    return field_linear(np.pi / 2)


def interpret(J):
    '''
    Interprets a Jones vector (Original version by Alexander Miles 2013)

    Parameters
    J     : A Jones vector, may be complex

    Examples
    -------
    interpret([1, -1j]) --> "Right circular polarization"

    interpret([0.5, 0.5]) -->
                      "Linear polarization at 45.000000 degrees CCW from x-axis"

    interpret( np.array([exp(-1j*pi), exp(-1j*pi/3)]) ) -->
                "Left elliptical polarization, rotated with respect to the axes"
    '''

    try:
        j1, j2 = J
    except:
        print("Jones vector must have two elements")
        return 0

    eps = 1e-12
    mag1, p1 = abs(j1), np.angle(j1)
    mag2, p2 = abs(j2), np.angle(j2)

    if np.remainder(p1 - p2, np.pi) < eps:
        ang = np.arctan2(mag2, mag1) * 180 / np.pi
        return "Linear polarization at %f degrees CCW from x-axis" % ang

    if abs(mag1 - mag2) < eps:
        if abs(p1 - p2 - np.pi / 2) < eps:
            s = "Right circular polarization"
        elif p1 > p2:
            s = "Right elliptical polarization, rotated with respect to the axes"
        if (p1 - p2 + np.pi / 2) < eps:
            s = "Left circular polarization"
        elif p1 < p2:
            s = "Left elliptical polarization, rotated with respect to the axes"
    else:
        if p1 - p2 == np.pi / 2:
            s = "Right elliptical polarization, non-rotated"
        elif p1 > p2:
            s = "Right elliptical polarization, rotated with respect to the axes"
        if p1 - p2 == -np.pi / 2:
            s = "Left circular polarization, non-rotated"
        elif p1 < p2:
            s = "Left elliptical polarization, rotated with respect to the axes"
    return s


def normalize_vector(J):
    """
    Normalizes a vector by dividing each part by common number.
    After normalization the magnitude should be equal to ~1.
    """
    norm = np.linalg.norm(J)
    if norm == 0:
        return J
    return J / norm


def intensity(J):
    """
    Returns the intensity
    """
    return np.abs(J[0])**2 + np.abs(J[1])**2


def phase(J):
    """
    Returns the phase
    """
    gamma = np.angle(J[1]) - np.angle(J[0])
    return gamma


def ellipse_azimuth(J):
    """
    Returns the angle between the major semi-axis and the x-axis of
    the polarization ellipse.
    """
    Exo, Eyo = np.abs(J)
    alpha = np.arctan2(Eyo, Exo)
    return alpha


def ellipse_ellipticity(J):
    """
    Returns the ellipticty of the polarization ellipse.
    """
    delta = phase(J)
    psi = ellipse_psi(J)
    chi = 0.5 * np.arcsin(np.sin(2 * psi) * np.sin(delta))
    return chi


def ellipse_psi(J):
    """
    Returns the angle between the major semi-axis and the x-axis of
    the polarization ellipse.
    """
    Exo, Eyo = np.abs(J)
    delta = phase(J)
    numer = 2 * Exo * Eyo * np.cos(delta)
    denom = Exo**2 - Eyo**2
    psi = 0.5 * np.arctan2(numer, denom)
    return psi


def ellipse_axes(J):
    """
    Returns the semi-major and semi-minor axes of the polarization ellipse.
    """
    Exo, Eyo = np.abs(J)
    psi = ellipse_psi(J)
    delta = phase(J)
    C = np.cos(psi)
    S = np.sin(psi)
    asqr = (Exo * C)**2 + (Eyo * S)**2 + 2 * Exo * Eyo * C * S * np.cos(delta)
    bsqr = (Exo * S)**2 + (Eyo * C)**2 - 2 * Exo * Eyo * C * S * np.cos(delta)
    return np.sqrt(abs(asqr)), np.sqrt(abs(bsqr))


def poincare_point(J):
    """
    Returns the point the Poincaré sphere
    """
    longitude = 2 * ellipse_azimuth(J)
    a, b = ellipse_axes(J)
    latitude = 2 * np.arctan2(b, a)
    return latitude, longitude


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
    return


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
    return


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
    return


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
    return


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
    return


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
    return


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
    return


def draw_field_ellipse(J):
    """
    Draw a simple 2D representation of the projected field

    Args:
        J:      Jones vector
        ax:     matplotlib axis to use
        offset: starting point
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

    psi = ellipse_psi(J)
    M = np.sqrt(intensity(J))
    plt.plot([-M * np.cos(psi), M * np.cos(psi)],
             [-M * np.sin(psi), M * np.sin(psi)])
    plt.xlim(-the_max, the_max)
    plt.ylim(-the_max, the_max)
    plt.xticks([])
    plt.yticks([])
    return plt


def draw_field(J, offset=0):
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


def draw_field_animated(J):
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


def jones_op_to_mueller_op(J):
    """
    Converts a complex 2x2 Jones matrix to a real 4x4 Mueller matrix

    Hauge, Muller, and Smith, "Conventions and Formulas for Using the Mueller-
    Stokes Calculus in Ellipsometry," Surface Science, 96, 81-107 (1980)
    Args:
        J:      Jones matrix
    Returns
        equivalent 4x4 Mueller matrix
    """
    M = np.zeros(shape=[4, 4], dtype=np.complex)
    C = np.conjugate(J)
    M[0, 0] = J[0, 0] * C[0, 0] + J[0, 1] * C[0, 1] + \
        J[1, 0] * C[1, 0] + J[1, 1] * C[1, 1]
    M[0, 1] = J[0, 0] * C[0, 0] + J[1, 0] * C[1, 0] - \
        J[0, 1] * C[0, 1] - J[1, 1] * C[1, 1]
    M[0, 2] = J[0, 1] * C[0, 0] + J[1, 1] * C[1, 0] + \
        J[0, 0] * C[0, 1] + J[1, 0] * C[1, 1]
    M[0, 3] = 1j * (J[0, 1] * C[0, 0] + J[1, 1] * C[1, 0] -
                    J[0, 0] * C[0, 1] - J[1, 0] * C[1, 1])
    M[1, 0] = J[0, 0] * C[0, 0] + J[0, 1] * C[0, 1] - \
        J[1, 0] * C[1, 0] - J[1, 1] * C[1, 1]
    M[1, 1] = J[0, 0] * C[0, 0] - J[1, 0] * C[1, 0] - \
        J[0, 1] * C[0, 1] + J[1, 1] * C[1, 1]
    M[1, 2] = J[0, 0] * C[0, 1] + J[0, 1] * C[0, 0] - \
        J[1, 0] * C[1, 1] - J[1, 1] * C[1, 0]
    M[1, 3] = 1j * (J[0, 1] * C[0, 0] + J[1, 0] * C[1, 1] -
                    J[1, 1] * C[1, 0] - J[0, 0] * C[0, 1])
    M[2, 0] = J[0, 0] * C[1, 0] + J[1, 0] * C[0, 0] + \
        J[0, 1] * C[1, 1] + J[1, 1] * C[0, 1]
    M[2, 1] = J[0, 0] * C[1, 0] + J[1, 0] * C[0, 0] - \
        J[0, 1] * C[1, 1] - J[1, 1] * C[0, 1]
    M[2, 2] = J[0, 0] * C[1, 1] + J[0, 1] * C[1, 0] + \
        J[1, 0] * C[0, 1] + J[1, 1] * C[0, 0]
    M[2, 3] = 1j * (-J[0, 0] * C[1, 1] + J[0, 1] * C[1, 0] -
                    J[1, 0] * C[0, 1] + J[1, 1] * C[0, 0])
    M[3, 0] = 1j * (J[0, 0] * C[1, 0] + J[0, 1] * C[1, 1] -
                    J[1, 0] * C[0, 0] - J[1, 1] * C[0, 1])
    M[3, 1] = 1j * (J[0, 0] * C[1, 0] - J[0, 1] * C[1, 1] -
                    J[1, 0] * C[0, 0] + J[1, 1] * C[0, 1])
    M[3, 2] = 1j * (J[0, 0] * C[1, 1] + J[0, 1] * C[1, 0] -
                    J[1, 0] * C[0, 1] - J[1, 1] * C[0, 0])
    M[3, 3] = J[0, 0] * C[1, 1] - J[0, 1] * C[1, 0] - \
        J[1, 0] * C[0, 1] + J[1, 1] * C[0, 0]
    MM = M.real / 2
    return MM
