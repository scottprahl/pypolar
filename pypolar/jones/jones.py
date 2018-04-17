"""
Useful basic routines for managing polarization using the Jones calculus

Todo:
    * improve documentation of each routine
    * modify interpret when phase difference differs by more than 2pi
    * improve interpret to give angle for elliptical polarization

Jones' First Law:  (albeit a different Jones)
	Anyone who makes a significant contribution to any field of
	endeavor, and stays in that field long enough, becomes an
	obstruction to its progress --- in direct proportion to the
	importance of their original contribution.
	
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
           'draw_3D_field',
           'draw_2D_field',
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
           'poincare_point']


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
    return np.array([[-fresnel.r_par(m, theta), 0],
                     [0, fresnel.r_per(m, theta)]])


def op_fresnel_transmission(m, theta):
    """
    Jones matrix operator for Fresnel transmission at angle theta

    Args:
        m :     complex index of refraction       [-]
        theta : angle from normal to surface      [radians]
    Returns:
        2x2 Fresnel transmission operator           [-]
    """
    return np.array([[fresnel.t_par(m, theta), 0],
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


def interpret(v):
    '''
    Interprets a Jones vector (Original version by Alexander Miles 2013)

    Parameters
    v     : A Jones vector, may be complex

    Examples
    -------
    interpret([1, -1j]) --> "Right circular polarization"

    interpret([0.5, 0.5]) -->
                      "Linear polarization at 45.000000 degrees CCW from x-axis"

    interpret( np.array([exp(-1j*pi), exp(-1j*pi/3)]) ) -->
                "Left elliptical polarization, rotated with respect to the axes"
    '''

    try:
        j1, j2 = v
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


def normalize_vector(v):
    """
    Normalizes a vector by dividing each part by common number.
    After normalization the magnitude should be equal to ~1.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def intensity(v):
    """
    Returns the intensity
    """
    return np.abs(v[0])**2 + np.abs(v[1])**2


def phase(v):
    """
    Returns the phase
    """
    gamma = np.angle(v[1]) - np.angle([0])
    return gamma


def ellipse_azimuth(v):
    """
    Returns the angle between the major semi-axis and the x-axis of
    the polarization ellipse.
    """
    Exo, Eyo = np.abs(v)
    alpha = np.arctan2(Eyo, Exo)
    return alpha


def ellipse_ellipticity(v):
    """
    Returns the ellipticty of the polarization ellipse.
    """
    delta = phase(v)
    psi = ellipse_psi(v)
    chi = 0.5 * np.arcsin(np.sin(2 * psi) * np.sin(delta))
    return chi


def ellipse_psi(v):
    """
    Returns the angle between the major semi-axis and the x-axis of
    the polarization ellipse.
    """
    Exo, Eyo = np.abs(v)
    delta = phase(v)
    numer = 2 * Exo * Eyo * np.cos(delta)
    denom = Exo**2 - Eyo**2
    psi = 0.5 * np.arctan2(numer, denom)
    return psi


def ellipse_axes(v):
    """
    Returns the semi-major and semi-minor axes of the polarization ellipse.
    """
    Exo, Eyo = np.abs(v)
    psi = ellipse_psi(v)
    delta = phase(v)
    C = np.cos(psi)
    S = np.sin(psi)
    asqr = (Exo * C)**2 + (Eyo * S)**2 + 2 * Exo * Eyo * C * S * np.cos(delta)
    bsqr = (Exo * S)**2 + (Eyo * C)**2 - 2 * Exo * Eyo * C * S * np.cos(delta)
    return np.sqrt(abs(asqr)), np.sqrt(abs(bsqr))


def poincare_point(v):
    """
    Returns the point the Poincaré sphere
    """
    longitude = 2 * ellipse_azimuth(v)
    a, b = ellipse_axes(v)
    latitude = 2 * np.arctan2(b, a)
    return latitude, longitude


def _drawAxis(v, ax, last=4 * np.pi):
    Ha, Va = np.abs(v)
    the_max = max(Ha, Va) * 1.1

    ax.plot([0, last], [0, 0], [0, 0], 'k')
    ax.plot([0, 0], [-the_max, the_max], [0, 0], 'g')
    ax.plot([0, 0], [0, 0], [-the_max, the_max], 'b')
    return


def _drawHwave(v, ax, last=4 * np.pi, offset=0):
    Hamp = abs(v[0])
    Hshift = np.angle(v[0])

    t = np.linspace(0, last, 100) + offset
    x = t - offset
    y = Hamp * np.cos(t - Hshift)
    z = 0 * t
    ax.plot(x, y, z, ':g')
    return


def _drawVwave(v, ax, last=4 * np.pi, offset=0):
    Vamp = abs(v[1])
    Vshift = np.angle(v[1])

    t = np.linspace(0, last, 100) + offset
    x = t - offset
    y = 0 * t
    z = Vamp * np.cos(t - Vshift)
    ax.plot(x, y, z, ':b')
    return


def _drawSumwave(v, ax, last=4 * np.pi, offset=0):
    Hamp, Vamp = np.abs(v)
    Hshift, Vshift = np.angle(v)

    t = np.linspace(0, last, 100) + offset
    x = t - offset
    yH = 0 * t
    yV = Hamp * np.cos(t - Hshift)
    zH = 0 * t
    zV = Vamp * np.cos(t - Vshift)
    y = yH + yV
    z = zH + zV
    ax.plot(x, y, z, 'r')
    return


def _drawVectorSum(v, ax, offset=0):
    Hamp, Vamp = np.abs(v)
    Hshift, Vshift = np.angle(v)

    t = offset
    yH = 0 * t
    yV = Hamp * np.cos(t - Hshift)
    zH = 0 * t
    zV = Vamp * np.cos(t - Vshift)
    y = yH + yV
    z = zH + zV

    x1 = 0
    y1 = y
    z1 = 0
    x2 = 0
    y2 = y
    z2 = z
    ax.plot([x1, x2], [y1, y2], [z1, z2], 'g--')

    x1 = 0
    y1 = 0
    z1 = z
    x2 = 0
    y2 = y
    z2 = z
    ax.plot([x1, x2], [y1, y2], [z1, z2], 'b--')

    x1 = 0
    y1 = 0
    z1 = 0
    x2 = 0
    y2 = y
    z2 = z
    ax.plot([x1, x2], [y1, y2], [z1, z2], 'r')
    ax.plot([x2], [y2], [z2], 'ro')
    return


def draw_3D_field(v, ax, off=0):
    """Draw a 3D representation of the fields for a Jones vector"""

    _drawAxis(v, ax)
    _drawHwave(v, ax, offset=off)
    _drawVwave(v, ax, offset=off)
    _drawSumwave(v, ax, offset=off)
    _drawVectorSum(v, ax, offset=off)

    ax.grid(False)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return


def draw_2D_field(v, ax, off=0):
    Ha, Va = np.abs(v)
    Hoffset, Voffset = np.angle(v)
    the_max = max(Ha, Va) * 1.1

    xmax = the_max
    ymax = the_max
    ax.plot([-xmax, xmax], [0, 0], 'g')
    ax.plot([0, 0], [-ymax, ymax], 'b')

    t = np.linspace(0, 2 * np.pi, 100) + off
    x = Ha * np.cos(t - Hoffset)
    y = Va * np.cos(t - Voffset)
    ax.plot(x, y, 'k')

    t = off
    x = Ha * np.cos(t - Hoffset)
    y = Va * np.cos(t - Voffset)
    ax.plot(x, y, 'ro')
    ax.plot([x, x], [0, y], 'g--')
    ax.plot([0, x], [y, y], 'b--')
    ax.plot([0, x], [0, y], 'r')

    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return


def draw_field(v, offset=0):
    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0], projection='3d')
    ax2 = plt.subplot(gs[1])
    draw_3D_field(v, ax1, off=offset)
    draw_2D_field(v, ax2, off=offset)
    return plt


def draw_field_ellipse(v):
    Exo, Eyo = np.abs(v)
    phix, phiy = np.angle(v)
    the_max = max(Exo, Eyo) * 1.2

    xmax = the_max
    ymax = the_max
    plt.axes().set_aspect('equal')
    plt.plot([-xmax, xmax], [0, 0], 'k')
    plt.plot([0, 0], [-ymax, ymax], 'k')

    plt.plot([-Exo, -Exo, Exo, Exo, -Exo], [-Eyo, Eyo, Eyo, -Eyo, -Eyo], ':g')
    plt.annotate(r' $E_{x0}$', xy=(Exo, 0), va='bottom', ha='left')
    plt.annotate(r'$-E_{x0} $', xy=(-Exo, 0), va='bottom', ha='right')
    plt.annotate(r'$E_{y0}$', xy=(0, Eyo), va='bottom', ha='left')
    plt.annotate(r'$-E_{y0}$', xy=(0, -Eyo), va='top', ha='left')
    plt.annotate(r'  $\psi$', xy=(0, 0), va='bottom', ha='left')

    t = np.linspace(0, 2 * np.pi, 100)
    plt.plot(Exo * np.cos(t + phix), Eyo * np.cos(t + phiy), 'b')

    psi = ellipse_psi(v)
    M = np.sqrt(intensity(v))
    plt.plot([-M * np.cos(psi), M * np.cos(psi)],
             [-M * np.sin(psi), M * np.sin(psi)])
    plt.xlim(-xmax, xmax)
    plt.ylim(-ymax, ymax)
    plt.xticks([])
    plt.yticks([])
    return plt


def _ani_update(startAngle, v, ax1, ax2):
    ax1.clear()
    ax2.clear()
    draw_3D_field(v, ax1, off=startAngle)
    draw_2D_field(v, ax2, off=startAngle)
    return ax1, ax2


def draw_field_animated(v):
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0], projection='3d')
    ax2 = plt.subplot(gs[1])

    startAngle = 0
    draw_3D_field(v, ax1, off=startAngle)
    draw_2D_field(v, ax2, off=startAngle)

    ani = animation.FuncAnimation(fig, _ani_update, frames=np.linspace(
        0, 2 * np.pi, 64), fargs=(v, ax1, ax2))
    return ani
