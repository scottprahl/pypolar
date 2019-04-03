# pylint: disable=invalid-name
# pylint: disable=bare-except

"""
Useful basic routines for managing polarization using the Jones calculus

Todo:
    * improve documentation of each routine
    * modify interpret() when phase difference differs by more than 2pi
    * improve interpret() to give angle for elliptical polarization

Scott Prahl
Apr 2018
"""

import numpy as np
import pypolar.fresnel

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
           'interpret',
           'intensity',
           'phase',
           'ellipse_azimuth',
           'ellipse_ellipticity',
           'ellipse_orientation',
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


def op_attenuator(t):
    """
    Jones matrix operator for an optical attenuator.
    f: fraction of light getting through attenuator  [---]
    """
    return np.matrix([[t / 2, 0], [0, t / 2]])


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
    return np.array([[pypolar.fresnel.r_par(m, theta), 0],
                     [0, pypolar.fresnel.r_per(m, theta)]])


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
    return a*np.array([[pypolar.fresnel.t_par(m, theta), 0], [0, pypolar.fresnel.t_per(m, theta)]])


def field_linear(theta):
    """Jones vector for linear polarized light at angle theta from horizontal plane"""

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
    inten = np.conjugate(J.T) * J
    return inten[0]


def phase(J):
    """
    Returns the phase
    """
    gamma = np.angle(J[1]) - np.angle(J[0])
    return gamma


def ellipse_orientation(J):
    """
    Returns the angle between the major semi-axis and the x-axis of
    the polarization ellipse (sometimes called the azimuth or psi)
    """
    Exo, Eyo = np.abs(J)
    delta = phase(J)
    numer = 2 * Exo * Eyo * np.cos(delta)
    denom = Exo**2 - Eyo**2
    psi = 0.5 * np.arctan2(numer, denom)
    return psi


def ellipse_ellipticity(J):
    """
    Returns the ellipticty of the polarization ellipse.
    """
    delta = phase(J)
    psi = ellipse_orientation(J)
    chi = 0.5 * np.arcsin(np.sin(2 * psi) * np.sin(delta))
    return chi


def ellipse_azimuth(J):
    """
    Returns the angle between the major semi-axis and the x-axis of
    the polarization ellipse.
    """
    Exo, Eyo = np.abs(J)
    alpha = np.arctan2(Eyo, Exo)
    return alpha


def ellipse_axes(J):
    """
    Returns the semi-major and semi-minor axes of the polarization ellipse.
    """
    Exo, Eyo = np.abs(J)
    psi = ellipse_orientation(J)
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
    longitude = 2 * ellipse_orientation(J)
    a, b = ellipse_axes(J)
    latitude = 2 * np.arctan2(b, a)
    return latitude, longitude


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
