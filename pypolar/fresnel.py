# pylint: disable=invalid-name
"""
Useful basic routines for managing Fresnel reflection

To do:
    * Make sure routines work for arrays of m or of theta_i
    * add ellipsometry routines for one layer

Scott Prahl
Apr 2018
"""

import numpy as np

__all__ = ['r_par',
           'r_per',
           't_par',
           't_per',
           'R_par',
           'R_per',
           'T_par',
           'T_per',
           'R_unpolarized',
           'T_unpolarized',
           'ellipsometry_rho',
           'ellipsometry_index',
           'ellipsometry_parameters']


def r_par(m, theta_i):
    """
    Calculates the reflected amplitude for parallel polarized light

    Args:
        m :       complex index of refraction   [-]
        theta_i : angle from normal to surface  [radians]
    Returns:
        reflected fraction of parallel field    [-]
    """
    c = m * m * np.cos(theta_i)
    s = np.sin(theta_i)
    d = np.sqrt(m * m - s * s, dtype=np.complex) # m*cos(theta_t)
    if m.imag == 0:
        d = np.conjugate(d)
    rp = (c - d) / (c + d)
    return np.real_if_close(rp)


def r_per(m, theta_i):
    """
    Calculates the reflected amplitude for perpendicular polarized light

    Args:
        m :       complex index of refraction     [-]
        theta_i : incidence angle from normal     [radians]
    Returns:
        reflected fraction of perpendicular field [-]
    """
    c = np.cos(theta_i)
    s = np.sin(theta_i)
    d = np.sqrt(m * m - s * s, dtype=np.complex) # m*cos(theta_t)
    if m.imag == 0:
        d = np.conjugate(d)
    rs = (c - d) / (c + d)
    return np.real_if_close(rs)


def t_par(m, theta_i):
    """
    Calculates the transmitted amplitude for parallel polarized light

    Args:
        m :       complex index of refraction  [-]
        theta_i : incidence angle from normal  [radians]
    Returns:
        transmitted fraction of parallel field [-]
    """
    c = np.cos(theta_i)
    s = np.sin(theta_i)
    d = np.sqrt(m * m - s * s, dtype=np.complex) # m*cos(theta_t)
    if m.imag == 0:
        d = np.conjugate(d)
    tp = 2 * c * m / (m * m * c + d)
    return np.real_if_close(tp)


def t_per(m, theta_i):
    """
    Calculates the transmitted amplitude for perpendicular polarized light

    Args:
        m :     complex index of refraction         [-]
        theta_i : incidence angle from normal       [radians]
    Returns:
        transmitted fraction of perpendicular field [-]
    """
    c = np.cos(theta_i)
    s = np.sin(theta_i)
    d = np.sqrt(m * m - s * s, dtype=np.complex) # m*cos(theta_t)
    if m.imag == 0:
        d = np.conjugate(d)
    ts = 2 * c / (c + d)
    return np.real_if_close(ts)


def R_par(m, theta_i):
    """
    Fraction of parallel-polarized light that is reflected (R_p).

    Calculates reflected fraction of incident power (or flux) assuming that
    the E-field of the incident light is parallel to the plane of incidence

    Args:
        m :       complex index of refraction [-]
        theta_i : incidence angle from normal [radians]
    Returns:
        reflected power                       [-]
    """
    return abs(r_par(m, theta_i))**2


def R_per(m, theta_i):
    """
    Fraction of perpendicular-polarized light that is reflected (R_s).

    Calculates reflected fraction of incident power (or flux) assuming that
    the E-field of the incident light is perpendicular to the plane of incidence

    Args:
        m :     complex index of refraction   [-]
        theta_i : incidence angle from normal [radians]
    Returns:
        reflected irradiance                  [-]
    """
    return abs(r_per(m, theta_i))**2


def T_par(m, theta_i):
    """
    Fraction of parallel-polarized light that is transmitted (T_p).

    Calculates transmitted fraction of incident power (or flux) assuming that
    the E-field of the incident light is parallel to the plane of incidence

    Args:
        m :     complex index of refraction   [-]
        theta_i : incidence angle from normal [radians]
    Returns:
        transmitted irradiance                [-]
    """
    c = np.cos(theta_i)
    s = np.sin(theta_i)
    d = np.sqrt(m * m - s * s, dtype=np.complex) # m*cos(theta_t)
    tp = 2 * c * m / (m * m * c + d)
    return np.real(d / c * abs(tp)**2)


def T_per(m, theta_i):
    """
    Fraction of perpendicular-polarized light that is transmitted (T_s).

    Calculates transmitted fraction of incident power (or flux) assuming that
    the E-field of the incident light is perpendicular to the plane of incidence

    Args:
        m :     complex index of refraction   [-]
        theta_i : incidence angle from normal [radians]
    Returns:
        transmitted field amplitude           [-]
    """
    c = np.cos(theta_i)
    s = np.sin(theta_i)
    d = np.sqrt(m * m - s * s, dtype=np.complex) # m*cos(theta_t)
    ts = 2 * c / (c + d)
    return np.real(d / c * abs(ts)**2)


def R_unpolarized(m, theta_i):
    """
    Fraction of unpolarized light that is reflected.

    Calculates reflection fraction of incident power (or flux) assuming that
    the incident light is unpolarized

    Args:
        m :     complex index of refraction   [-]
        theta_i : incidence angle from normal [radians]
    Returns:
        reflected irradiance                  [-]
    """
    return (R_par(m, theta_i) + R_per(m, theta_i)) / 2


def T_unpolarized(m, theta_i):
    """
    Fraction of unpolarized light that is transmitted.

    Calculates transmitted fraction of incident power (or flux) assuming that
    the incident light is unpolarized

    Args:
        m :     complex index of refraction   [-]
        theta_i : incidence angle from normal [radians]
    Returns:
        reflected irradiance                  [-]
    """
    return (T_par(m, theta_i) + T_per(m, theta_i)) / 2


def ellipsometry_rho(m, theta_i):
    """
    Calculate the ellipsometer parameter rho

    Args:
        m :     complex index of refraction   [-]
        theta_i : incidence angle from normal [radians]
    Returns:
        ellipsometer parameter rho            [-]
    """
    return r_par(m, theta_i) / r_per(m, theta_i)


def ellipsometry_index(rho, theta_i):
    """
    Calculate the index of refraction for an isotropic sample

    Args:
        rho :  r_par/r_per                    [-]
        theta_i : incidence angle from normal [radians]
    Returns:
        complex index of refraction           [-]
    """
    e_index = np.sqrt(1 - 4 * rho * np.sin(theta_i)**2 / (1 + rho)**2)
    return np.tan(theta_i) * e_index


def ellipsometry_parameters(phi, signal, P):
    """
    Recover ellipsometer parameters Delta and tan(psi) by fitting to

             I_DC + I_S*sin(2*phi)+I_C*cos(2*phi)

    Args:
        phi    - array of analyzer angles               [radians]
        signal - array of ellipsometer intensities      [AU]
        P      - incident polarization azimuthal angle  [radians]
    """

    I_DC = np.average(signal)
    I_S = 2 * np.average(signal * np.sin(2 * phi))
    I_C = 2 * np.average(signal * np.cos(2 * phi))

    tanP = np.tan(P)
    arg = I_S / np.sqrt(abs(I_DC**2 - I_C**2)) * np.sign(tanP)
    if arg > 1:
        Delta = 0
    elif arg < -1:
        Delta = np.pi
    else:
        Delta = np.arccos(arg)

    tanpsi = np.sqrt(abs(I_DC + I_C) / abs(I_DC - I_C)) * np.abs(tanP)

    return Delta, tanpsi
