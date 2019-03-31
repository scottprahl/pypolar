# pylint: disable=invalid-name
"""
Useful basic routines for managing symbolic Fresnel reflection

To do:
    * tests and documentation

Scott Prahl
Apr 2019
"""

import sympy

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
           'ellipsometry_index']


def r_par(m, theta_i):
    """
    Calculates the reflected amplitude for parallel polarized light

    Args:
        m :       complex index of refraction   [-]
        theta_i : angle from normal to surface  [radians]
    Returns:
        reflected fraction of parallel field    [-]
    """
    c = m * m * sympy.cos(theta_i)
    s = sympy.sin(theta_i)
    d = sympy.sqrt(m * m - s * s) # m*cos(theta_t)
    if m.imag == 0:
        d = sympy.conjugate(d)
    rp = (c - d) / (c + d)
    return rp


def r_per(m, theta_i):
    """
    Calculates the reflected amplitude for perpendicular polarized light

    Args:
        m :       complex index of refraction     [-]
        theta_i : incidence angle from normal     [radians]
    Returns:
        reflected fraction of perpendicular field [-]
    """
    c = sympy.cos(theta_i)
    s = sympy.sin(theta_i)
    d = sympy.sqrt(m * m - s * s) # m*cos(theta_t)
    if m.imag == 0:
        d = sympy.conjugate(d)
    rs = (c - d) / (c + d)
    return rs


def t_par(m, theta_i):
    """
    Calculates the transmitted amplitude for parallel polarized light

    Args:
        m :       complex index of refraction  [-]
        theta_i : incidence angle from normal  [radians]
    Returns:
        transmitted fraction of parallel field [-]
    """
    c = sympy.cos(theta_i)
    s = sympy.sin(theta_i)
    d = sympy.sqrt(m * m - s * s) # m*cos(theta_t)
    if m.imag == 0:
        d = sympy.conjugate(d)
    tp = 2 * c * m / (m * m * c + d)
    return tp


def t_per(m, theta_i):
    """
    Calculates the transmitted amplitude for perpendicular polarized light

    Args:
        m :     complex index of refraction         [-]
        theta_i : incidence angle from normal       [radians]
    Returns:
        transmitted fraction of perpendicular field [-]
    """
    c = sympy.cos(theta_i)
    s = sympy.sin(theta_i)
    d = sympy.sqrt(m * m - s * s) # m*cos(theta_t)
    if m.imag == 0:
        d = sympy.conjugate(d)
    ts = 2 * c / (c + d)
    return ts


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
    return sympy.abs(r_par(m, theta_i))**2


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
    return sympy.abs(r_per(m, theta_i))**2


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
    c = sympy.cos(theta_i)
    s = sympy.sin(theta_i)
    d = sympy.sqrt(m * m - s * s) # m*cos(theta_t)
    tp = 2 * c * m / (m * m * c + d)
    return d / c * sympy.abs(tp)**2


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
    c = sympy.cos(theta_i)
    s = sympy.sin(theta_i)
    d = sympy.sqrt(m * m - s * s) # m*cos(theta_t)
    ts = 2 * c / (c + d)
    return d / c * sympy.abs(ts)**2


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
    e_index = sympy.sqrt(1 - 4 * rho * sympy.sin(theta_i)**2 / (1 + rho)**2)
    return sympy.tan(theta_i) * e_index

