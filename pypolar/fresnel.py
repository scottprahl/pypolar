"""
Useful basic routines for managing Fresnel reflection

Todo:
    * improve documentation of each routine

Scott Prahl
Apr 2018
"""

import numpy as np

__all__ = ['r_par',
           'r_per',
           'R_par',
           'R_per',
           'R_unpolarized']


def r_par(m, theta):
    """
    Calculates the reflected amplitude for parallel polarized light

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        reflected field amplitude             [-]
    """
    m2 = m * m
    c = np.cos(theta)
    s = np.sin(theta)
    d = np.sqrt(m2 - s * s)
    return (m2 * c - d) / (m2 * c + d)


def r_per(m, theta):
    """
    Calculates the reflected amplitude for perpendicular polarized light

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        reflected field amplitude             [-]
    """
    m2 = m * m
    c = np.cos(theta)
    s = np.sin(theta)
    d = np.sqrt(m2 - s * s)
    return (c - d) / (c + d)


def R_par(m, theta):
    """
    Calculates the reflected irradiance for parallel polarized light

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        reflected power                       [-]
    """
    return abs(r_par(m, theta))**2


def R_per(m, theta):
    """
    Calculates the reflected irradiance for perpendicular polarized light

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        reflected irradiance                  [-]
    """
    return abs(r_per(m, theta))**2


def R_unpolarized(m, theta):
    """
    Calculates the reflected irradiance for unpolarized incident light

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        reflected irradiance                  [-]
    """
    return (R_par(m, theta) + R_per(m, theta)) / 2
