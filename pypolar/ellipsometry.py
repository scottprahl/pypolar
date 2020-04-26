# pylint: disable=invalid-name
"""
Useful functions for ellipsometry.

Scott Prahl
Apr 2020
"""

import numpy as np

__all__ = ('m_from_rho',
           'm_from_tanpsi_and_Delta',
           'rho_from_null_data',
           'rho_from_rotating_analyzer_data')


def m_from_rho(rho, theta_i):
    """
    Calculate the index of refraction for an isotropic sample.

    Formula from McCrackin "Measurement of the thickness and refractive
    index of very thin films and the optical properties of surfaces by
    ellipsometry", Journal of Research of the National Bureau of Standards,
    (1963).

    Args:
        rho :  r_par/r_per or tan(psi)*exp(j*Delta)        [-]
        theta_i : incidence angle from normal              [radians]
    Returns:
        complex index of refraction                        [-]
    """
    e_index = np.sqrt(1 - 4 * rho * np.sin(theta_i)**2 / (1 + rho)**2)
    return np.tan(theta_i) * e_index


def m_from_tanpsi_and_Delta(tanpsi, Delta, theta_i):
    """
    Return the index of refraction for observed Delta, tanpsi, and theta_i.

    Args:
        tanpsi : ratio of field amplitudes |r_par/r_per|   [-]
        Delta :  phase change caused by reflection         [-]
        theta_i : incidence angle from normal              [radians]
    Returns:
        complex index of refraction                        [-]
    """
    rho = tanpsi * np.exp(1j*Delta)
    return m_from_rho(rho, theta_i)


def rho_from_null_data(P1, P2, A1, A2):
    """
    Recover rho from Null ellipsometer measurements.
    """

    psi = np.pi/4 - (A1-A2)/2
    Delta = np.pi - (P1+P2)
    rho = np.tan(psi) * np.exp(1j*Delta)
    return rho


def rho_from_rotating_analyzer_data(phi, signal, P):
    """
    Recover rho from rotating analyzer data.

    This is done by fitting to

             I_DC + I_S*sin(2*phi)+I_C*cos(2*phi)

    Args:
        phi    - array of analyzer angles               [radians]
        signal - array of ellipsometer intensities      [AU]
        P      - incident polarization azimuthal angle  [radians]
    Returns:
        rho = tan(psi)*exp(1j*Delta)                    [-]
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
    rho = np.tan(psi) * np.exp(1j*Delta)

    return rho
