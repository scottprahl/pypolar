# pylint: disable=invalid-name
"""
Useful functions for ellipsometry.

Scott Prahl
Apr 2020
"""

import numpy as np
import pypolar.fresnel

__all__ = ('rho_from_m',
           'rho_from_tanpsi_Delta',
           'm_from_rho',
           'm_from_tanpsi_and_Delta',
           'rho_from_null_data',
           'rho_from_rotating_analyzer_data',
           'null_angles'
           )


def rho_from_m(m, theta_i):
    """
    Calculate the complex ratio of reflection amplitudes.

    This assumes that the material is flat and isotropic (e.g., no
    surface film).  It also assumes that the parallel (or perpendicular)
    field remains entirely parallel (or perpendicular) and is fully
    characterized by Fresnel reflection.

    Args:
        m :     complex index of refraction   [-]
        theta_i : incidence angle from normal [radians]
    Returns:
        complex ellipsometer parameter rho    [-]
    """
    rp = pypolar.fresnel.r_par(m, theta_i)
    rs = pypolar.fresnel.r_per(m, theta_i)
    return rp/rs


def rho_from_tanpsi_Delta(tanpsi, Delta):
    """
    Calculate the index of refraction for an isotropic sample.

    Formula from McCrackin "Measurement of the thickness and refractive
    index of very thin films and the optical properties of surfaces by
    ellipsometry", Journal of Research of the National Bureau of Standards,
    (1963).

    Args:
        tanpsi : tan(psi) or |r_p/r_s|                     [-]
        Delta :  phase change caused by reflection         [radians]
    Returns:
        complex ellipsometer parameter rho                 [-]
    """
    return tanpsi*np.exp(1j*Delta)


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
    rho = rho_from_tanpsi_Delta(tanpsi, Delta)
    return m_from_rho(rho, theta_i)


def rho_from_null_data(P1, P2, A1, A2):
    """
    Recover rho from Null ellipsometer measurements.
    """

    psi = np.pi/4 - (A1-A2)/2
    Delta = np.pi - (P1+P2)
    rho = rho_from_tanpsi_Delta(np.tan(psi), Delta)
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
    rho = tanpsi * np.exp(1j*Delta)

    return rho


def null_angles(m, theta_i):
    """
    Expected ellipsometer angles for all zones.

    Args:
        m :     complex index of refraction   [-]
        theta_i : incidence angle from normal [radians]
    Returns:
        dictionary with null angles for each zone   [-]
    """
    rho = rho_from_m(m, theta_i)
    tanpsi = np.abs(rho)
    psi = np.arctan(tanpsi)
    Delta = np.angle(rho)
    p = Delta/2-np.pi/4
    a = psi
    pi = np.pi
    PA = {}
    PA[1] = np.array([(p, a), (p+pi, a), (p, a+pi), (p+np.pi, np.pi)])
    PA[2] = np.array([(pi/2-p, a), (3*pi/2-p, a), (pi/2-p, a+pi), (3*pi/2-p, a+np.pi)])
    PA[3] = np.array([(p+pi/2, pi-a), (p+3*pi/2, pi-a), (p+pi/2, 2*pi-a), (p+3*pi/2, 2*pi-a)])
    PA[4] = np.array([(pi-p, pi-a), (2*pi-p, pi-a), (pi-p, 2*pi-a), (2*pi-p, 2*pi-a)])
    
    # make all measurements between 0 and 2pi
    PA[1] = np.remainder(PA[1] +2*np.pi,2*np.pi)
    PA[2] = np.remainder(PA[2] +2*np.pi,2*np.pi)
    PA[3] = np.remainder(PA[3] +2*np.pi,2*np.pi)
    PA[4] = np.remainder(PA[4] +2*np.pi,2*np.pi)
    
    return PA
