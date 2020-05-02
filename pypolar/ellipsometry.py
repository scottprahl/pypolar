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
           'rho_from_zone_2_null_angles',
           'rho_from_zone_4_null_angles',
           'rho_from_rotating_analyzer_data',
           'null_angles',
           'null_angles_report',
           'rotating_analyzer_signal',
           'rotating_analyzer_signal_from_rho',
           'rotating_analyzer_signal_from_m',
           'find_fourier',
           'rho_from_rotating_analyzer_data',
           'm_from_rotating_analyzer_data',
           )


def rho_from_m(m, theta_i):
    """
    Calculate the complex ratio of reflection amplitudes.

    This assumes that the material is flat and isotropic (e.g., no
    surface film).  It also assumes that the parallel (or perpendicular)
    field remains entirely parallel (or perpendicular) and is fully
    characterized by Fresnel reflection.

    Args:
        m:       complex index of refraction   [-]
        theta_i: incidence angle from normal [radians]
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

    # choose other branch
    if np.angle(rho) < 0:
        e_index = np.conjugate(e_index)
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


def rho_from_zone_2_null_angles(P2, A2):
    """
    Recover rho from Null ellipsometer measurements in zone 2.

    Args:
        P2 : polarizer angle for null reading     [radians]
        A2 : analyzer angle for null reading     [radians]
    Returns:
        complex ellipsometer parameter rho                 [-]
    """
    if (A2 < 0 or A2 > np.pi/2):
        print("Analyzer is not zone 2 (0 < %.2f < pi/2)" % P2)
        return 0
    if (P2 < -np.pi/4 or P2 > 3*np.pi/4):
        print("Polarizer is not zone 2 (-pi/4 < %.2f < 3pi/4)" % A2)
        return 0
    psi = A2
    Delta = 3*np.pi/2 - 2*P2
    rho = rho_from_tanpsi_Delta(np.tan(psi), Delta)
    return rho


def rho_from_zone_4_null_angles(P4, A4):
    """
    Recover rho from Null ellipsometer measurements in zone 4.

    Args:
        P4 : polarizer angle for null reading in zone 4    [radians]
        A4 : analyzer angle for null reading in zone 4   [radians]
    Returns:
        complex ellipsometer parameter rho                 [-]
    """
    if (A4 < -np.pi/2 or A4 > 0):
        print("Analyzer is not zone 4 (-pi/2 < %.2f < 0)" % P4)
        return 0
    if (P4 < -3*np.pi/4 or P4 > np.pi/4):
        print("Polarizer is not zone 4 (-3pi/4 < %.2f < pi/4)" % A4)
        return 0
    psi = -A4
    Delta = np.pi/2 - 2*P4
    rho = rho_from_tanpsi_Delta(np.tan(psi), Delta)
    return rho


def null_angles(m, theta_i):
    """
    Expected ellipsometer angles for all zones.

    The various null angles fall into four sets called zones, two with the fast-axis
    of the quarter wave plate set 45° (2 & 4) and two with the fast-axis of the
    quarter wave plate set to -45° (1 & 3).  In each zone there are four combinations
    of polarizer and analyzer angles that have a null reading (because rotation of a
    linear polarizer by 180° should give the same result.

    Table 1 from McCrackin "Measurement of the thickness and refractive
    index of very thin films and the optical properties of surfaces by
    ellipsometry", Journal of Research of the National Bureau of Standards,
    (1963).

    All angles returned fall between 0 and 2pi.

    Args:
        m :     complex index of refraction   [-]
        theta_i : incidence angle from normal [radians]
    Returns:
        dictionary with null angles [(P1, A1), (P2, A2), (P3, A3), (P4, A4)] for each zone
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
    PA[1] = np.remainder(PA[1] +2*np.pi, 2*np.pi)
    PA[2] = np.remainder(PA[2] +2*np.pi, 2*np.pi)
    PA[3] = np.remainder(PA[3] +2*np.pi, 2*np.pi)
    PA[4] = np.remainder(PA[4] +2*np.pi, 2*np.pi)

    return PA


def null_angles_report(m, theta_i):
    """
    Create a report showing null angles for sample.

    Args:
        m :     complex index of refraction   [-]
        theta_i : incidence angle from normal [radians]
    Returns:
        string containing a report listing null angles for each zone.
    """
    pa = null_angles(m, theta_i)

    rho = rho_from_m(m, theta_i)
    tanpsi = np.abs(rho)
    psi = np.arctan(tanpsi)
    Delta = np.angle(rho)
    p = Delta/2-np.pi/4
    a = psi

    s = "m       = %.4f%+.4fj\n" % (m.real, m.imag)
    s += "theta_i = %7.1f°\n" % np.degrees(theta_i)
    s += '\n'

    s += "zone  theta_p   theta_a\n"
    for zone in [1, 3, 2, 4]:
        for pair in pa[zone]:
            thetap, thetaa = np.degrees(pair)
            s += "  %d  %7.1f°  %7.1f°\n" % (zone, thetap, thetaa)
        s += '\n'

    s += "p       = %7.1f°\n" % np.degrees(p)
    s += "a       = %7.1f°\n" % np.degrees(a)
    s += '\n'
    s += "psi     = %7.1f°\n" % np.degrees(psi)
    s += "Delta   = %7.1f°\n" % np.degrees(Delta)
    s += '\n'

    return s


def rotating_analyzer_signal(phi, IDC, IS, IC, error=0):
    """
    Create theoretical rotating ellipsometer signal.

    In theory the rotating analyzer ellipsometer generates a
    sinusoidal signal with an offset.  This function does that
    and allows the optional addition of normally distributed
    noise.

    Args:
        phi    array of analyzer angles                [radians]
        IDC:   DC amplitude of signal                     [-]
        IS:    sin(2*phi) amplitude coefficient           [-]
        IC:    cos(2*phi) amplitude coefficient           [-]
        error: std dev of normal error distribution       [-]
    Returns:
        Array of ellipsometer readings for each angle phi [-]
    """
    base = IDC+IS*np.sin(2*phi)+IC*np.cos(2*phi)
    noise = np.random.normal(0, error, len(phi))
    return base+noise


def rotating_analyzer_signal_from_rho(phi, rho, theta_p, error=0, QWP=False):
    """
    Create normalized rotating ellipsometer signal for sample.

    The expected reading for each phi in a rotating analyzer ellipsometer
    can be generated using a
         rho = tan(psi)exp(j*Delta)
    This function does that and allows the optional addition
    of normally distributed noise.

    Note that the returned array is normalized between 0 and 1,
    therefore the error should be scaled accordingly.

    Args:
        phi:     array of analyzer angles                 [radians]
        rho:     ellipsometer parameter for surface       [complex]
        theta_p: angle of polarizer                       [radians]
        error:   std dev of normal error distribution     [-]
        QWP:     True if QWP is present
    Returns:
        Array of ellipsometer readings for each angle phi [-]
    """
    tanpsi = np.abs(rho)
    Delta = np.angle(rho)
    if QWP:
        Delta -= np.pi/2
    base = (tanpsi**2-np.tan(theta_p)**2) * np.cos(2*phi)
    base += 2*tanpsi*np.cos(Delta)*np.tan(theta_p)*np.sin(2*phi)
    base /= tanpsi**2+np.tan(theta_p)**2
    base += 1
    noise = np.random.normal(0, error, len(phi))
    return base+noise


def rotating_analyzer_signal_from_m(phi, m, theta_i, theta_p, error=0):
    """
    Create rotating ellipsometer signal for sample with known index.

    Args:
        phi:     array of analyzer angles                 [radians]
        m:       complex index of refraction of sample    [-]
        theta_i: angle of incidence (from normal)         [radians]
        theta_p: angle of incident polarized light        [radians]
        error:   std dev of normal error distribution     [-]

    Returns:
        Array of ellipsometer readings for each angle phi [-]

    """
    sig = pypolar.fresnel.r_par(m, theta_i)*np.cos(theta_p)*np.cos(phi)
    sig += pypolar.fresnel.r_per(m, theta_i)*np.sin(theta_p)*np.sin(phi)
    base = np.cos(theta_p)**2*abs(sig)**2
    noise = np.random.normal(0, error, len(phi))
    return base + noise


def find_fourier(phi, signal):
    """
    Calculate first few Fourier series coefficients.

    Fit the signal to the function
        I_0 * ( 1 + alpha*cos(2*phi) + beta*sin(2*phi) )
    args:
        phi:    array of analyzer angles
        signal: array of ellipsometer intensities
    returns:
        I_0, alpha, beta
    """
    I_0 = np.average(signal)
    I_S = 2*np.average(signal*np.sin(2*phi))
    I_C = 2*np.average(signal*np.cos(2*phi))
    alpha = I_C / I_0
    beta = I_S / I_0
    return I_0, alpha, beta


def rho_from_rotating_analyzer_data(phi, signal, theta_p, QWP=False):
    """
    Recover rho from rotating analyzer data.

    This is done by fitting the signal to
             I_0 * (1 + alpha*cos(2*phi) + beta*sin(2*phi))
    Then alpha and beta are used to find tan(psi) and Delta

    Args:
        phi:     array of analyzer angles               [radians]
        signal:  array of ellipsometer intensities      [AU]
        theta_p: incident polarization azimuthal angle  [radians]
        QWP:     True if QWP is present
    Returns:
        rho = tan(psi)*exp(1j*Delta)                    [-]
    """
    _, alpha, beta = find_fourier(phi, signal)

    tanP = np.tan(theta_p)
    arg = beta / np.sqrt(abs(1 - alpha**2)) * np.sign(tanP)
    if arg > 1:
        Delta = 0
    elif arg < -1:
        Delta = np.pi
    else:
        Delta = np.arccos(arg)

    if QWP:
        Delta += np.pi/2

    tanpsi = np.sqrt(abs(1 + alpha) / abs(1 - alpha)) * np.abs(tanP)
    rho = tanpsi * np.exp(1j*Delta)

    return rho

def m_from_rotating_analyzer_data(phi, signal, theta_i, theta_p, QWP=False):
    """
    Recover m from rotating analyzer data.

    Args:
        phi:     array of analyzer angles               [radians]
        signal:  array of ellipsometer intensities      [AU]
        theta_i: incidence angle from normal            [radians]
        theta_p: incident polarization azimuthal angle  [radians]
        QWP:     True if QWP is present
    Returns:
        complex index of refraction                     [-]
    """
    rho = rho_from_rotating_analyzer_data(phi, signal, theta_p, QWP)
    m = m_from_rho(rho, theta_i)
    return m
