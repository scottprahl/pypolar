# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
Useful functions for ellipsometry.

Scott Prahl
Apr 2021
"""

import numpy as np
import pypolar.fresnel

__all__ = ('rho_from_m',
           'rho_from_tanpsi_Delta',
           'tanpsi_Delta_from_rho',
           'm_from_rho',
           'm_from_tanpsi_and_Delta',
           'rho_from_zone_2_null_angles',
           'rho_from_zone_4_null_angles',
           'rho_from_rotating_analyzer_data',
           'null_angles',
           'null_angles_report',
           'rotating_analyzer_signal',
           'RAE_from_rho',
           'rotating_analyzer_signal_from_rho',
           'rotating_analyzer_signal_from_m',
           'find_fourier',
           'rho_from_rotating_analyzer_data',
           'rho_from_PSA',
           'm_from_rotating_analyzer_data',
           )


def rho_from_m(m, theta_i, deg=False):
    """
    Calculate the complex ratio of reflection amplitudes.

    This assumes that the material is flat and isotropic (e.g., no
    surface film).  It also assumes that the parallel (or perpendicular)
    field remains entirely parallel (or perpendicular) and is fully
    characterized by Fresnel reflection.

    Args:
        m:       complex index of refraction   [-]
        theta_i: incidence angle from normal   [radians/degrees]
        deg:     theta_i is in degrees         [True/False]
    Returns:
        complex ellipsometer parameter rho     [-]
    """
    rp = pypolar.fresnel.r_par_amplitude(m, theta_i, deg=deg)
    rs = pypolar.fresnel.r_per_amplitude(m, theta_i, deg=deg)
    return rp/rs


def rho_from_tanpsi_Delta(tanpsi, Delta, deg=False):
    """
    Calculate the index of refraction for an isotropic sample.

    Formula from McCrackin "Measurement of the thickness and refractive
    index of very thin films and the optical properties of surfaces by
    ellipsometry", Journal of Research of the National Bureau of Standards,
    (1963).

    Args:
        tanpsi:  tan(psi) or abs(rpar/rperp)        [-]
        Delta:   phase change caused by reflection  [radians/degrees]
        deg:     Delta is in degrees                [True/False]
    Returns:
        complex ellipsometer parameter rho          [-]
    """
    if deg:
        d = np.radians(Delta)
    else:
        d = Delta
    return tanpsi*np.exp(1j*d)


def tanpsi_Delta_from_rho(rho, deg=False):
    """
    Extract ellipsometer parameters from rho.

    rho = r_par_amplitude/r_per_amplitude or

    rho = tan(psi)*exp(j*Delta)

    Formula from Fujiwara 2007 eqn 4.6 and correspond to the case
    when the complex refractive index is negative (m = n-k*1j)

    Args:
        rho:   complex reflectance ratio            [-]
        deg:   return Delta in degrees?             [True/False]
    Returns:
        tanpsi:  tan(psi) or abs(r_p/r_s)           [-]
        Delta:   phase change caused by reflection  [radians/degrees]
    """
    Delta = np.arctan2(rho.imag, rho.real)
    if rho.real < 0:
        if rho.imag > 0:
            Delta += np.pi
        else:
            Delta -= np.pi
    tanpsi = abs(rho)
    if deg:
        Delta = np.degrees(Delta)
    return tanpsi, Delta


def m_from_rho(rho, theta_i, deg=False):
    """
    Calculate the index of refraction for an isotropic sample.

    rho = r_par_amplitude/r_per_amplitude or

    rho = tan(psi)*exp(j*Delta)

    Formula from McCrackin "Measurement of the thickness and refractive
    index of very thin films and the optical properties of surfaces by
    ellipsometry", Journal of Research of the National Bureau of Standards,
    (1963).

    Args:
        rho:     complex reflectance ratio      [-]
        theta_i: incidence angle from normal    [radians/degrees]
        deg:     theta_i is in degrees          [True/False]
    Returns:
        complex index of refraction             [-]
    """
    if deg:
        theta = np.radians(theta_i)
    else:
        theta = theta_i
    e_index = np.sqrt(1 - 4 * rho * np.sin(theta)**2 / (1 + rho)**2)

    # choose proper branch
    if np.isscalar(rho):
        if np.angle(rho) < 0:
            e_index = np.conjugate(e_index)
    else:
        for i, r in enumerate(rho):
            if np.angle(r) < 0:
                e_index[i] = np.conjugate(e_index[i])

    return np.tan(theta) * e_index


def m_from_tanpsi_and_Delta(tanpsi, Delta, theta_i, deg=False):
    """
    Return the index of refraction for observed Delta, tanpsi, and theta_i.

    Args:
        tanpsi:  abs() of ratio of field amplitudes        [-]
        Delta:   phase change caused by reflection         [-]
        theta_i:  incidence angle from normal              [radians/degrees]
        deg:     theta_i and Delta are in degrees          [True/False]
    Returns:
        complex index of refraction                        [-]
    """
    rho = rho_from_tanpsi_Delta(tanpsi, Delta, deg=deg)
    return m_from_rho(rho, theta_i, deg=deg)


def rho_from_zone_2_null_angles(P, A, deg=False):
    """
    Recover rho from Null ellipsometer measurements in zone 2.

    Args:
        P:    polarizer angle for null reading  [radians/degrees]
        A:    analyzer angle for null reading   [radians/degrees]
        deg:  P and A are in degrees            [True/False]
    Returns:
        complex ellipsometer parameter rho      [-]
    """
    if deg:
        A2 = np.radians(A)
        P2 = np.radians(P)
    else:
        A2 = A
        P2 = P

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


def rho_from_zone_4_null_angles(P, A, deg=False):
    """
    Recover rho from Null ellipsometer measurements in zone 4.

    Args:
        P: polarizer angle for null reading in zone 4  [radians/degrees]
        A: analyzer angle for null reading in zone 4   [radians/degrees]
        deg:  P and A are in degrees                   [True/False]
    Returns:
        complex ellipsometer parameter rho             [-]
    """
    if deg:
        A4 = np.radians(A)
        P4 = np.radians(P)
    else:
        A4 = A
        P4 = P

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


def null_angles(m, theta_i, deg=False):
    """
    Generate expected ellipsometer angles for all zones.

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
        m:       complex index of refraction   [-]
        theta_i: incidence angle from normal [radians/degrees]
        deg:     theta_i in degrees, return null angles in degrees? [True/False]
    Returns:
        dictionary with null angles [(P1, A1), (P2, A2), (P3, A3), (P4, A4)] for each zone
    """
    rho = rho_from_m(m, theta_i, deg=deg)
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

    if deg:
        PA[1] = np.degrees(PA[1])
        PA[2] = np.degrees(PA[2])
        PA[3] = np.degrees(PA[3])
        PA[4] = np.degrees(PA[4])
    return PA


def null_angles_report(m, theta_i, deg=False):
    """
    Create a report showing null angles for sample.

    Args:
        m:       complex index of refraction [-]
        theta_i: incidence angle from normal [radians/degrees]
        deg:     theta_i is in degrees       [True/False]
    Returns:
        string containing a report listing null angles for each zone.
    """
    if deg:
        theta = theta_i
    else:
        theta = np.degrees(theta_i)

    pa = null_angles(m, theta_i)
    rho = rho_from_m(m, theta_i)

    tanpsi = np.abs(rho)
    psi = np.arctan(tanpsi)
    Delta = np.angle(rho)

    s = "m       = %.4f%+.4fj\n" % (m.real, m.imag)
    s += "theta_i = %7.1f°\n" % theta
    s += '\n'

    s += "zone  P   theta_a\n"
    for zone in [1, 3, 2, 4]:
        for pair in pa[zone]:
            thetap, thetaa = pair
            s += "  %d  %7.1f°  %7.1f°\n" % (zone, thetap, thetaa)
        s += '\n'

    s += "p       = %7.1f°\n" % np.degrees(Delta/2-np.pi/4)
    s += "a       = %7.1f°\n" % np.degrees(psi)
    s += '\n'
    s += "psi     = %7.1f°\n" % np.degrees(psi)
    s += "Delta   = %7.1f°\n" % np.degrees(Delta)
    s += '\n'

    return s


def rotating_analyzer_signal(phi, IDC, IS, IC, noise=0, deg=False):
    """
    Create theoretical rotating ellipsometer signal.

    In theory the rotating analyzer ellipsometer generates a
    sinusoidal signal with an offset.  This function does that
    and allows the optional addition of normally distributed
    noise.

    Args:
        phi    array of analyzer angles                   [radians/degrees]
        IDC:   DC amplitude of signal                     [-]
        IS:    sin(2*phi) amplitude coefficient           [-]
        IC:    cos(2*phi) amplitude coefficient           [-]
        noise: std dev of normal noise distribution       [-]
        deg:   phi is in degrees                          [True/False]
    Returns:
        Array of ellipsometer readings for each angle phi [-]
    """
    if deg:
        phi_radians = np.radians(phi)
    else:
        phi_radians = phi
    base = IDC+IS*np.sin(2*phi_radians)+IC*np.cos(2*phi_radians)
    noise = np.random.normal(0, noise, len(phi))
    return base+noise


def rotating_analyzer_signal_from_rho_old(phi, rho, P, QWP=False, average=1, noise=0, deg=False):
    """
    Create normalized rotating ellipsometer signal for sample.

    Generate the expected reading at each analyzer angle in an ellipsometer
    with a sample characterized by a material with an ellipsometer parameter
    rho = tan(psi)exp(j*Delta)

    This is a classic
    source::polarizer::QWP::sample::analyzer::detector
    arrangement.  The QWP is oriented at +45° if present.

    Note that the default returned array is normalized between 0 and 1.
    therefore the noise should be scaled accordingly.

    Args:
        phi:     array of analyzer angles from 0 to 2pi   [radians/degrees]
        rho:     ellipsometer parameter for surface       [complex]
        P:       angle of polarizer                       [radians/degrees]
        QWP:     True if QWP is present
        average: average value of signal over 2pi         [AU]
        noise:   std dev of normal noise distribution     [AU]
        deg:     phi and P are in degrees                 [True/False]
    Returns:
        Array of ellipsometer readings for each angle phi [-]
    """
    if deg:
        tanP = np.tan(np.radians(P))
        phi_radians = np.radians(phi)
    else:
        tanP = np.tan(P)
        phi_radians = phi
    tanpsi = np.abs(rho)
    Delta = np.angle(rho)
    if QWP:
        Delta -= np.pi/2
    denom = tanpsi**2+tanP**2
    alpha = (tanpsi**2-tanP**2)/denom
    beta = 2*tanpsi*np.cos(Delta)*tanP/denom
    base = 1 + alpha * np.cos(2*phi_radians) + beta * np.sin(2*phi_radians)
    noise = np.random.normal(0, noise, len(phi))
    return average*base+noise


def RAE_from_rho(phi, rho, P, average=1, noise=0, deg=False):
    """
    Create normalized rotating ellipsometer signal for sample.

    See eqn 4.19 and eqn 4.23 in Fujiwara 2007

    Generate the expected reading at each analyzer angle in an ellipsometer
    with a sample characterized by a material with an ellipsometer parameter
    rho = tan(psi)exp(j*Delta)

    This is a classic
    source::polarizer::QWP::sample::analyzer::detector
    arrangement.  The QWP is oriented at +45° if present.

    Note that the default returned array is normalized between 0 and 1.
    therefore the noise should be scaled accordingly.

    Args:
        phi:     array of analyzer angles from 0 to 2pi   [radians/degrees]
        rho:     ellipsometer parameter for surface       [complex]
        P:       angle of polarizer                       [radians/degrees]
        average: average value of signal over 2pi         [AU]
        noise:   std dev of normal noise distribution     [AU]
        deg:     phi and P are in degrees                 [True/False]
    Returns:
        Array of ellipsometer readings for each angle phi [-]
    """
    if deg:
        tanP = np.tan(np.radians(P))
        phi_radians = np.radians(phi)
    else:
        tanP = np.tan(P)
        phi_radians = phi

    tanpsi = np.abs(rho)
    Delta = np.angle(rho)
    denom = tanpsi**2 + tanP**2
    alpha = (tanpsi**2 - tanP**2)/denom
    beta = (2*tanpsi*np.cos(Delta)*tanP)/denom
    base = 1 + alpha * np.cos(2*phi_radians) + beta * np.sin(2*phi_radians)

    noise = np.random.normal(0, noise, len(phi_radians))
    return average*base+noise


def rotating_analyzer_signal_from_rho(phi, rho, P, QWP=False, average=1, noise=0, deg=False):
    """
    Create normalized rotating ellipsometer signal for sample.

    Generate the expected reading at each analyzer angle in an ellipsometer
    with a sample characterized by a material with an ellipsometer parameter
    rho = tan(psi)exp(j*Delta)

    This is a classic
    source::polarizer::QWP::sample::analyzer::detector
    arrangement.  The QWP is oriented at +45° if present.

    Note that the default returned array is normalized between 0 and 1.
    therefore the noise should be scaled accordingly.

    Args:
        phi:     array of analyzer angles from 0 to 2pi   [radians/degrees]
        rho:     ellipsometer parameter for surface       [complex]
        P:       angle of polarizer                       [radians/degrees]
        QWP:     True if QWP is present
        average: average value of signal over 2pi         [AU]
        noise:   std dev of normal noise distribution     [AU]
        deg:     phi and P are in degrees                 [True/False]
    Returns:
        Array of ellipsometer readings for each angle phi [-]
    """
    if deg:
        P_radians = np.radians(P)
        phi_radians = np.radians(phi)
    else:
        P_radians = P
        phi_radians = phi

    tanpsi = np.abs(rho)
    Delta = np.angle(rho)
    if QWP:
        psi = np.arctan(tanpsi)
        alpha = -np.cos(2*psi)
        beta = np.sin(2*psi)*np.cos(Delta + 2 * P_radians)
    else:
        tanP = np.tan(P)
        denom = tanpsi**2 + tanP**2
        alpha = (tanpsi**2 - tanP**2)/denom
        beta = (2*tanpsi*np.cos(Delta)*tanP)/denom
    base = 1 + alpha * np.cos(2*phi_radians) + beta * np.sin(2*phi_radians)
    noise = np.random.normal(0, noise, len(phi))
    return average*base+noise


def rotating_analyzer_signal_from_m(phi, m, theta_i, P, average=1, noise=0, deg=False):
    """
    Create rotating ellipsometer signal for sample with known index.

    Args:
        phi:     array of analyzer angles from 0 to 2pi   [radians/degrees]
        m:       complex index of refraction of sample    [-]
        theta_i: angle of incidence (from normal)         [radians/degrees]
        P:       angle of incident polarized light        [radians/degrees]
        average: average value of signal over 2pi         [AU]
        noise:   std dev of normal noise distribution     [-]
        deg:     phi, theta_i and P ar in degrees         [True/False]

    Returns:
        Array of ellipsometer readings for each angle phi [-]
    """
    if deg:
        P_radians = np.radians(P)
        phi_radians = np.radians(phi)
    else:
        P_radians = P
        phi_radians = phi

    sig = pypolar.fresnel.r_par_amplitude(m, theta_i, deg=deg)*np.cos(P_radians)*np.cos(phi_radians)
    sig += pypolar.fresnel.r_per_amplitude(m, theta_i, deg=deg)*np.sin(P_radians)*np.sin(phi_radians)
    base = np.cos(P_radians)**2*abs(sig)**2
    noise = np.random.normal(0, noise, len(phi))
    return average*base + noise


def find_fourier(phi, signal, deg=False):
    """
    Calculate first few Fourier series coefficients.

    Fit the signal to the function
        I_ave * ( 1 + alpha*cos(2*phi) + beta*sin(2*phi) )

    args:
        phi:    array of analyzer angles           [radians/degrees]
        signal: array of ellipsometer intensities  [AU]
        deg:    phi is in degrees                  [True/False]
    returns:
        I_ave, alpha, beta
    """
    if deg:
        phi_radians = np.radians(phi)
    else:
        phi_radians = phi
    I_ave = np.average(signal)
    I_S = 2*np.average(signal*np.sin(2*phi_radians))
    I_C = 2*np.average(signal*np.cos(2*phi_radians))
    alpha = I_C / I_ave
    beta = I_S / I_ave
    return I_ave, alpha, beta


def rho_from_rotating_analyzer_data_old(phi, signal, P, QWP=False, deg=False):
    """
    Recover rho from rotating analyzer data (old version).

    This is done by fitting the signal to
             I_ave * (1 + alpha*cos(2*phi) + beta*sin(2*phi))
    Then alpha and beta are used to find tan(psi) and Delta following
    e.g. eqn 4.24 in Fujiwara.

    Args:
        phi:     array of analyzer angles               [radians/degrees]
        signal:  array of ellipsometer intensities      [AU]
        P:       incident polarization azimuthal angle  [radians/degrees]
        QWP:     True if QWP is present
        deg:     phi and P are in degrees               [True/False]
    Returns:
        rho = tan(psi)*exp(1j*Delta)                    [-]
        fit: array of fitted data
    """
    if deg:
        tanP = np.tan(np.radians(P))
        phi_radians = np.radians(phi)
    else:
        tanP = np.tan(P)
        phi_radians = phi

    I_ave, alpha, beta = find_fourier(phi_radians, signal)
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

    fit = I_ave * (1 + alpha * np.cos(2*phi_radians) + beta * np.sin(2*phi_radians))
    return rho, fit


def rho_from_rotating_analyzer_data(phi, signal, P, QWP=False, deg=False):
    """
    Recover rho from rotating analyzer data.

    Based on equation 3.297 from Azzam
    (should be fixed to work with any P value)

    Args:
        phi:     array of analyzer angles               [radians/degrees]
        signal:  array of ellipsometer intensities      [AU]
        P:       incident polarization azimuthal angle  [radians/degrees]
        QWP:     True if QWP is present
        deg:     phi and P are in degrees               [True/False]
    Returns:
        rho = tan(psi)*exp(1j*Delta)                    [-]
        fit: array of fitted data
    """
    if deg:
        P_radians = np.radians(P)
        phi_radians = np.radians(phi)
    else:
        P_radians = P
        phi_radians = phi

    I_ave, alpha, beta = find_fourier(phi_radians, signal)

    if QWP:
        tanPC = np.tan(P_radians+np.pi/4)
        factor = (1+1j*tanPC)/(1-1j*tanPC)
    else:
        factor = np.tan(P_radians)

    delta = complex(1-alpha**2-beta**2, 0)
    rho = (1+alpha)/(beta-1j*np.sqrt(delta)) * factor
    rho *= np.exp(-1j*np.pi/2*QWP)

    fit = I_ave * (1 + alpha * np.cos(2*phi_radians) + beta * np.sin(2*phi_radians))

    if 0 <= P <= np.pi/2:
        return rho, fit

    return  np.conjugate(rho), fit


def rho_from_PSA(phi, signal, P, deg=False):
    """
    Recover rho from polarizer/sample/rotating analyzer system.

    Based on equation 4.24 in Fujiwara 2005.  Note that the PSA
    arrangement allows 0<=psi<=90° and 0<=Delta<=180°.

    In this system the measurement error increases when Delta is
    near zero or 180°.  Since this corresponds to linearly polarized
    light and would be exactly what would be measured for dielectric
    samples.

    Args:
        phi:     array of analyzer angles               [radians/degrees]
        signal:  array of ellipsometer intensities      [AU]
        P:       incident polarization azimuthal angle  [radians/degrees]
        deg:     phi and P are in degrees               [True/False]
    Returns:
        rho = tan(psi)*exp(1j*Delta)                    [-]
    """
    if deg:
        P_radians = np.radians(P)
        phi_radians = np.radians(phi)
    else:
        P_radians = P
        phi_radians = phi

    I_ave, alpha, beta = find_fourier(phi_radians, signal)
    tanpsi = np.sqrt((1+alpha)/(1-alpha)) * abs(np.tan(P_radians))
    Delta = np.arccos(beta/np.sqrt(1-alpha**2))
    rho = tanpsi * np.exp(1j*Delta)
    fit = I_ave * (1 + alpha * np.cos(2*phi_radians) + beta * np.sin(2*phi_radians))
    return rho, fit


def m_from_rotating_analyzer_data(phi, signal, theta_i, P, QWP=False, deg=False):
    """
    Recover m from rotating analyzer data.

    Args:
        phi:     array of analyzer angles               [radians/degrees]
        signal:  array of ellipsometer intensities      [AU]
        theta_i: incidence angle from normal            [radians/degrees]
        P:       incident polarization azimuthal angle  [radians/degrees]
        QWP:     True if QWP is present
        deg:     phi, theta_i, and P are in degrees     [True/False]
    Returns:
        complex index of refraction                     [-]
    """
    rho, fit = rho_from_rotating_analyzer_data(phi, signal, P, QWP, deg=deg)
    m = m_from_rho(rho, theta_i, deg=deg)
    return m, fit
