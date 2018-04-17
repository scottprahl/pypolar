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
           't_par',
           't_per',
           'R_par',
           'R_per',
           'T_par',
           'T_per',
           'R_unpolarized',
           'ellipsometry_rho',
           'ellipsometry_index',
           'ellipsometry_parameters']


def r_par(m, theta):
    """
    Calculates the reflected amplitude for parallel polarized light

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        reflected field amplitude             [-]
    """
    c = m * m * np.cos(theta)
    s = np.sin(theta)
    d = np.sqrt(m * m - s * s, dtype=np.complex)
    if m.imag == 0 :
        d = np.conjugate(d)    
    rp = (c - d) / (c + d)
    return np.real_if_close(rp)


def r_per(m, theta):
    """
    Calculates the reflected amplitude for perpendicular polarized light

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        reflected field amplitude             [-]
    """
    c = np.cos(theta)
    s = np.sin(theta)
    d = np.sqrt(m * m - s * s, dtype=np.complex)
    if m.imag == 0 :
        d = np.conjugate(d)    
    rs = (c - d) / (c + d)
    return np.real_if_close(rs)


def t_par(m, theta):
    """
    Calculates the transmitted amplitude for parallel polarized light

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        transmitted field amplitude           [-]
    """
    c = np.cos(theta)
    s = np.sin(theta)
    d = np.sqrt(m * m - s * s, dtype=np.complex)
    if m.imag == 0 :
        d = np.conjugate(d)    
    tp = 2 * c * m/ (m * m * c + d)
    return np.real_if_close(tp)


def t_per(m, theta):
    """
    Calculates the transmitted amplitude for perpendicular polarized light

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        transmitted field amplitude           [-]
    """
    c = np.cos(theta)
    s = np.sin(theta)
    d = np.sqrt(m * m - s * s, dtype=np.complex)
    if m.imag == 0 :
        d = np.conjugate(d)    
    ts = 2 * c / (c + d)
    return np.real_if_close(ts)


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


def T_par(m, theta):
    """
    Calculates the transmitted irradiance for parallel polarized light

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        transmitted irradiance                [-]
    """
    return abs(t_par(m, theta))**2


def T_per(m, theta):
    """
    Calculates the transmitted irradiance for perpendicular polarized light

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        transmitted irradiance                [-]
    """
    return abs(t_per(m, theta))**2


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


def ellipsometry_rho(m, theta):
    """
    Calculate the ellipsometer parameter rho
    
    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        ellipsometer parameter rho            [-]
    """
    return r_par(m, theta) / r_per(m, theta)


def ellipsometry_index(rho, theta):
    """
    Calculate the index of refraction for an isotropic sample
    
    Args:
        rho :  r_par/r_per                    [-]
        theta : angle from normal to surface  [radians]
    Returns:
        complex index of refraction           [-]
    """
    return np.tan(theta)*np.sqrt(1-4*rho*np.sin(theta)**2/(1+rho)**2)


def ellipsometry_parameters(phi, signal, P):
    """
    Recover ellipsometer parameters $\Delta$ and $\tan\psi$ by fitting to
    
             I_DC + I_S*sin(2*phi)+I_C*cos(2*phi)
             
    Args:
        phi    - array of analyzer angles
        signal - array of ellipsometer intensities
        P      - incident polarization azimuthal angle
    """

    I_DC = np.average(signal)
    I_S = 2*np.average(signal*np.sin(2*phi))
    I_C = 2*np.average(signal*np.cos(2*phi))

    tanP = np.tan(P)
    arg = I_S/np.sqrt(abs(I_DC**2 - I_C**2))*np.sign(tanP)
    if arg>1 :
        Delta = 0
    elif arg<-1 :
        Delta = np.pi
    else :
        Delta = np.arccos(arg)
        
    tanpsi = np.sqrt(abs(I_DC+I_C)/abs(I_DC-I_C))*np.abs(tanP)

    return Delta,tanpsi

