# pylint: disable=invalid-name

"""
Useful functions for calculating light interaction at planar boundaries.

The underlying assumptions are that there a two semi-infinite media with a
planar interface.  For convenience, assume the light is incident from the top.
The upper medium is characterized by a purely real index of refraction `n_i` which
has a default value of 1.  The lower medium is characterized by a complex index
of refraction `m = n - n * kappa * 1j`.  Note that `pypolar` assumes the sign
of the imaginary part of the index of refraction is negative.

The Fresnel equations assume that the electric field has been decomposed into
fields relative to the plane of incidence (a plane defined by the incoming
light direction and the normal to the surface).

The incidence angle is measured from the normal to the surface and is measured
in radians.

To Do::
    * Make sure routines work for arrays of m or of theta_i
    * fail for positive imaginary refractive indices
    * fail for out-of-range angles to catch degrees/radians error

Scott Prahl
Apr 2021
"""

import numpy as np

__all__ = ('brewster',
           'critical',
           'r_par_amplitude',
           'r_per_amplitude',
           't_par_amplitude',
           't_per_amplitude',
           'R_par',
           'R_per',
           'T_par',
           'T_per',
           'R_unpolarized',
           'T_unpolarized'
           )

def brewster(m, n_i=1, deg=False):
    """
    Brewster's angle for an interface.

    Args:
        m:       complex index of refraction of medium    [-]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]
    Returns:
        Brewster's angle from normal to surface           [radians/degrees]
    """
    if deg:
        return np.degrees(np.arctan2(m, n_i))
    return np.arctan2(m, n_i)


def critical(m, n_i=1, deg=False):
    """
    Critical angle for total internal reflection at interface.

    Args:
        m:       complex index of refraction of medium    [-]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]
    Returns:
        critical angle from normal to surface             [radians/degrees]
    """
    if deg:
        return np.degrees(np.arcsin(m/n_i))
    return np.arcsin(m/n_i)

def _cosines(m, theta_i, n_i, deg=False):
    """
    Intermediate cosines needed for Fresnel equations.
    
    This is split out because so that special casing for
    degrees is not needed everywhere and so that algorithms
    work properly when m is an array as well.
    
    n_i * sin(theta_i) = m * sin(theta_t)
    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]
    Returns:
        cos(theta_i) and cos(theta_t)                     [-]
    """
    if deg:
        theta = np.radians(theta_i)
    else:
        theta = theta_i
    m2 = (m/n_i)**2
    c = np.cos(theta)
    s = np.sin(theta)
    d = np.sqrt(m2 - s * s, dtype=np.complex) # = m*cos(theta_t)
    if np.isscalar(m):
        if m.imag == 0:  # choose right branch for dielectrics
            d = np.conjugate(d)
    else:
        d = np.where(m.imag == 0, np.conjugate(d), d)
    return c, d

def r_par_amplitude(m, theta_i, n_i=1, deg=False):
    """
    Reflected fraction of parallel-polarized field at an interface.

    This is the fraction of the incident electric field reflected at the
    interface between two semi-infinite media. The incident field is assumed
    to be polarized parallel (p) to the plane of incidence (transverse magnetic
    or TM field).

    The index of refraction for medium of the incoming field defaults to 1, but
    can be set any real value. The medium of the outgoing field is characterized
    by an index of refraction that may be complex.

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]
    Returns:
        reflected fraction of parallel field              [-]
    """
    c, d = _cosines(m, theta_i, n_i, deg)
    m2 = (m/n_i)**2
    rp = (m2 * c - d) / (m2 * c + d)
    return np.real_if_close(rp)


def r_per_amplitude(m, theta_i, n_i=1, deg=False):
    """
    Reflected fraction of perpendicular-polarized field at an interface.

    This is the fraction of the incident electric field reflected at the
    interface between two semi-infinite media. The incident field is assumed
    to be polarized perpendicular (s, or senkrecht) to the plane of incidence
    (transverse electric or TE field).

    The index of refraction for medium of the incoming field defaults to 1, but
    can be set any real value. The medium of the outgoing field is characterized
    by an index of refraction that may be complex.

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]
    Returns:
        reflected fraction of perpendicular field         [-]
    """
    c, d = _cosines(m, theta_i, n_i, deg)
    rs = (c - d) / (c + d)
    return np.real_if_close(rs)


def t_par_amplitude(m, theta_i, n_i=1, deg=False):
    """
    Find the transmitted fraction of parallel-polarized field through an interface.

    This is the fraction of the incident electric field transmitted through the
    interface between two semi-infinite media. The incident field is assumed
    to be polarized parallel (p) to the plane of incidence (transverse magnetic
    or TM field).

    The index of refraction for medium of the incoming field defaults to 1, but
    can be set any real value. The medium of the outgoing field is characterized
    by an index of refraction that may be complex.

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]
    Returns:
        transmitted fraction of parallel field            [-]
    """
    c, d = _cosines(m, theta_i, n_i, deg)
    m2 = (m/n_i)**2
    tp = 2 * c * (m/n_i) / (m2 * c + d)
    return np.real_if_close(tp)


def t_per_amplitude(m, theta_i, n_i=1, deg=False):
    """
    Return the transmitted fraction of perpendicular-polarized field through an interface.

    This is the fraction of the incident electric field transmitted through the
    interface between two semi-infinite media. The incident field is assumed
    to be polarized perpendicular (s, or senkrecht) to the plane of incidence
    (transverse electric or TE field).

    The index of refraction for medium of the incoming field defaults to 1, but
    can be set any real value. The medium of the outgoing field is characterized
    by an index of refraction that may be complex.

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]
    Returns:
        transmitted fraction of perpendicular field       [-]
    """
    c, d = _cosines(m, theta_i, n_i, deg)
    ts = 2 * d / (m/n_i)/ (c + d)
    return np.real_if_close(ts)


def R_par(m, theta_i, n_i=1, deg=False):
    """
    Reflected fraction of parallel-polarized optical power by an interface.

    The reflected fraction of incident power (or flux) assuming that
    the electric field of the incident light is polarized parallel (p) to the
    plane of incidence (transverse magnetic or TM electric field).

    The index of refraction for medium of the incoming field defaults to 1, but
    can be set any real value. The medium of the outgoing field is characterized
    by an index of refraction that may be complex.

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]
    Returns:
        reflected fraction of parallel-polarized irradiance [-]
    """
    return np.abs(r_par_amplitude(m, theta_i, n_i, deg))**2


def R_per(m, theta_i, n_i=1, deg=False):
    """
    Return the fraction of perpendicular-polarized optical power reflectedby an interface.

    The fraction of the incident power (or flux) reflected at the
    interface between two semi-infinite media. The incident light is assumed
    to be polarized perpendicular (s, or senkrecht) to the plane of incidence
    (transverse electric or TE field).

    The index of refraction for medium of the incoming field defaults to 1, but
    can be set any real value. The medium of the outgoing field is characterized
    by an index of refraction that may be complex.

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]
    Returns:
        reflected fraction of perpendicular-polarized irradiance [-]
    """
    return np.abs(r_per_amplitude(m, theta_i, n_i, deg))**2


def T_par(m, theta_i, n_i=1, deg=False):
    """
    Return the transmitted fraction of parallel-polarized optical power through an interface.

    The transmitted fraction of incident power (or flux) assuming that
    the electric field of the incident light is polarized parallel (p) to the
    plane of incidence (transverse magnetic or TM electric field).

    The index of refraction for medium of the incoming field defaults to 1, but
    can be set any real value. The medium of the outgoing field is characterized
    by an index of refraction that may be complex.

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]
    Returns:
        transmitted fraction of parallel-polarized irradiance [-]
    """
    c, d = _cosines(m, theta_i, n_i, deg)
    tp = 2 * c * (m/n_i) / ((m/n_i)**2 * c + d)
    return np.abs(d / c * np.abs(tp)**2)


def T_per(m, theta_i, n_i=1, deg=False):
    """
    Return the transmitted fraction of perpendicular-polarized optical power through an interface.

    The transmitted fraction of the incident power (or flux) through the
    interface between two semi-infinite media. The incident light is assumed
    to be polarized perpendicular (s, or senkrecht) to the plane of incidence
    (transverse electric or TE field).

    The index of refraction for medium of the incoming field defaults to 1, but
    can be set any real value. The medium of the outgoing field is characterized
    by an index of refraction that may be complex.

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]
    Returns:
        transmitted fraction of perpendicular-polarized irradiance [-]
    """
    c, d = _cosines(m, theta_i, n_i, deg)
    ts = 2 * c / (c + d)
    return np.abs(d / c * abs(ts)**2)


def R_unpolarized(m, theta_i, n_i=1, deg=False):
    """
    Fraction of unpolarized light that is reflected.

    Calculate reflection fraction of incident power (or flux) assuming that
    the incident light is unpolarized

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]
    Returns:
        fraction of unpolarized irradiance reflected      [-]
    """
    return (R_par(m, theta_i, n_i, deg) + R_per(m, theta_i, n_i, deg)) / 2


def T_unpolarized(m, theta_i, n_i=1, deg=False):
    """
    Fraction of unpolarized light that is transmitted.

    Calculate transmitted fraction of incident power (or flux) assuming that
    the incident light is unpolarized

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]
    Returns:
        fraction of unpolarized irradiance transmitted    [-]
    """
    return (T_par(m, theta_i, n_i, deg) + T_per(m, theta_i, n_i, deg)) / 2
