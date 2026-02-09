"""
Useful routines for planar-interface Fresnel calculations.

The functions in this module assume two semi-infinite media with a planar
interface. Light is incident from a medium with real refractive index `n_i`
(default 1), onto a medium with complex refractive index `m`. Absorbing media
follow the `pypolar` convention `m = n - 1j * n * kappa`.

Angles are measured from the surface normal. Set `deg=True` when passing
angles in degrees.

Incidence-angle helpers::

    * brewster(index_of_refraction, n_i=1, deg=False)
    * critical(index_of_refraction, n_i=1, deg=False)

Field-amplitude Fresnel coefficients::

    * r_par_amplitude(index_of_refraction, angle, n_i=1, deg=False)
    * r_per_amplitude(index_of_refraction, angle, n_i=1, deg=False)
    * t_par_amplitude(index_of_refraction, angle, n_i=1, deg=False)
    * t_per_amplitude(index_of_refraction, angle, n_i=1, deg=False)

Power (irradiance) Fresnel coefficients::

    * R_par(index_of_refraction, angle, n_i=1, deg=False)
    * R_per(index_of_refraction, angle, n_i=1, deg=False)
    * T_par(index_of_refraction, angle, n_i=1, deg=False)
    * T_per(index_of_refraction, angle, n_i=1, deg=False)
    * R_unpolarized(index_of_refraction, angle, n_i=1, deg=False)
    * T_unpolarized(index_of_refraction, angle, n_i=1, deg=False)

Ellipsometry helpers::

    * ellipsometry_rho(index_of_refraction, angle, n_i=1, deg=False)
    * ellipsometry_index(rho, angle, n_i=1, deg=False)
"""

import numpy as np

__all__ = (
    "brewster",
    "critical",
    "r_par_amplitude",
    "r_per_amplitude",
    "t_par_amplitude",
    "t_per_amplitude",
    "R_par",
    "R_per",
    "T_par",
    "T_per",
    "R_unpolarized",
    "T_unpolarized",
    "ellipsometry_rho",
    "ellipsometry_index",
)


def _sanitize_refractive_index(m):
    """
    Apply the pypolar absorption-sign convention to refractive-index input.

    Positive imaginary parts are conjugated so absorbing media are represented
    with non-positive imaginary parts.
    """
    m_arr = np.asarray(m, dtype=complex)
    if np.any(~np.isfinite(m_arr.real)) or np.any(~np.isfinite(m_arr.imag)):
        raise ValueError("Refractive index must contain finite values")

    m_arr = np.where(np.imag(m_arr) > 0, np.conjugate(m_arr), m_arr)
    if np.isscalar(m):
        return m_arr.item()
    return m_arr


def _validate_incidence_angle(theta_i, deg=False):
    """Validate incidence-angle range and return a radian-valued array/scalar."""
    theta = np.asarray(theta_i)
    if np.iscomplexobj(theta):
        if np.any(np.imag(theta) != 0):
            raise ValueError("Incidence angle must be real-valued")
        theta = np.real(theta)

    theta = np.asarray(theta, dtype=float)
    if np.any(~np.isfinite(theta)):
        raise ValueError("Incidence angle must contain finite values")

    max_angle = 90.0 if deg else np.pi / 2
    units = "degrees" if deg else "radians"
    eps = 1e-12
    if np.any(theta < -eps) or np.any(theta > max_angle + eps):
        raise ValueError(f"Incidence angle must be between 0 and {max_angle:g} {units}")

    theta = np.radians(theta) if deg else theta
    if np.isscalar(theta_i):
        return theta.item()
    return theta


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
        return np.degrees(np.arcsin(m / n_i))
    return np.arcsin(m / n_i)


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
        cos(theta_i), m_rel*cos(theta_t), and m_rel      [-]
    """
    theta = _validate_incidence_angle(theta_i, deg=deg)
    m_rel = _sanitize_refractive_index(m) / n_i
    m2 = m_rel**2
    c = np.cos(theta)
    s = np.sin(theta)
    d = np.sqrt(m2 - s * s, dtype=complex)  # = m_rel*cos(theta_t)
    if np.isscalar(m_rel):
        if np.imag(m_rel) == 0:  # choose right branch for dielectrics
            d = np.conjugate(d)
    else:
        d = np.where(np.imag(m_rel) == 0, np.conjugate(d), d)
    return c, d, m_rel


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
    c, d, m_rel = _cosines(m, theta_i, n_i, deg)
    m2 = m_rel**2
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
    c, d, _ = _cosines(m, theta_i, n_i, deg)
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
    c, d, m_rel = _cosines(m, theta_i, n_i, deg)
    m2 = m_rel**2
    tp = 2 * c * m_rel / (m2 * c + d)
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
    c, d, _ = _cosines(m, theta_i, n_i, deg)
    ts = 2 * c / (c + d)
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
    return np.abs(r_par_amplitude(m, theta_i, n_i, deg)) ** 2


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
    return np.abs(r_per_amplitude(m, theta_i, n_i, deg)) ** 2


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
    c, d, m_rel = _cosines(m, theta_i, n_i, deg)
    tp = 2 * c * m_rel / (m_rel**2 * c + d)
    return np.abs(d / c * np.abs(tp) ** 2)


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
    c, d, _ = _cosines(m, theta_i, n_i, deg)
    ts = 2 * c / (c + d)
    return np.abs(d / c * np.abs(ts) ** 2)


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


def ellipsometry_rho(m, theta_i, n_i=1, deg=False):
    """
    Calculate the ellipsometer parameter rho.

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]

    Returns:
        ellipsometer parameter rho                        [-]
    """
    return r_par_amplitude(m, theta_i, n_i=n_i, deg=deg) / r_per_amplitude(m, theta_i, n_i=n_i, deg=deg)


def ellipsometry_index(rho, theta_i, n_i=1, deg=False):
    """
    Calculate the index of refraction for an isotropic sample.

    Args:
        rho:     r_par_amplitude/r_per_amplitude          [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]

    Returns:
        complex index of refraction                       [-]
    """
    theta = _validate_incidence_angle(theta_i, deg=deg)
    e_index = np.sqrt(1 - 4 * rho * np.sin(theta) ** 2 / (1 + rho) ** 2)
    return np.real_if_close(n_i * np.tan(theta) * e_index)
