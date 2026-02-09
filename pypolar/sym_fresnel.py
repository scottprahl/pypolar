"""
Useful routines for symbolic Fresnel calculations at planar interfaces.

The functions in this module return symbolic SymPy expressions for Fresnel
field and power coefficients at a planar interface. Light is incident from a
medium with real refractive index `n_i` (default 1), onto a medium with
complex refractive index `m`.

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

import sympy

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


def brewster(m, n_i=1, deg=False):
    """
    Brewster's angle for an interface.

    Args:
        m:   complex index of refraction of medium    [-]
        n_i: real refractive index of incident medium [-]
        deg: theta_i is in degrees                    [True/False]

    Returns:
        Brewster's angle from normal to surface       [radians/degrees]
    """
    theta_b = sympy.atan2(m, n_i)
    if deg:
        return sympy.deg(theta_b)
    return theta_b


def critical(m, n_i=1, deg=False):
    """
    Critical angle for total internal reflection at interface.

    Args:
        m:   complex index of refraction of medium    [-]
        n_i: real refractive index of incident medium [-]
        deg: theta_i is in degrees                    [True/False]

    Returns:
        critical angle from normal to surface         [radians/degrees]
    """
    theta_c = sympy.asin(m / n_i)
    if deg:
        return sympy.deg(theta_c)
    return theta_c


def _cosines(m, theta_i, n_i=1, deg=False):
    """
    Intermediate cosines needed for Fresnel equations.

    n_i * sin(theta_i) = m * sin(theta_t)

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]

    Returns:
        cos(theta_i), m*cos(theta_t), and relative index m/n_i
    """
    if deg:
        theta = theta_i * sympy.pi / 180
    else:
        theta = theta_i
    m_rel = m / n_i
    c = sympy.cos(theta)
    s = sympy.sin(theta)
    d = sympy.sqrt(m_rel * m_rel - s * s)  # = m_rel*cos(theta_t)
    if sympy.im(m_rel) == 0:
        d = sympy.conjugate(d)
    return c, d, m_rel


def r_par_amplitude(m, theta_i, n_i=1, deg=False):
    """
    Calculate the reflected amplitude for parallel polarized light.

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: angle from normal to surface             [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]

    Returns:
        reflected fraction of parallel field    [-]
    """
    c, d, m_rel = _cosines(m, theta_i, n_i=n_i, deg=deg)
    m2 = m_rel * m_rel
    rp = (m2 * c - d) / (m2 * c + d)
    return rp


def r_per_amplitude(m, theta_i, n_i=1, deg=False):
    """
    Calculate the reflected amplitude for perpendicular polarized light.

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]

    Returns:
        reflected fraction of perpendicular field [-]
    """
    c, d, _ = _cosines(m, theta_i, n_i=n_i, deg=deg)
    rs = (c - d) / (c + d)
    return rs


def t_par_amplitude(m, theta_i, n_i=1, deg=False):
    """
    Calculate the transmitted amplitude for parallel polarized light.

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]

    Returns:
        transmitted fraction of parallel field [-]
    """
    c, d, m_rel = _cosines(m, theta_i, n_i=n_i, deg=deg)
    m2 = m_rel * m_rel
    tp = 2 * c * m_rel / (m2 * c + d)
    return tp


def t_per_amplitude(m, theta_i, n_i=1, deg=False):
    """
    Calculate the transmitted amplitude for perpendicular polarized light.

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]

    Returns:
        transmitted fraction of perpendicular field [-]
    """
    c, d, _ = _cosines(m, theta_i, n_i=n_i, deg=deg)
    ts = 2 * c / (c + d)
    return ts


def R_par(m, theta_i, n_i=1, deg=False):
    """
    Fraction of parallel-polarized light that is reflected (R_p).

    Calculate reflected fraction of incident power (or flux) assuming that
    the E-field of the incident light is parallel to the plane of incidence

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]

    Returns:
        reflected power                       [-]
    """
    return sympy.Abs(r_par_amplitude(m, theta_i, n_i=n_i, deg=deg)) ** 2


def R_per(m, theta_i, n_i=1, deg=False):
    """
    Fraction of perpendicular-polarized light that is reflected (R_s).

    Calculate reflected fraction of incident power (or flux) assuming that
    the E-field of the incident light is perpendicular to the plane of incidence

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]

    Returns:
        reflected irradiance                  [-]
    """
    return sympy.Abs(r_per_amplitude(m, theta_i, n_i=n_i, deg=deg)) ** 2


def T_par(m, theta_i, n_i=1, deg=False):
    """
    Fraction of parallel-polarized light that is transmitted (T_p).

    Calculate transmitted fraction of incident power (or flux) assuming that
    the E-field of the incident light is parallel to the plane of incidence

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]

    Returns:
        transmitted irradiance                [-]
    """
    c, d, m_rel = _cosines(m, theta_i, n_i=n_i, deg=deg)
    m2 = m_rel * m_rel
    tp = 2 * c * m_rel / (m2 * c + d)
    return sympy.Abs(d / c * sympy.Abs(tp) ** 2)


def T_per(m, theta_i, n_i=1, deg=False):
    """
    Fraction of perpendicular-polarized light that is transmitted (T_s).

    Calculate transmitted fraction of incident power (or flux) assuming that
    the E-field of the incident light is perpendicular to the plane of incidence

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]

    Returns:
        transmitted field amplitude           [-]
    """
    c, d, _ = _cosines(m, theta_i, n_i=n_i, deg=deg)
    ts = 2 * c / (c + d)
    return sympy.Abs(d / c * sympy.Abs(ts) ** 2)


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
        reflected irradiance                  [-]
    """
    return (R_par(m, theta_i, n_i=n_i, deg=deg) + R_per(m, theta_i, n_i=n_i, deg=deg)) / 2


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
        reflected irradiance                  [-]
    """
    return (T_par(m, theta_i, n_i=n_i, deg=deg) + T_per(m, theta_i, n_i=n_i, deg=deg)) / 2


def ellipsometry_rho(m, theta_i, n_i=1, deg=False):
    """
    Calculate the ellipsometer parameter rho.

    Args:
        m:       complex index of refraction of medium    [-]
        theta_i: incidence angle from normal              [radians/degrees]
        n_i:     real refractive index of incident medium [-]
        deg:     theta_i is in degrees                    [True/False]

    Returns:
        ellipsometer parameter rho            [-]
    """
    return r_par_amplitude(m, theta_i, n_i=n_i, deg=deg) / r_per_amplitude(m, theta_i, n_i=n_i, deg=deg)


def ellipsometry_index(rho, theta_i, n_i=1, deg=False):
    """
    Calculate the index of refraction for an isotropic sample.

    Args:
        rho:     r_par_amplitude/r_per_amplitude                  [-]
        theta_i: incidence angle from normal                      [radians/degrees]
        n_i:     real refractive index of incident medium         [-]
        deg:     theta_i is in degrees                            [True/False]

    Returns:
        complex index of refraction           [-]
    """
    if deg:
        theta = theta_i * sympy.pi / 180
    else:
        theta = theta_i
    e_index = sympy.sqrt(1 - 4 * rho * sympy.sin(theta) ** 2 / (1 + rho) ** 2)
    return n_i * sympy.tan(theta) * e_index
