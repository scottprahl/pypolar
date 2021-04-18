# pylint: disable=invalid-name
# pylint: disable=bare-except
# pylint: disable=no-member

"""
Useful routines for symbolic manipulation of Jones vectors and matrices.

Creating Jones vectors for specific polarization states::

    * field_linear(angle)
    * field_left_circular()
    * field_right_circular()
    * field_horizontal()
    * field_vertical()
    * field_ellipsometry(tanpsi, Delta)
    * field_elliptical(azimuth, elliptic_angle)

Creating Jones Matrices for polarizing elements::

    * op_linear_polarizer(angle)
    * op_retarder(fast_axis_angle, retardance)
    * op_attenuator(optical_density)
    * op_mirror()
    * op_rotation(angle)
    * op_quarter_wave_plate(fast_axis_angle)
    * op_half_wave_plate(fast_axis_angle)
    * op_fresnel_reflection(index_of_refraction, angle)
    * op_fresnel_transmission(index_of_refraction, angle)

Interpreting the polarization state::

    * use_alternate_convention(boolean)
    * interpret(jones_vector)
    * intensity(jones_vector)
    * phase(jones_vector)
    * ellipse_azimuth(jones_vector)
    * ellipse_axes(jones_vector)
    * ellipticity(jones_vector)
    * ellipticity_angle(jones_vector)
    * amplitude_ratio(jones_vector)
    * amplitude_ratio_angle(jones_vector)
    * polarization_variable(jones_vector)

Converting to Mueller formalism::

    * jones_op_to_mueller_op(jones_matrix)
    * jones_to_stokes(jones_vector)
"""

import sympy
import pypolar.sym_fresnel as sym_fresnel

__all__ = ('op_linear_polarizer',
           'op_retarder',
           'op_attenuator',
           'op_mirror',
           'op_rotation',
           'op_quarter_wave_plate',
           'op_half_wave_plate',
           'op_fresnel_reflection',
           'op_fresnel_transmission',
           'field_linear',
           'field_left_circular',
           'field_right_circular',
           'field_horizontal',
           'field_vertical',
           'intensity',
           'phase')


def op_linear_polarizer(theta):
    """
    Jones matrix operator for a rotated linear polarizer.

    The polarizer is rotated around a normal to its surface.

    Args:
        theta: rotation angle measured from the horizontal plane [radians]
    """
    return sympy.Matrix([[sympy.cos(theta)**2, sympy.sin(theta) * sympy.cos(theta)],
                         [sympy.sin(theta) * sympy.cos(theta), sympy.sin(theta)**2]])


def op_retarder(theta, delta):
    """
    Jones matrix operator for a rotated optical retarder.

    The retarder is rotated around a normal to its surface.

    Args:
        theta: rotation angle between fast-axis and the horizontal plane [radians]
        delta: phase delay introduced between fast and slow-axes         [radians]
    """
    P = sympy.exp(+delta / 2 * sympy.I)
    Q = sympy.exp(-delta / 2 * sympy.I)
    D = sympy.sin(delta / 2) * 2 * sympy.I
    C = sympy.cos(theta)
    S = sympy.sin(theta)
    return sympy.Matrix([[C * C * P + S * S * Q, C * S * D],
                         [C * S * D, C * C * Q + S * S * P]])


def op_attenuator(t):
    """
    Jones matrix operator for an optical attenuator.

    Args:
        t: fraction of intensity passing through attenuator [---]
    """
    f = sympy.sqrt(t)
    return sympy.Matrix([[f, 0], [0, f]])


def op_neutral_density_filter(nd):
    """
    Jones matrix operator for a neutral density filter with decadic attenuation.

    Args:
        nd: base ten optical density  [---]
    """
    return sympy.Matrix([[10**-nd / 2, 0], [0, 10**-nd / 2]])

def op_mirror():
    """Jones matrix operator for a perfect mirror."""
    return sympy.Matrix([[1, 0], [0, -1]])


def op_rotation(theta):
    """
    Jones matrix operator to rotate light around the optical axis.

    Args:
        theta : angle of rotation about optical axis  [radians]
    Returns:
        2x2 matrix of the rotation operator           [-]
    """
    return sympy.Matrix([[sympy.cos(theta), sympy.sin(theta)],
                         [-sympy.sin(theta), sympy.cos(theta)]])


def op_quarter_wave_plate(theta):
    """
    Jones matrix operator for a rotated quarter-wave plate.

    The QWP is rotated about a normal to its surface.

    Args:
        theta : angle from fast-axis to horizontal plane  [radians]
    Returns:
        2x2 matrix of the quarter-wave plate operator     [-]
    """
    return op_retarder(theta, sympy.pi / 2)


def op_half_wave_plate(theta):
    """
    Jones matrix operator for a rotated half-wave plate.

    The HWP is rotated about a normal to its surface.

    Args:
        theta : angle from fast-axis to horizontal plane  [radians]
    Returns:
        2x2 matrix of the half-wave plate operator     [-]
    """
    return op_retarder(theta, sympy.pi)


def op_fresnel_reflection(m, theta):
    """
    Jones matrix operator for Fresnel reflection at angle.

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        2x2 matrix of the Fresnel transmission operator     [-]
    """
    return sympy.Matrix([[sym_fresnel.r_par_amplitude(m, theta), 0],
                         [0, sym_fresnel.r_per_amplitude(m, theta)]])


def op_fresnel_transmission(m, theta):
    """
    Jones matrix operator for Fresnel transmission at angle theta.

    *** THIS IS ALMOST CERTAINLY WRONG ***

    Args:
        m :     complex index of refraction       [-]
        theta : angle from normal to surface      [radians]
    Returns:
        2x2 Fresnel transmission operator           [-]
    """
    c = sympy.cos(theta)
    d = sympy.sqrt(m * m - sympy.sin(theta)**2)
    if m.imag == 0:
        d = sympy.conjugate(d)
    a = sympy.sqrt(d/c)
    tpar = sym_fresnel.t_par_amplitude(m, theta)
    tper = sym_fresnel.t_per_amplitude(m, theta)
    return a*sympy.Matrix([[tpar, 0], [0, tper]])


def field_linear(theta):
    """Jones vector for linear polarized light at angle theta from horizontal plane."""
    return sympy.Matrix([sympy.cos(theta), sympy.sin(theta)])


def field_right_circular():
    """Jones Vector for right circular polarized light."""
    return 1 / sympy.sqrt(2) * sympy.Matrix([1, -sympy.I])


def field_left_circular():
    """Jones Vector for left circular polarized light."""
    return 1 / sympy.sqrt(2) * sympy.Matrix([1, sympy.I])


def field_horizontal():
    """Jones Vector for horizontal polarized light."""
    return field_linear(0)


def field_vertical():
    """Jones Vector for vertical polarized light."""
    return field_linear(sympy.pi / 2)


def field_elliptical(A, B):
    """Jones Vector for elliptically polarized light."""
    return sympy.Matrix([A, B])


def intensity(J):
    """Return the intensity."""
    return sympy.conjugate(J.T).dot(J)


def phase(J):
    """Return the phase."""
    gamma = sympy.arg(J[1]) - sympy.arg(J[0])
    return gamma


def ellipse_orientation(J):
    """
    Return the angle between the major semi-axis and the x-axis.

    This angle is sometimes called the azimuth or psi.
    """
    Ex = sympy.Abs(J[0, :])
    Ey = sympy.Abs(J[1, :])
    delta = phase(J)
    numer = 2 * Ex * Ey * sympy.cos(delta)
    denom = Ex**2 - Ey**2
    psi = 0.5 * sympy.arctan2(numer, denom)
    return psi


def ellipse_ellipticity(J):
    """Return the ellipticity of the polarization ellipse."""
    delta = phase(J)
    psi = ellipse_orientation(J)
    chi = 0.5 * sympy.arcsin(sympy.sin(2 * psi) * sympy.sin(delta))
    return chi


def ellipse_axes(J):
    """Return the semi-major and semi-minor axes of the polarization ellipse."""
    Exo, Eyo = sympy.conjugate(J.T)*J
    psi = ellipse_orientation(J)
    delta = phase(J)
    C = sympy.cos(psi)
    S = sympy.sin(psi)
    asqr = (Exo * C)**2 + (Eyo * S)**2 + 2 * Exo * Eyo * C * S * sympy.cos(delta)
    bsqr = (Exo * S)**2 + (Eyo * C)**2 - 2 * Exo * Eyo * C * S * sympy.cos(delta)
    return sympy.sqrt(abs(asqr)), sympy.sqrt(abs(bsqr))
