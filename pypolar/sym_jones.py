# pylint: disable=invalid-name
# pylint: disable=bare-except

"""
Useful routines for symbolic manipulation of Jones vectors and matrices

Todo:
    * tests and documentation

Scott Prahl
Apr 2019
"""

import sympy
import pypolar.sym_fresnel as sym_fresnel

__all__ = ['op_linear_polarizer',
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
           'field_vertical']


def op_linear_polarizer(theta):
    """
    Jones matrix operator for a linear polarizer rotated about a normal to
    the surface of the polarizer.
    theta: rotation angle measured from the horizontal plane [radians]
    """

    return sympy.Matrix([[sympy.cos(theta)**2, sympy.sin(theta) * sympy.cos(theta)],
                         [sympy.sin(theta) * sympy.cos(theta), sympy.sin(theta)**2]])


def op_retarder(theta, delta):
    """
    Jones matrix operator for an optical retarder rotated about a normal to
    the surface of the retarder.
    theta: rotation angle between fast-axis and the horizontal plane [radians]
    delta: phase delay introduced between fast and slow-axes         [radians]
    """
    P = sympy.exp(+delta / 2 * sympy.numbers.I)
    Q = sympy.exp(-delta / 2 * sympy.numbers.I)
    D = sympy.sin(delta / 2) * 2 * sympy.numbers.I
    C = sympy.cos(theta)
    S = sympy.sin(theta)
    return sympy.Matrix([[C * C * P + S * S * Q, C * S * D],
                         [C * S * D, C * C * Q + S * S * P]])


def op_attenuator(od):
    """
    Jones matrix operator for an optical attenuator.
    od: base ten optical density  [---]
    """

    return sympy.Matrix([[10**-od / 2, 0], [0, 10**-od / 2]])


def op_mirror():
    """
    Jones matrix operator for a perfect mirror.
    """
    return sympy.Matrix([[1, 0], [0, -1]])


def op_rotation(theta):
    """
    Jones matrix operator to rotate light about the optical axis.

    Args:
        theta : angle of rotation about optical axis  [radians]
    Returns:
        2x2 matrix of the rotation operator           [-]
    """
    return sympy.Matrix([[sympy.cos(theta), sympy.sin(theta)],
                     [-sympy.sin(theta), sympy.cos(theta)]])


def op_quarter_wave_plate(theta):
    """
    Jones matrix operator for an quarter-wave plate rotated about a normal to
    the surface of the plate.

    Args:
        theta : angle from fast-axis to horizontal plane  [radians]
    Returns:
        2x2 matrix of the quarter-wave plate operator     [-]
    """

    return op_retarder(theta, np.pi / 2)


def op_half_wave_plate(theta):
    """
    Jones matrix operator for an half-wave plate rotated about a normal to
    the surface of the plate.

    Args:
        theta : angle from fast-axis to horizontal plane  [radians]
    Returns:
        2x2 matrix of the half-wave plate operator     [-]
    """

    return op_retarder(theta, np.pi)


def op_fresnel_reflection(m, theta):
    """
    Jones matrix operator for Fresnel reflection at angle theta
    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        2x2 matrix of the Fresnel transmission operator     [-]
    """
    return sympy.Matrix([[sym_fresnel.r_par(m, theta), 0],
                     [0, sym_fresnel.r_per(m, theta)]])


def op_fresnel_transmission(m, theta):
    """
    Jones matrix operator for Fresnel transmission at angle theta

    *** THIS IS ALMOST CERTAINLY WRONG ***
    Args:
        m :     complex index of refraction       [-]
        theta : angle from normal to surface      [radians]
    Returns:
        2x2 Fresnel transmission operator           [-]
    """
    c = sympy.cos(theta)
    d = sympy.sqrt(m * m - sympy.sin(theta)**2, dtype=np.complex)
    if m.imag == 0:
        d = np.conjugate(d)
    a = sympy.sqrt(d/c)
    return a*sympy.Matrix([[sym_fresnel.t_par(m, theta), 0], [0, sym_fresnel.t_per(m, theta)]])


def field_linear(theta):
    """Jones vector for linear polarized light at angle theta from horizontal plane"""

    return sympy.Matrix([sympy.cos(theta), sympy.sin(theta)])


def field_right_circular():
    """Jones Vector corresponding to right circular polarized light"""

    return 1 / sympy.sqrt(2) * sympy.Matrix([1, -sympy.numbers.I])


def field_left_circular():
    """Jones Vector corresponding to left circular polarized light"""

    return 1 / sympy.sqrt(2) * sympy.Matrix([1, sympy.numbers.I])


def field_horizontal():
    """Jones Vector corresponding to horizontal polarized light"""

    return field_linear(0)


def field_vertical():
    """Jones Vector corresponding to vertical polarized light"""

    return field_linear(np.pi / 2)
