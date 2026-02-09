"""
Useful routines for symbolic manipulation of Jones vectors and matrices.

Creating Jones vectors for specific polarization states::

    * field_linear(angle)
    * field_left_circular()
    * field_right_circular()
    * field_horizontal()
    * field_vertical()
    * field_elliptical(A, B)

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

    * intensity(jones_vector)
    * phase(jones_vector)
    * ellipse_orientation(jones_vector)
    * ellipse_axes(jones_vector)
    * ellipse_ellipticity(jones_vector)

Converting to Mueller formalism::

    * jones_op_to_mueller_op(jones_matrix)
    * jones_to_stokes(jones_vector)
"""

import sympy
from pypolar import sym_fresnel

__all__ = (
    "op_linear_polarizer",
    "op_retarder",
    "op_attenuator",
    "op_neutral_density_filter",
    "op_mirror",
    "op_rotation",
    "op_quarter_wave_plate",
    "op_half_wave_plate",
    "op_fresnel_reflection",
    "op_fresnel_transmission",
    "field_linear",
    "field_left_circular",
    "field_right_circular",
    "field_horizontal",
    "field_vertical",
    "field_elliptical",
    "intensity",
    "phase",
    "ellipse_orientation",
    "ellipse_ellipticity",
    "ellipse_axes",
    "jones_op_to_mueller_op",
    "jones_to_stokes",
)


def op_linear_polarizer(theta):
    """
    Jones matrix operator for a rotated linear polarizer.

    The polarizer is rotated around a normal to its surface.

    Args:
        theta: rotation angle measured from the horizontal plane [radians]
    """
    return sympy.Matrix(
        [
            [sympy.cos(theta) ** 2, sympy.sin(theta) * sympy.cos(theta)],
            [sympy.sin(theta) * sympy.cos(theta), sympy.sin(theta) ** 2],
        ]
    )


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
    return sympy.Matrix([[C * C * P + S * S * Q, C * S * D], [C * S * D, C * C * Q + S * S * P]])


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
    return sympy.Matrix([[sympy.cos(theta), sympy.sin(theta)], [-sympy.sin(theta), sympy.cos(theta)]])


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


def op_fresnel_reflection(m, theta, n_i=1):
    """
    Jones matrix operator for Fresnel reflection at angle.

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
        n_i:    real refractive index of incident medium [-]

    Returns:
        2x2 matrix of the Fresnel reflection operator     [-]
    """
    m_rel = m / n_i
    return sympy.Matrix(
        [[sym_fresnel.r_par_amplitude(m_rel, theta), 0], [0, sym_fresnel.r_per_amplitude(m_rel, theta)]]
    )


def op_fresnel_transmission(m, theta, n_i=1):
    """
    Jones matrix operator for Fresnel transmission at angle theta.

    This operator acts on field amplitudes in the p/s basis and returns the
    transmitted field amplitudes.  It does not include irradiance
    normalization factors.

    Args:
        m :     complex index of refraction       [-]
        theta : angle from normal to surface      [radians]
        n_i:    real refractive index of incident medium [-]

    Returns:
        2x2 Fresnel transmission operator           [-]
    """
    m_rel = m / n_i
    tpar = sym_fresnel.t_par_amplitude(m_rel, theta)
    tper = sym_fresnel.t_per_amplitude(m_rel, theta)
    return sympy.Matrix([[tpar, 0], [0, tper]])


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
    return sympy.conjugate(J.T) * J


def phase(J):
    """Return the phase."""
    gamma = sympy.arg(J[1]) - sympy.arg(J[0])
    return gamma


def ellipse_orientation(J):
    """
    Return the angle between the major semi-axis and the x-axis.

    This angle is sometimes called the azimuth or psi.
    """
    Ex = sympy.Abs(J[0])
    Ey = sympy.Abs(J[1])
    delta = phase(J)
    numer = 2 * Ex * Ey * sympy.cos(delta)
    denom = Ex**2 - Ey**2
    psi = 0.5 * sympy.atan2(numer, denom)
    return psi


def ellipse_ellipticity(J):
    """Return the ellipticity of the polarization ellipse."""
    delta = phase(J)
    psi = ellipse_orientation(J)
    chi = 0.5 * sympy.asin(sympy.sin(2 * psi) * sympy.sin(delta))
    return chi


def ellipse_axes(J):
    """Return the semi-major and semi-minor axes of the polarization ellipse."""
    Exo = sympy.Abs(J[0])
    Eyo = sympy.Abs(J[1])
    psi = ellipse_orientation(J)
    delta = phase(J)
    C = sympy.cos(psi)
    S = sympy.sin(psi)
    asqr = (Exo * C) ** 2 + (Eyo * S) ** 2 + 2 * Exo * Eyo * C * S * sympy.cos(delta)
    bsqr = (Exo * S) ** 2 + (Eyo * C) ** 2 - 2 * Exo * Eyo * C * S * sympy.cos(delta)
    return sympy.sqrt(abs(asqr)), sympy.sqrt(abs(bsqr))


def jones_op_to_mueller_op(J):
    """
    Convert a symbolic 2x2 Jones matrix to a symbolic 4x4 Mueller matrix.

    Args:
        J: Jones matrix in the x/y basis
    """
    sigma = (
        sympy.Matrix([[1, 0], [0, 1]]),
        sympy.Matrix([[1, 0], [0, -1]]),
        sympy.Matrix([[0, 1], [1, 0]]),
        sympy.Matrix([[0, sympy.I], [-sympy.I, 0]]),
    )

    J_dagger = sympy.conjugate(J.T)
    M = sympy.Matrix.zeros(4, 4)
    for i in range(4):
        for j in range(4):
            M[i, j] = sympy.trace(sigma[i] * J * sigma[j] * J_dagger) / 2
    return M


def jones_to_stokes(J):
    """
    Convert a symbolic Jones vector to a symbolic Stokes vector.

    Args:
        J: Jones vector in the x/y basis
    """
    E = sympy.Matrix(J)
    if E.shape == (1, 2):
        E = E.T

    sigma = (
        sympy.Matrix([[1, 0], [0, 1]]),
        sympy.Matrix([[1, 0], [0, -1]]),
        sympy.Matrix([[0, 1], [1, 0]]),
        sympy.Matrix([[0, sympy.I], [-sympy.I, 0]]),
    )

    E_dagger = sympy.conjugate(E.T)
    return sympy.Matrix([(E_dagger * s * E)[0] for s in sigma])
