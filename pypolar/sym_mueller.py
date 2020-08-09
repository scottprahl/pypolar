# pylint: disable=invalid-name
# pylint: disable=no-member

"""
Symbolic manipulation of polarization using the Stokes/Mueller calculus.

The routines are broken up into four groups: (1) creating Stokes vectors, (2)
creating Mueller matrix operators, (3) interpretation, and (4) conversion.

Functions to create Stokes vectors::

    stokes_linear(angle)
    stokes_left_circular()
    stokes_right_circular()
    stokes_horizontal()
    stokes_vertical()
    stokes_unpolarized()
    stokes_elliptical(DOP, azimuth, ellipticity)
    stokes_ellipsometry(tanpsi, Delta)

Functions to create Mueller matrix operators::

    op_linear_polarizer(angle)
    op_retarder(fast_axis_angle, phase_delay)
    op_attenuator(optical_density)
    op_mirror()
    op_rotation(angle)
    op_quarter_wave_plate(fast_axis_angle)
    op_half_wave_plate(fast_axis_angle)
    op_fresnel_reflection(index_of_refraction, incidence_angle)
    op_fresnel_transmission(index_of_refraction, incidence_angle)

Functions to interpret Stokes vectors::

    intensity(stokes_vector)
    degree_of_polarization(stokes_vector)
    ellipse_orientation(stokes_vector)
    ellipse_ellipticity(stokes_vector)
    ellipse_axes(stokes_vector)

Functions to convert::

    stokes_to_jones(stokes_vector)
    mueller_to_jones(mueller_matrix)
"""

import sympy
import pypolar.jones

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
           'stokes_linear',
           'stokes_left_circular',
           'stokes_right_circular',
           'stokes_horizontal',
           'stokes_vertical',
           'stokes_unpolarized',
           'intensity',
           'degree_of_polarization',
           'ellipse_orientation',
           'ellipse_ellipticity',
           'ellipse_axes',
           'stokes_to_jones',
           'mueller_to_jones')


def op_linear_polarizer(theta):
    """
    Mueller matrix operator for a rotated linear polarizer.

    The polarizer is rotated about a normal to its surface.

    Args:
        theta: rotation angle measured from the horizontal plane [radians]
    """
    C2 = sympy.cos(2 * theta)
    S2 = sympy.sin(2 * theta)
    lp = sympy.Matrix([[1, C2, S2, 0],
                       [C2, C2**2, C2 * S2, 0],
                       [S2, C2 * S2, S2 * S2, 0],
                       [0, 0, 0, 0]])
    return 0.5 * lp


def op_retarder(theta, delta):
    """
    Mueller matrix operator for a rotated optical retarder.

    The retarder is rotated about a normal to its surface.

    Args:
        theta: rotation angle between fast-axis and the horizontal plane [radians]
        delta: phase delay introduced between fast and slow-axes         [radians]
    """
    C2 = sympy.cos(2 * theta)
    S2 = sympy.sin(2 * theta)
    C = sympy.cos(delta)
    S = sympy.sin(delta)
    ret = sympy.Matrix([[1, 0, 0, 0],
                        [0, C2**2 + C * S2**2, (1 - C) * S2 * C2, -S * S2],
                        [0, (1 - C) * C2 * S2, S2**2 + C * C2**2, S * C2],
                        [0, S * S2, -S * C2, C]])
    return ret


def op_attenuator(t):
    """
    Mueller matrix operator for an optical attenuator.

    Args:
        t: fraction of light getting through attenuator  [---]
    """
    att = sympy.Matrix([[t, 0, 0, 0],
                        [0, t, 0, 0],
                        [0, 0, t, 0],
                        [0, 0, 0, t]])
    return att


def op_mirror():
    """Mueller matrix operator for a perfect mirror."""
    mir = sympy.Matrix([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, -1]])
    return mir


def op_rotation(theta):
    """
    Mueller matrix operator to rotate light about the optical axis.

    Args:
        theta: rotation angle  [radians]
    """
    C2 = sympy.cos(2 * theta)
    S2 = sympy.sin(2 * theta)
    rot = sympy.Matrix([[1, 0, 0, 0],
                        [0, C2, S2, 0],
                        [0, -S2, C2, 0],
                        [0, 0, 0, 1]])
    return rot


def op_quarter_wave_plate(theta):
    """
    Mueller matrix operator for a rotated quarter-wave plate.

    The QWP is rotated about a normal to its surface.

    Args:
        theta: rotation angle between fast-axis and the horizontal plane [radians]
    """
    C2 = sympy.cos(2 * theta)
    S2 = sympy.sin(2 * theta)
    qwp = sympy.Matrix([[1, 0, 0, 0],
                        [0, C2**2, C2 * S2, -S2],
                        [0, C2 * S2, S2 * S2, C2],
                        [0, S2, -C2, 0]])
    return qwp


def op_half_wave_plate(theta):
    """
    Mueller matrix operator for rotated half-wave plate.

    The HWP is rotated about a normal to its surface.

    Args:
        theta: rotation angle between fast-axis and the horizontal plane [radians]
    """
    C2 = sympy.cos(2 * theta)
    S2 = sympy.sin(2 * theta)
    qwp = sympy.Matrix([[1, 0, 0, 0],
                        [0, C2**2 - S2**2, 2 * C2 * S2, 0],
                        [0, 2 * C2 * S2, S2 * S2 - C2**2, 0],
                        [0, 0, 0, -1]])
    return qwp


def op_fresnel_reflection(m, theta):
    """
    Mueller matrix operator for Fresnel reflection.

    Convert from the Jones operator to ensure that phase
    change are handled properly.

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        4x4 Fresnel reflection operator       [-]
    """
    J = pypolar.jones.op_fresnel_reflection(m, theta)
    R = pypolar.jones.jones_op_to_mueller_op(J)
    return R


def op_fresnel_transmission(m, theta):
    """
    Mueller matrix operator for Fresnel transmission.

    Unclear if phase changes are handled properly.  See Collett,
    "Mueller-Stokes Matrix Formulation of Fresnel's Equations,"
    Am. J. Phys. 39, 517 (1971).

    Args:
        m :     complex index of refraction       [-]
        theta : angle from normal to surface      [radians]
    Returns:
        4x4 Fresnel transmission operator         [-]
    """
    tau_p = sym_fresnel.T_par(m, theta)
    tau_s = sym_fresnel.T_per(m, theta)
    a = tau_s + tau_p
    b = tau_s - tau_p
    c = 2 * sympy.sqrt(tau_s*tau_p)
    mat = sympy.Matrix([[a, b, 0, 0],
                        [b, a, 0, 0],
                        [0, 0, c, 0],
                        [0, 0, 0, c]])
    return 0.5 * mat


def stokes_linear(theta):
    """
    Stokes vector for linear polarized light at angle.

    Args:
        m :     complex index of refraction       [-]
        theta : angle from horizontal plane      [radians]
    """
    return sympy.Matrix([1, sympy.cos(2*theta), sympy.sin(2*theta), 0])


def stokes_right_circular():
    """Stokes vector for right circular polarized light."""
    return sympy.Matrix([1, 0, 0, 1])


def stokes_left_circular():
    """Stokes vector for left circular polarized light."""
    return sympy.Matrix([1, 0, 0, -1])


def stokes_horizontal():
    """Stokes vector for horizontal polarized light."""
    return sympy.Matrix([1, 1, 0, 0])


def stokes_vertical():
    """Stokes vector for vertical polarized light."""
    return sympy.Matrix([1, -1, 0, 0])


def stokes_unpolarized():
    """Stokes vector for unpolarized light."""
    return sympy.Matrix([1, 0, 0, 0])


def intensity(S):
    """Return the intensity."""
    return S[0]


def degree_of_polarization(S):
    """Return the degree of polarization."""
    return S[0]/sympy.sqrt(S[1]**2+S[2]**2+S[3]**2)


def ellipse_orientation(S):
    """
    Return orientation of the polarization ellipse.

    The orientation is the angle between the major semi-axis and the x-axis of
    the polarization ellipse (often represented by psi).
    """
    return 1/2 * sympy.arctan2(S[2], S[1])


def ellipse_ellipticity(S):
    """
    Return ellipticity of the polarization ellipse.

    The ellipticity of the polarization ellipse (often
    represented by chi)
    """
    return 1/2 * sympy.arcsin(S[3]/S[0])


def ellipse_axes(S):
    """
    Return the semi-major and semi-minor axes.

    These are the axes of the polarization ellipse.
    """
    absL = sympy.sqrt(S[1]**2 + S[2]**2)
    A = sympy.sqrt((S[0] + absL)/2)
    B = sympy.sqrt((S[0] - absL)/2)
    return A, B


def stokes_to_jones(S):
    """
    Convert a Stokes vector to a Jones vector.

    This conversion loses some of the information in the Stokes
    vector because the unpolarized fraction is lost.  Furthermore,
    since the Jones vector can differ by an arbitrary phase,
    the phase is chosen to make the horizontal component real.

    Args:
        S : a Stokes vector

    Returns:
         a corresponding Jones vector
    """
    # Calculate the degree of polarization
    p = sympy.sqrt(S[1]**2 + S[2]**2 + S[3]**2) / S[0]

    # Normalize the Stokes parameters (first one will be 1, of course)
    Q = S[1] / (S[0] * p)
    U = S[2] / (S[0] * p)
    V = S[3] / (S[0] * p)

    # the Jones components
    A = sympy.sqrt((1 + Q) / 2)
    if A == 0:
        B = 1
    else:
        B = sympy.complex(U, -V) / (2 * A)

    # put them together in a vector with the amplitude of the polarized part
    return sympy.sqrt(S[0] * p) * sympy.Matrix([A, B])


def mueller_to_jones(M):
    """
    Convert a Mueller matrix to a Jones matrix.

    Theocaris, Matrix Theory of Photoelasticity, eqns 4.70-4.76, 1979

    Args:
        M : a 4x4 Mueller matrix

    Returns:
         the corresponding 2x2 Jones matrix
    """
    A = sympy.empty((2, 2))
    A[0, 0] = sympy.sqrt((M[0, 0]+M[0, 1]+M[1, 0]+M[1, 1])/2)
    A[0, 1] = sympy.sqrt((M[0, 0]+M[0, 1]-M[1, 0]-M[1, 1])/2)
    A[1, 0] = sympy.sqrt((M[0, 0]-M[0, 1]+M[1, 0]-M[1, 1])/2)
    A[1, 1] = sympy.sqrt((M[0, 0]-M[0, 1]-M[1, 0]+M[1, 1])/2)

    theta = sympy.empty((2, 2))
    theta[0, 0] = 0
    theta[0, 1] = -sympy.arctan2(M[0, 3]+M[1, 3], M[0, 2]+M[1, 2])
    theta[1, 0] = sympy.arctan2(M[3, 0]+M[3, 1], M[2, 0]+M[2, 1])
    theta[1, 1] = sympy.arctan2(M[3, 2]-M[2, 3], M[2, 2]+M[3, 3])

    return A*sympy.exp(1j*theta)
