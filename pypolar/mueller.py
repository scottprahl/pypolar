# pylint: disable=invalid-name
"""
Useful basic routines for managing polarization using the Stokes/Mueller calculus

Todo:
    * complete Jupyter notebook documentation
    * test mueller_to_jones()
    * improve internal documentation

Scott Prahl
May 2018
"""

import numpy as np
import pypolar.jones
import pypolar.fresnel

__all__ = ['op_linear_polarizer',
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
           'mueller_to_jones',
           'interpret']


def op_linear_polarizer(theta):
    """
    Mueller matrix operator for a linear polarizer rotated about a normal to
    the surface of the polarizer.

    theta: rotation angle measured from the horizontal plane [radians]
    """

    C2 = np.cos(2 * theta)
    S2 = np.sin(2 * theta)
    lp = np.array([[1, C2, S2, 0],
                   [C2, C2**2, C2 * S2, 0],
                   [S2, C2 * S2, S2 * S2, 0],
                   [0, 0, 0, 0]])
    return 0.5 * lp


def op_retarder(theta, delta):
    """
    Mueller matrix operator for an optical retarder rotated about a normal to
    the surface of the retarder.

    theta: rotation angle between fast-axis and the horizontal plane [radians]
    delta: phase delay introduced between fast and slow-axes         [radians]
    """

    C2 = np.cos(2 * theta)
    S2 = np.sin(2 * theta)
    C = np.cos(delta)
    S = np.sin(delta)
    ret = np.array([[1, 0, 0, 0],
                    [0, C2**2 + C * S2**2, (1 - C) * S2 * C2, -S * S2],
                    [0, (1 - C) * C2 * S2, S2**2 + C * C2**2, S * C2],
                    [0, S * S2, -S * C2, C]])
    return ret


def op_attenuator(transmittance):
    """
    Mueller matrix operator for an optical attenuator.

    t : fraction of light getting through attenuator [---]
    """
    att = np.array([[t, 0, 0, 0],
                    [0, t, 0, 0],
                    [0, 0, t, 0],
                    [0, 0, 0, t]])
    return att


def op_mirror():
    """
    Mueller matrix operator for a perfect mirror.
    """

    mir = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, -1]])
    return mir


def op_rotation(theta):
    """
    Mueller matrix operator to rotate light about the optical axis.

    theta: rotation angle  [radians]
    """

    C2 = np.cos(2 * theta)
    S2 = np.sin(2 * theta)
    rot = np.array([[1, 0, 0, 0],
                    [0, C2, S2, 0],
                    [0, -S2, C2, 0],
                    [0, 0, 0, 1]])
    return rot


def op_quarter_wave_plate(theta):
    """
    Mueller matrix operator for an quarter-wave plate rotated about a normal to
    the surface of the plate.

    theta: rotation angle between fast-axis and the horizontal plane [radians]
    """

    C2 = np.cos(2 * theta)
    S2 = np.sin(2 * theta)
    qwp = np.array([[1, 0, 0, 0],
                    [0, C2**2, C2 * S2, -S2],
                    [0, C2 * S2, S2 * S2, C2],
                    [0, S2, -C2, 0]])
    return qwp


def op_half_wave_plate(theta):
    """
    Mueller matrix operator for an half-wave plate rotated about a normal to
    the surface of the plate.

    theta: rotation angle between fast-axis and the horizontal plane [radians]
    """

    C2 = np.cos(2 * theta)
    S2 = np.sin(2 * theta)
    qwp = np.array([[1, 0, 0, 0],
                    [0, C2**2 - S2**2, 2 * C2 * S2, 0],
                    [0, 2 * C2 * S2, S2 * S2 - C2**2, 0],
                    [0, 0, 0, -1]])
    return qwp


def op_fresnel_reflection(m, theta):
    """
    Mueller matrix operator for Fresnel reflection at angle theta

    Convert from the Jones operator to ensure that phase change are
    handled properly
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
    Mueller matrix operator for Fresnel transmission at angle theta

    Unclear if phase changes are handled properly.  See Collett, "Mueller-Stokes
    Matrix Formulation of Fresnel's Equations," Am. J. Phys. 39, 517 (1971).

    Args:
        m :     complex index of refraction       [-]
        theta : angle from normal to surface      [radians]
    Returns:
        4x4 Fresnel transmission operator         [-]
    """
    tau_p = pypolar.fresnel.T_par(m, theta)
    tau_s = pypolar.fresnel.T_per(m, theta)
    a = tau_s + tau_p
    b = tau_s - tau_p
    c = 2 * np.sqrt(tau_s*tau_p)
    mat = np.array([[a, b, 0, 0],
                    [b, a, 0, 0],
                    [0, 0, c, 0],
                    [0, 0, 0, c]])
    return 0.5 * mat


def stokes_linear(theta):
    """Stokes vector for linear polarized light at angle theta from
       horizontal plane"""

    return np.array([1, np.cos(2*theta), np.sin(2*theta), 0])


def stokes_right_circular():
    """Stokes vector corresponding to right circular polarized light"""

    return np.array([1, 0, 0, 1])


def stokes_left_circular():
    """Stokes vector corresponding to left circular polarized light"""

    return np.array([1, 0, 0, -1])


def stokes_horizontal():
    """
    Stokes vector corresponding to horizontal polarized light
    """
    return np.array([1, 1, 0, 0])


def stokes_vertical():
    """
    Stokes vector corresponding to vertical polarized light
    """
    return np.array([1, -1, 0, 0])


def stokes_unpolarized():
    """
    Stokes vector corresponding to vertical polarized light
    """
    return np.array([1, 0, 0, 0])


def intensity(S):
    """
    Returns the intensity
    """
    return S[0]


def degree_of_polarization(S):
    """
    Returns the degree of polarization
    """
    return S[0]/np.sqrt(S[1]**2+s[2]**2+s[3]**2)


def ellipse_orientation(S):
    """
    Returns the angle between the major semi-axis and the x-axis of
    the polarization ellipse (often represented by psi)
    """
    return 1/2 * np.arctan2(S[2], S[1])


def ellipse_ellipticity(S):
    """
    Returns the ellipticity of the polarization ellipse (often represented by chi)
    """
    return 1/2 * np.arcsin(S[3]/S[0])


def ellipse_axes(S):
    """
    Returns the semi-major and semi-minor axes of the polarization ellipse.
    """
    absL = np.sqrt(S[1]**2 + S[2]**2)
    A = np.sqrt((S[0] + absL)/2)
    B = np.sqrt((S[0] - absL)/2)
    return A, B


def stokes_to_jones(S):
    """
    Convert a Stokes vector to a Jones vector

    The Jones vector only has the polarized part of the intensity, of course.
    Also, since the Jones vector can differ by an arbitrary phase, the phase
    is chosen which makes the horizontal component purely real.

    Inputs:
        S : a Stokes vector

    Returns:
         the Jones vector corresponding to
    """

    # Calculate the degree of polarization
    p = np.sqrt(S[1]**2 + S[2]**2 + S[3]**2) / S[0]

    # Normalize the Stokes parameters (first one will be 1, of course)
    Q = S[1] / (S[0] * p)
    U = S[2] / (S[0] * p)
    V = S[3] / (S[0] * p)

    # the Jones components
    A = np.sqrt((1 + Q) / 2)
    if A == 0:
        B = 1
    else:
        B = complex(U, -V) / (2 * A)

    # put them together in a vector with the amplitude of the polarized part
    return np.sqrt(S[0] * p) * np.array([A, B])


def mueller_to_jones(M):
    """
    Convert a Mueller matrix to a Jones matrix

    Theocaris, Matrix Theory of Photoelasticity, eqns 4.70-4.76, 1979

    Inputs:
        M : a 4x4 Mueller matrix

    Returns:
         the corresponding 2x2 Jones matrix
    """
    A = np.empty((2, 2))
    A[0, 0] = np.sqrt((M[0, 0]+M[0, 1]+M[1, 0]+M[1, 1])/2)
    A[0, 1] = np.sqrt((M[0, 0]+M[0, 1]-M[1, 0]-M[1, 1])/2)
    A[1, 0] = np.sqrt((M[0, 0]-M[0, 1]+M[1, 0]-M[1, 1])/2)
    A[1, 1] = np.sqrt((M[0, 0]-M[0, 1]-M[1, 0]+M[1, 1])/2)

    theta = np.empty((2, 2))
    theta[0, 0] = 0
    theta[0, 1] = -np.arctan2(M[0, 3]+M[1, 3], M[0, 2]+M[1, 2])
    theta[1, 0] = np.arctan2(M[3, 0]+M[3, 1], M[2, 0]+M[2, 1])
    theta[1, 1] = np.arctan2(M[3, 2]-M[2, 3], M[2, 2]+M[3, 3])

    return A*np.exp(1j*theta)


def interpret(S):
    '''
    Interprets a Stokes vector

    Parameters
    S    : A Stokes vector

    Examples
    -------
    interpret([1, 0, 0, 0]) --> "Unpolarized Light"

    '''

    try:
        S0, S1, S2, S3 = S
    except:
        print("Stokes vector must have four real elements")
        return 0

    eps = 1e-12
    s = "not implemented yet"
    return s
