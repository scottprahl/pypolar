# pylint: disable=invalid-name
# pylint: disable=bare-except

"""
Useful basic routines for managing polarization with the Stokes/Mueller calculus.

To Do
    * complete Jupyter notebook documentation
    * test mueller_to_jones()
    * improve internal documentation

Scott Prahl
May 2018
"""

import numpy as np
import pypolar.jones
import pypolar.fresnel

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
           'stokes_elliptical',
           'intensity',
           'degree_of_polarization',
           'ellipse_orientation',
           'ellipse_ellipticity',
           'ellipse_axes',
           'stokes_to_jones',
           'mueller_to_jones',
           'interpret')


def op_linear_polarizer(theta):
    """
    Mueller matrix operator for a rotated linear polarizer.

    The polarizer is rotated around a normal to its surface.

    Args:
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
    Mueller matrix operator for an rotated optical retarder.

    The retarder is rotated around a normal to its surface.

    Args:
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


def op_attenuator(t):
    """
    Mueller matrix operator for an optical attenuator.

    Args:
        t : fraction of light getting through attenuator [---]
    """
    att = np.array([[t, 0, 0, 0],
                    [0, t, 0, 0],
                    [0, 0, t, 0],
                    [0, 0, 0, t]])
    return att


def op_mirror():
    """Mueller matrix operator for a perfect mirror."""
    mir = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, -1]])
    return mir


def op_rotation(theta):
    """
    Mueller matrix operator to rotate light around the optical axis.

    Args:
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
    Mueller matrix operator for an quarter-wave plate.

    Args:
        theta: rotation angle between fast-axis and the horizontal plane [radians]

    Returns:
        a Mueller matrix operator for the rotated quarter-wave plate.
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
    Mueller matrix for a rotated half-wave plate.

    Args:
        theta: rotation angle between fast-axis and the horizontal plane [radians]

    Returns:
        a Mueller matrix operator for the rotated half-wave plate.
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
    Mueller matrix operator for Fresnel reflection at angle theta.

    These are based on Collett, Mueller-Stokes Matrix Formulation of Fresnel
    equation, Am. J Phys., 39, 1971.

    The changes in direction and detector orientation are included in the
    Mueller matrix.  See Clark, *Stellar Polarimetry*, Appendix A.

    Still needs sign testing for angles above Brewster's angle.

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
    Returns:
        4x4 Fresnel reflection Mueller matrix       [-]
    """
    rho_p = pypolar.fresnel.r_par(m, theta)
    rho_s = pypolar.fresnel.r_per(m, theta)
    a = abs(rho_s)**2 + abs(rho_p)**2
    b = abs(rho_s)**2 - abs(rho_p)**2
    c = 2 * rho_s * rho_p
    mat = np.array([[a, b, 0, 0],
                    [b, a, 0, 0],
                    [0, 0, c, 0],
                    [0, 0, 0, c]])
    return 0.5 * mat


def op_fresnel_transmission(m, theta):
    """
    Mueller matrix operator for Fresnel transmission at angle theta.

    These are based on Collett, Mueller-Stokes Matrix Formulation of Fresnel
    equation, Am. J Phys., 39, 1971.

    Still needs sign testing for angles above Brewster's angle.

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
    c = 2 * np.sqrt(tau_s * tau_p)
    mat = np.array([[a, b, 0, 0],
                    [b, a, 0, 0],
                    [0, 0, c, 0],
                    [0, 0, 0, c]])
    return 0.5 * mat


def stokes_linear(theta):
    """Stokes vector for light polarized at angle theta from the horizontal plane."""
    return np.array([1, np.cos(2*theta), np.sin(2*theta), 0])


def stokes_right_circular():
    """Stokes vector for right circular polarized light."""
    return np.array([1, 0, 0, 1])


def stokes_left_circular():
    """Stokes vector for left circular polarized light."""
    return np.array([1, 0, 0, -1])


def stokes_horizontal():
    """Stokes vector for horizontal polarized light."""
    return np.array([1, 1, 0, 0])


def stokes_vertical():
    """Stokes vector for vertical polarized light."""
    return np.array([1, -1, 0, 0])


def stokes_unpolarized():
    """Stokes vector for vertical polarized light."""
    return np.array([1, 0, 0, 0])


def stokes_elliptical(DOP, azimuth, ellipticity):
    """
    Stokes vector for partially polarized elliptically polarized light.

    Args:
        DOP: degree of polarization                     [-]
        azimuth: tilt of ellipse relative to horizontal [radians]
        ellipticity: ratio of minor to major axes       [-]
    Returns:
        normalized stokes vector with specified properties
    """
    omega = np.arctan(ellipticity)
    cw = np.cos(2*omega)
    sw = np.sin(2*omega)
    ca = np.cos(2*azimuth)
    sa = np.sin(2*azimuth)
    unpolarized = np.array([1-DOP, 0, 0, 0])
    polarized = DOP * np.array([1, cw*ca, cw*sa, sw])
    return unpolarized + polarized


def intensity(S):
    """Return the intensity."""
    return S[0]


def degree_of_polarization(S):
    """Return the degree of polarization."""
    return S[0]/np.sqrt(S[1]**2+S[2]**2+S[3]**2)


def ellipse_orientation(S):
    """
    Return the angle between the major semi-axis and the x-axis.

    The polarization ellipse is rotated by an angle from the
    laboratory frame.  This is that angle: often represented by psi.
    """
    return 1/2 * np.arctan2(S[2], S[1])


def ellipse_ellipticity(S):
    """
    Return the ellipticity of the polarization ellipse.

    This parameter is often represented by Chi.
    """
    return 1/2 * np.arcsin(S[3]/S[0])


def ellipse_axes(S):
    """Returns the semi-major and semi-minor axes of the polarization ellipse."""
    absL = np.sqrt(S[1]**2 + S[2]**2)
    A = np.sqrt((S[0] + absL)/2)
    B = np.sqrt((S[0] - absL)/2)
    return A, B


def stokes_to_jones(S):
    """
    Convert a Stokes vector to a Jones vector.

    The Jones vector can only represent the part of the Stokes vector that is
    polarized.  This fraction is calculated and represented as a Jones vector
    with its horizontal component represented as a real number.

    The sign convention for the Jones vector can be set by calling
    `pypolar.jones.use_alternate_convention(True)`.  The default is to assume that
    the field is represented by exp(j*omega*t-k*z).

    Inputs:
        S : a Stokes vector

    Returns:
         the Jones vector for
    """
    if S[0] == 0:
        return np.array([0, 0])

    # Fraction of intensity that is polarized
    Ip = np.sqrt(S[1]**2 + S[2]**2 + S[3]**2)

    # Normalize the remaining Stokes parameters to this fraction
    Q = S[1] / Ip
    U = S[2] / Ip
    V = S[3] / Ip

    # Amplitude of the polarized field
    E_0 = np.sqrt(Ip)

    # vertically polarized light has no E_x field
    if Q == -1:
        return np.array([0, E_0])

    # Assemble the Jones vector
    A = np.sqrt((1 + Q) / 2)
    J = E_0 * np.array([A, complex(U, V) / (2 * A)])

    if pypolar.jones.alternate_sign_convention:
        return np.conjugate(J)

    return J


def mueller_to_jones(M):
    """
    Convert a Mueller matrix to a Jones matrix.

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
    """
    Interpret a Stokes vector.

    Parameters
    S    : A Stokes vector

    Examples
    -------
    interpret([1, 0, 0, 0]) --> "Unpolarized Light"
    """
    try:
        S0, S1, S2, S3 = S
    except:
        print("Stokes vector must have four real elements")
        return 0

#    eps = 1e-12
    print("I = %.3f" % S0)
    print("Q = %.3f" % S1)
    print("U = %.3f" % S2)
    print("V = %.3f" % S3)

    s = "not implemented yet"
    return s
