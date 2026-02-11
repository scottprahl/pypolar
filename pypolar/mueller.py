"""
Useful basic routines for managing polarization with the Stokes / Mueller calculus.

The routines are broken up into four groups: (1) creating Stokes vectors, (2)
creating Mueller matrix operators, (3) interpretation, and (4) conversion.

Functions to create Stokes vectors::

    * stokes_components(I, Q, U, V)
    * stokes_linear(angle)
    * stokes_left_circular()
    * stokes_right_circular()
    * stokes_horizontal()
    * stokes_vertical()
    * stokes_unpolarized()
    * stokes_elliptical(DOP, azimuth, ellipticity)
    * stokes_ellipsometry(tanpsi, Delta)

Functions to create Mueller matrix operators::

    * op_linear_polarizer(angle)
    * op_retarder(fast_axis_angle, phase_delay)
    * op_attenuator(transmittance)
    * op_neutral_density(transmittance) [alias]
    * op_neutral_density_filter(transmittance) [alias]
    * op_mirror()
    * op_rotation(angle)
    * op_quarter_wave_plate(fast_axis_angle)
    * op_half_wave_plate(fast_axis_angle)
    * op_fresnel_reflection(index_of_refraction, incidence_angle)
    * op_fresnel_transmission(index_of_refraction, incidence_angle)

Functions to interpret Stokes vectors::

    * intensity(stokes_vector)
    * degree_of_polarization(stokes_vector)
    * ellipse_orientation(stokes_vector)
    * ellipse_ellipticity(stokes_vector)
    * ellipse_axes(stokes_vector)
    * interpret(stokes_vector)

Functions to convert::

    * stokes_to_jones(stokes_vector)
    * mueller_to_jones(mueller_matrix)
"""

import numpy as np
import pypolar.jones
import pypolar.fresnel

__all__ = (
    "op_linear_polarizer",
    "op_retarder",
    "op_attenuator",
    "op_neutral_density",
    "op_neutral_density_filter",
    "op_mirror",
    "op_rotation",
    "op_quarter_wave_plate",
    "op_half_wave_plate",
    "op_fresnel_reflection",
    "op_fresnel_transmission",
    "stokes_components",
    "stokes_linear",
    "stokes_left_circular",
    "stokes_right_circular",
    "stokes_horizontal",
    "stokes_vertical",
    "stokes_unpolarized",
    "stokes_ellipsometry",
    "stokes_elliptical",
    "intensity",
    "degree_of_polarization",
    "ellipse_orientation",
    "ellipse_ellipticity",
    "ellipse_axes",
    "stokes_to_jones",
    "mueller_to_jones",
    "interpret",
)


def op_linear_polarizer(theta):
    """
    Mueller matrix operator for a rotated linear polarizer.

    The polarizer is rotated around a normal to its surface.

    Args:
        theta: rotation angle measured from the horizontal plane [radians]
    """
    C2 = np.cos(2 * theta)
    S2 = np.sin(2 * theta)
    lp = np.array([[1, C2, S2, 0], [C2, C2**2, C2 * S2, 0], [S2, C2 * S2, S2 * S2, 0], [0, 0, 0, 0]])
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
    ret = np.array(
        [
            [1, 0, 0, 0],
            [0, C2**2 + C * S2**2, (1 - C) * S2 * C2, -S * S2],
            [0, (1 - C) * C2 * S2, S2**2 + C * C2**2, S * C2],
            [0, S * S2, -S * C2, C],
        ]
    )
    return ret


def op_attenuator(t):
    """
    Mueller matrix operator for an optical attenuator.

    Args:
        t : fraction of light getting through attenuator [---]
    """
    att = np.array([[t, 0, 0, 0], [0, t, 0, 0], [0, 0, t, 0], [0, 0, 0, t]])
    return att


def op_neutral_density(t):
    """Alias for :func:`op_attenuator`."""
    return op_attenuator(t)


def op_neutral_density_filter(t):
    """
    Backward-compatible alias for :func:`op_neutral_density`.

    Args:
        t: fraction of light getting through attenuator [---]
    """
    return op_neutral_density(t)


def op_mirror():
    """Mueller matrix operator for a perfect mirror."""
    mir = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
    return mir


def op_rotation(theta):
    """
    Mueller matrix operator to rotate light around the optical axis.

    Args:
        theta: rotation angle  [radians]
    """
    C2 = np.cos(2 * theta)
    S2 = np.sin(2 * theta)
    rot = np.array([[1, 0, 0, 0], [0, C2, S2, 0], [0, -S2, C2, 0], [0, 0, 0, 1]])
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
    qwp = np.array([[1, 0, 0, 0], [0, C2**2, C2 * S2, -S2], [0, C2 * S2, S2 * S2, C2], [0, S2, -C2, 0]])
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
    qwp = np.array(
        [[1, 0, 0, 0], [0, C2**2 - S2**2, 2 * C2 * S2, 0], [0, 2 * C2 * S2, S2 * S2 - C2**2, 0], [0, 0, 0, -1]]
    )
    return qwp


def op_fresnel_reflection(m, theta, n_i=1):
    """
    Mueller matrix operator for Fresnel reflection at angle theta.

    Convert from the Jones operator to ensure that phase changes are handled
    correctly and that results remain consistent with the field-amplitude
    Fresnel model.

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
        n_i:    real refractive index of incident medium [-]

    Returns:
        4x4 Fresnel reflection Mueller matrix       [-]
    """
    J = pypolar.jones.op_fresnel_reflection(m, theta, n_i=n_i)
    R = pypolar.jones.jones_op_to_mueller_op(J)
    return R


def op_fresnel_transmission(m, theta, n_i=1):
    """
    Mueller matrix operator for Fresnel transmission at angle theta.

    Convert from the Jones operator so the Mueller operator remains consistent
    with field-amplitude Fresnel transmission (including phase effects).

    Args:
        m :     complex index of refraction       [-]
        theta : angle from normal to surface      [radians]
        n_i:    real refractive index of incident medium [-]

    Returns:
        4x4 Fresnel transmission operator         [-]
    """
    J = pypolar.jones.op_fresnel_transmission(m, theta, n_i=n_i)
    T = pypolar.jones.jones_op_to_mueller_op(J)
    return T


def stokes_linear(theta):
    """Stokes vector for light polarized at angle theta from the horizontal plane."""
    if np.isscalar(theta):
        return np.array([1, np.cos(2 * theta), np.sin(2 * theta), 0])
    return np.array([np.ones_like(theta), np.cos(2 * theta), np.sin(2 * theta), np.zeros_like(theta)]).T


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


def stokes_components(I, Q, U, V):
    """
    Stokes vector from explicit components.

    Args:
        I: Total intensity component.
        Q: Horizontal/vertical linear polarization component.
        U: ±45 degree linear polarization component.
        V: Circular polarization component.

    Returns:
        Stokes vector with shape `(4,)` for scalar inputs, or stacked array with
        shape `(..., 4)` for broadcast-compatible array inputs.
    """
    if np.isscalar(I) and np.isscalar(Q) and np.isscalar(U) and np.isscalar(V):
        return np.array([I, Q, U, V])
    i, q, u, v = np.broadcast_arrays(I, Q, U, V)
    return np.stack((i, q, u, v), axis=-1)


def stokes_ellipsometry(tanpsi, Delta):
    """
    Stokes vector using ellipsometer parameters.

    This creates a Stokes vector for the specific set of ellipsometry
    parameters tanpsi and Delta.  See Fujiwara table 3.1 for example.

    Args:
        tanpsi: abs(E_x / E_y)             [-]
        Delta: angle(E_x) - angle(E_y)   [radians]

    Returns:
        normalized Stokes vector with specified properties
    """
    psi = np.arctan(tanpsi)
    cp = np.cos(2 * psi)
    sp = np.sin(2 * psi)
    cd = np.cos(2 * Delta)
    sd = np.sin(2 * Delta)
    if np.isscalar(tanpsi) and np.isscalar(Delta):
        return np.array([1, -cp, sp * cd, -sp * sd])

    return np.array([np.ones_like(tanpsi), -cp, sp * cd, -sp * sd]).T


def stokes_elliptical(DOP, azimuth, ellipticity):
    """
    Stokes vector for partially polarized elliptically polarized light.

    Args:
        DOP: degree of polarization                     [-]
        azimuth: tilt of ellipse relative to horizontal [radians]
        ellipticity: ratio of minor to major axes       [-]

    Returns:
        normalized Stokes vector with specified properties
    """
    omega = np.arctan(ellipticity)
    cw = np.cos(2 * omega)
    sw = np.sin(2 * omega)
    ca = np.cos(2 * azimuth)
    sa = np.sin(2 * azimuth)
    if np.isscalar(DOP) and np.isscalar(azimuth) and np.isscalar(ellipticity):
        unpolarized = np.array([1 - DOP, 0, 0, 0])
        polarized = DOP * np.array([1, cw * ca, cw * sa, sw])
        return unpolarized + polarized

    dop, cw, sw, ca, sa = np.broadcast_arrays(DOP, cw, sw, ca, sa)
    unpolarized = np.array([np.ones_like(dop) - dop, np.zeros_like(dop), np.zeros_like(dop), np.zeros_like(dop)])
    polarized = dop * np.array([np.ones_like(dop), cw * ca, cw * sa, sw])
    return (unpolarized + polarized).T


def intensity(S):
    """Return the intensity."""
    if S.ndim == 1:
        return S[0]
    return S[..., 0]


def _degree_of_polarization(S):
    """Return the degree of polarization."""
    if S[0] == 0:
        return 0
    return np.sqrt(S[1] ** 2 + S[2] ** 2 + S[3] ** 2) / S[0]


def degree_of_polarization(S):
    """Return the degree of polarization."""
    if S.ndim == 1:
        return _degree_of_polarization(S)

    n, _ = S.shape
    dop = np.zeros(n)
    for i, SS in enumerate(S):
        dop[i] = _degree_of_polarization(SS)
    return dop


def ellipse_orientation(S):
    """
    Return the angle between the major semi-axis and the x-axis.

    The polarization ellipse is rotated by an angle from the
    laboratory frame.  This is that angle: often represented by psi.
    """
    return 1 / 2 * np.arctan2(S[..., 2], S[..., 1])


def ellipse_ellipticity(S):
    """
    Return the ellipticity of the polarization ellipse.

    This parameter is often represented by Chi.
    """
    return 1 / 2 * np.arcsin(S[..., 3] / S[..., 0])


def ellipse_axes(S):
    """Return the semi-major and semi-minor axes of the polarization ellipse."""
    absL = np.sqrt(S[..., 1] ** 2 + S[..., 2] ** 2)
    A = np.sqrt((S[..., 0] + absL) / 2)
    B = np.sqrt((S[..., 0] - absL) / 2)
    return A, B


def _stokes_to_jones(S):
    """
    Convert a Stokes vector to a Jones vector.

    The Jones vector can only represent the part of the Stokes vector that is
    polarized.  This fraction is calculated and represented as a Jones vector
    with its horizontal component represented as a real number.

    The sign convention for the Jones vector can be set by calling
    `pypolar.jones.use_alternate_convention(True)`.  The default is to assume that
    the field is represented by exp(j * omega * t-k * z).

    Args:
        S: Stokes vector.

    Returns:
        Jones vector representing the polarized component of `S`.
    """
    if S[0] == 0:
        return np.array([0, 0])

    # Fraction of intensity that is polarized
    Ip = np.sqrt(S[1] ** 2 + S[2] ** 2 + S[3] ** 2)

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


def stokes_to_jones(S):
    """
    Convert a (list of) Stokes vector(s) to a (list of) Jones vector(s).

    The sign convention for the Jones vector can be set by calling
    `pypolar.jones.use_alternate_convention(True)`.  The default is to assume that
    the field is represented by exp(j * omega * t-k * z).

    Args:
        S: Single Stokes vector with shape `(4,)` or array with shape `(n, 4)`.

    Returns:
        Jones vector with shape `(2,)`, array of Jones vectors with shape `(n, 2)`,
        or `None` for invalid array shape.
    """
    if S.ndim == 1:
        return _stokes_to_jones(S)

    n, m = S.shape
    if m != 4:
        print("Wrong shape ... should be %dx4 not %dx%d" % (m, n, m))
        return None

    J = np.empty(shape=(n, 2), dtype=np.ndarray)
    for i, SS in enumerate(S):
        J[i] = _stokes_to_jones(SS)

    return J


def mueller_to_jones(M):
    """
    Convert a Mueller matrix to a Jones matrix.

    Theocaris, Matrix Theory of Photoelasticity, eqns 4.70-4.76, 1979

    Args:
        M: 4x4 Mueller matrix.

    Returns:
        Corresponding 2x2 Jones matrix.
    """
    A = np.empty((2, 2))
    A[0, 0] = np.sqrt((M[0, 0] + M[0, 1] + M[1, 0] + M[1, 1]) / 2)
    A[0, 1] = np.sqrt((M[0, 0] + M[0, 1] - M[1, 0] - M[1, 1]) / 2)
    A[1, 0] = np.sqrt((M[0, 0] - M[0, 1] + M[1, 0] - M[1, 1]) / 2)
    A[1, 1] = np.sqrt((M[0, 0] - M[0, 1] - M[1, 0] + M[1, 1]) / 2)

    theta = np.empty((2, 2))
    theta[0, 0] = 0
    theta[0, 1] = -np.arctan2(M[0, 3] + M[1, 3], M[0, 2] + M[1, 2])
    theta[1, 0] = np.arctan2(M[3, 0] + M[3, 1], M[2, 0] + M[2, 1])
    theta[1, 1] = np.arctan2(M[3, 2] - M[2, 3], M[2, 2] + M[3, 3])

    return A * np.exp(1j * theta)


def _to_real_array(x, eps=1e-12):
    """Convert input to a finite real numpy array or return an error message."""
    arr = np.asarray(x)

    if np.iscomplexobj(arr):
        imag_max = np.max(np.abs(arr.imag))
        if imag_max > eps:
            return None, "input has non-zero imaginary parts; Mueller/Stokes quantities must be real"
        arr = arr.real

    try:
        arr = np.asarray(arr, dtype=float)
    except (TypeError, ValueError):
        return None, "input elements must be numeric"

    if not np.all(np.isfinite(arr)):
        return None, "input contains NaN or infinite values"

    return arr, None


def _interpret_mueller_matrix(M):
    """Interpret a 4x4 Mueller matrix with basic physical-admissibility checks."""
    eps = 1e-12
    MM, error = _to_real_array(M, eps=eps)
    if error is not None:
        message = "Malformed input: %s" % error
        print(message)
        return message

    if MM.shape != (4, 4):
        message = (
            "Malformed input: expected a Stokes vector with shape (4,) "
            "or a Mueller matrix with shape (4, 4), got shape %s" % (MM.shape,)
        )
        print(message)
        return message

    lines = ["Detected 4x4 Mueller matrix input.", "Basic physical-admissibility checks:"]
    warnings = []

    M00 = MM[0, 0]
    if M00 < -eps:
        warnings.append("M[0,0] is negative (negative output intensity for unpolarized input)")

    if abs(M00) <= eps:
        if np.linalg.norm(MM) > eps:
            warnings.append("M[0,0] is zero while other entries are non-zero")
        D = np.inf
        P = np.inf
    else:
        D = np.linalg.norm(MM[0, 1:]) / abs(M00)
        P = np.linalg.norm(MM[1:, 0]) / abs(M00)
        if D > 1 + 1e-9:
            warnings.append("Diattenuation > 1 (||row0[1:]|| / |M00| = %.3f)" % D)
        if P > 1 + 1e-9:
            warnings.append("Polarizance > 1 (||col0[1:]|| / |M00| = %.3f)" % P)

    test_states = {
        "unpolarized": np.array([1.0, 0.0, 0.0, 0.0]),
        "+Q": np.array([1.0, 1.0, 0.0, 0.0]),
        "-Q": np.array([1.0, -1.0, 0.0, 0.0]),
        "+U": np.array([1.0, 0.0, 1.0, 0.0]),
        "-U": np.array([1.0, 0.0, -1.0, 0.0]),
        "+V": np.array([1.0, 0.0, 0.0, 1.0]),
        "-V": np.array([1.0, 0.0, 0.0, -1.0]),
    }
    violating_states = []
    for name, Sin in test_states.items():
        Sout = MM @ Sin
        Iout = Sout[0]
        pnorm = np.linalg.norm(Sout[1:])
        if Iout < -eps or pnorm > Iout + 1e-9:
            violating_states.append(name)

    if violating_states:
        warnings.append("maps valid test states to unphysical outputs (%s)" % ", ".join(violating_states))

    if warnings:
        lines.append("WARNING: matrix is not physically admissible under these checks:")
        for warning in warnings:
            lines.append("  - %s" % warning)
    else:
        lines.append("No violations found in necessary checks.")
        lines.append("Note: this is not a complete physical-realizability proof.")

    return "\n".join(lines)


def interpret(S):
    """
    Interpret a Stokes vector.

    If a 4x4 array is passed, run basic Mueller-matrix admissibility checks.

    Args:
        S: Stokes vector with shape `(4,)` or Mueller matrix with shape `(4, 4)`.

    Examples:
        interpret([1, 0, 0, 0]) -> "Unpolarized light"

    Returns:
        Human-readable interpretation string.
    """
    arr, error = _to_real_array(S)
    if error is not None:
        message = "Malformed input: %s" % error
        return message

    if arr.shape == (4, 4):
        return _interpret_mueller_matrix(arr)

    if arr.shape != (4,):
        message = (
            "Malformed input: expected a Stokes vector with shape (4,) "
            "or a Mueller matrix with shape (4, 4), got shape %s" % (arr.shape,)
        )
        return message

    S0, S1, S2, S3 = arr
    pnorm = np.sqrt(S1**2 + S2**2 + S3**2)
    if S0 < -1e-12:
        message = (
            "Physically impossible Stokes vector: intensity I must be >= 0, got I = %.6g" % S0
        )
        return message
    if abs(S0) <= 1e-12 and pnorm > 1e-12:
        message = (
            "Physically impossible Stokes vector: non-zero polarization components with zero intensity"
        )
        return message
    if pnorm > S0 + 1e-9:
        message = (
            "Physically impossible Stokes vector: sqrt(Q^2+U^2+V^2) = %.6g exceeds I = %.6g" % (pnorm, S0)
        )
        return message

    dop = _degree_of_polarization(np.array([S0, S1, S2, S3]))
    dop = np.clip(dop, 0, 1)
    eps = 1e-12

    lines = [
        "I = %.3f" % S0,
        "Q = %.3f" % S1,
        "U = %.3f" % S2,
        "V = %.3f" % S3,
        "Degree of polarization = %.3f" % dop,
    ]

    if abs(S0) < eps:
        lines.append("No light (zero intensity)")
    elif dop < eps:
        lines.append("Unpolarized light")
    else:
        polarized_intensity = S0 * dop
        unpolarized_intensity = S0 - polarized_intensity
        if dop < 1 - eps:
            lines.append(
                "Partially polarized: polarized intensity = %.3f, unpolarized intensity = %.3f"
                % (polarized_intensity, unpolarized_intensity)
            )
        else:
            lines.append("Fully polarized light")

        # describe the polarized component using normalized Stokes parameters
        s1 = np.clip(S1 / (S0 * dop), -1, 1)
        s2 = np.clip(S2 / (S0 * dop), -1, 1)
        s3 = np.clip(S3 / (S0 * dop), -1, 1)

        psi = 0.5 * np.arctan2(s2, s1)
        chi = 0.5 * np.arcsin(s3)
        psi_deg = np.degrees(psi)
        chi_deg = np.degrees(chi)
        ell = np.tan(chi)

        if abs(abs(chi) - np.pi / 4) < 1e-6:
            if chi >= 0:
                lines.append("Right circular polarization")
            else:
                lines.append("Left circular polarization")
        elif abs(chi) < 1e-6:
            lines.append("Linear polarization at %.1f degrees CCW from x-axis" % psi_deg)
        else:
            if chi >= 0:
                lines.append("Right elliptical polarization")
            else:
                lines.append("Left elliptical polarization")
            lines.append("    azimuth = %.1f degrees CCW from x-axis" % psi_deg)
            lines.append("    ellipticity angle = %.1f degrees" % chi_deg)
            lines.append("    ellipticity (b/a) = %.3f" % ell)

    return "\n".join(lines)
