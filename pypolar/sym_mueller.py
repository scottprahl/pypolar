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
    op_attenuator(transmittance)
    op_neutral_density(transmittance) [alias]
    op_neutral_density_filter(transmittance) [alias]
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
    interpret(stokes_vector)

Functions to convert::

    stokes_to_jones(stokes_vector)
    mueller_to_jones(mueller_matrix)
"""

import sympy
from pypolar import sym_fresnel


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
    "interpret",
    "stokes_to_jones",
    "mueller_to_jones",
)


def op_linear_polarizer(theta):
    """
    Mueller matrix operator for a rotated linear polarizer.

    The polarizer is rotated about a normal to its surface.

    Args:
        theta: rotation angle measured from the horizontal plane [radians]
    """
    C2 = sympy.cos(2 * theta)
    S2 = sympy.sin(2 * theta)
    lp = sympy.Matrix([[1, C2, S2, 0], [C2, C2**2, C2 * S2, 0], [S2, C2 * S2, S2 * S2, 0], [0, 0, 0, 0]])
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
    ret = sympy.Matrix(
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
        t: fraction of light getting through attenuator  [---]
    """
    att = sympy.Matrix([[t, 0, 0, 0], [0, t, 0, 0], [0, 0, t, 0], [0, 0, 0, t]])
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
    mir = sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
    return mir


def op_rotation(theta):
    """
    Mueller matrix operator to rotate light about the optical axis.

    Args:
        theta: rotation angle  [radians]
    """
    C2 = sympy.cos(2 * theta)
    S2 = sympy.sin(2 * theta)
    rot = sympy.Matrix([[1, 0, 0, 0], [0, C2, S2, 0], [0, -S2, C2, 0], [0, 0, 0, 1]])
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
    qwp = sympy.Matrix([[1, 0, 0, 0], [0, C2**2, C2 * S2, -S2], [0, C2 * S2, S2 * S2, C2], [0, S2, -C2, 0]])
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
    qwp = sympy.Matrix(
        [[1, 0, 0, 0], [0, C2**2 - S2**2, 2 * C2 * S2, 0], [0, 2 * C2 * S2, S2 * S2 - C2**2, 0], [0, 0, 0, -1]]
    )
    return qwp


def op_fresnel_reflection(m, theta, n_i=1):
    """
    Mueller matrix operator for Fresnel reflection.

    Build the Mueller operator from complex field-amplitude reflection
    coefficients.  This preserves phase-delay effects while ensuring real
    Mueller matrix entries.

    Args:
        m :     complex index of refraction   [-]
        theta : angle from normal to surface  [radians]
        n_i:    real refractive index of incident medium [-]

    Returns:
        4x4 Fresnel reflection operator       [-]
    """
    m_rel = m / n_i
    r_par = sym_fresnel.r_par_amplitude(m_rel, theta)
    r_per = sym_fresnel.r_per_amplitude(m_rel, theta)

    rr_par = sympy.Abs(r_par) ** 2
    rr_per = sympy.Abs(r_per) ** 2
    cross = r_par * sympy.conjugate(r_per)

    a = sympy.Rational(1, 2) * (rr_par + rr_per)
    b = sympy.Rational(1, 2) * (rr_par - rr_per)
    c = sympy.re(cross)
    d = sympy.im(cross)

    return sympy.Matrix([[a, b, 0, 0], [b, a, 0, 0], [0, 0, c, d], [0, 0, -d, c]])


def op_fresnel_transmission(m, theta, n_i=1):
    """
    Mueller matrix operator for Fresnel transmission.

    Build the Mueller operator from the complex field-amplitude transmission
    coefficients.  This preserves phase-delay effects while ensuring the
    Mueller matrix entries are real-valued.

    Args:
        m :     complex index of refraction       [-]
        theta : angle from normal to surface      [radians]
        n_i:    real refractive index of incident medium [-]

    Returns:
        4x4 Fresnel transmission operator         [-]
    """
    m_rel = m / n_i
    t_par = sym_fresnel.t_par_amplitude(m_rel, theta)
    t_per = sym_fresnel.t_per_amplitude(m_rel, theta)

    tt_par = sympy.Abs(t_par) ** 2
    tt_per = sympy.Abs(t_per) ** 2
    cross = t_par * sympy.conjugate(t_per)

    a = sympy.Rational(1, 2) * (tt_par + tt_per)
    b = sympy.Rational(1, 2) * (tt_par - tt_per)
    c = sympy.re(cross)
    d = sympy.im(cross)

    return sympy.Matrix([[a, b, 0, 0], [b, a, 0, 0], [0, 0, c, d], [0, 0, -d, c]])


def stokes_linear(theta):
    """
    Stokes vector for linear polarized light at angle.

    Args:
        m :     complex index of refraction       [-]
        theta : angle from horizontal plane      [radians]
    """
    return sympy.Matrix([1, sympy.cos(2 * theta), sympy.sin(2 * theta), 0])


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


def stokes_ellipsometry(tanpsi, Delta):
    """
    Stokes vector from ellipsometer parameters.

    Args:
        tanpsi: abs(E_x / E_y)                    [-]
        Delta: angle(E_x) - angle(E_y)        [radians]
    """
    psi = sympy.atan(tanpsi)
    c2p = sympy.cos(2 * psi)
    s2p = sympy.sin(2 * psi)
    return sympy.Matrix([1, -c2p, s2p * sympy.cos(Delta), -s2p * sympy.sin(Delta)])


def stokes_elliptical(DOP, azimuth, ellipticity):
    """
    Stokes vector from polarization ellipse parameters.

    Args:
        DOP: degree of polarization                   [-]
        azimuth: orientation of ellipse major-axis [radians]
        ellipticity: arctan(minor/major)           [radians]
    """
    ce = sympy.cos(2 * ellipticity)
    se = sympy.sin(2 * ellipticity)
    ca = sympy.cos(2 * azimuth)
    sa = sympy.sin(2 * azimuth)
    return sympy.Matrix([1, DOP * ce * ca, DOP * ce * sa, DOP * se])


def intensity(S):
    """Return the intensity."""
    return S[0]


def degree_of_polarization(S):
    """Return the degree of polarization."""
    return sympy.Piecewise((0, sympy.Eq(S[0], 0)), (sympy.sqrt(S[1] ** 2 + S[2] ** 2 + S[3] ** 2) / S[0], True))


def ellipse_orientation(S):
    """
    Return orientation of the polarization ellipse.

    The orientation is the angle between the major semi-axis and the x-axis of
    the polarization ellipse (often represented by psi).
    """
    return 1 / 2 * sympy.atan2(S[2], S[1])


def ellipse_ellipticity(S):
    """
    Return ellipticity of the polarization ellipse.

    The ellipticity of the polarization ellipse (often
    represented by chi)
    """
    return 1 / 2 * sympy.asin(S[3] / S[0])


def ellipse_axes(S):
    """
    Return the semi-major and semi-minor axes.

    These are the axes of the polarization ellipse.
    """
    absL = sympy.sqrt(S[1] ** 2 + S[2] ** 2)
    A = sympy.sqrt((S[0] + absL) / 2)
    B = sympy.sqrt((S[0] - absL) / 2)
    return A, B


def _definitely_true(expr):
    """Return True only if SymPy can prove that `expr` is true."""
    simplified = sympy.simplify(expr)
    return simplified is True or simplified == sympy.S.true


def _to_real_sympy_matrix(x):
    """Convert input to a finite real SymPy matrix or return an error string."""
    try:
        M = sympy.Matrix(x)
    except (TypeError, ValueError):
        return None, "Malformed input: elements must be convertible to a matrix of numbers"

    for value in M:
        if value.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo):
            return None, "Malformed input: matrix contains NaN or infinite values"
        if value.is_real is False:
            return None, "Malformed input: Stokes and Mueller quantities must be real"

    return M, None


def _interpret_mueller_matrix(M):
    """Interpret a 4x4 symbolic Mueller matrix with basic admissibility diagnostics."""
    lines = ["Detected 4x4 Mueller matrix input.", "Basic physical-admissibility checks (symbolic):"]
    warnings = []

    M00 = sympy.simplify(M[0, 0])
    if _definitely_true(M00 < 0):
        warnings.append("M[0,0] is negative (negative output intensity for unpolarized input)")

    lines.append(f"M[0,0] = {M00}")

    if _definitely_true(sympy.Eq(M00, 0)):
        D = sympy.oo
        P = sympy.oo
        if _definitely_true(sympy.simplify(M.norm()) > 0):
            warnings.append("M[0,0] is zero while other entries are non-zero")
    else:
        D = sympy.simplify(sympy.sqrt(M[0, 1] ** 2 + M[0, 2] ** 2 + M[0, 3] ** 2) / sympy.Abs(M00))
        P = sympy.simplify(sympy.sqrt(M[1, 0] ** 2 + M[2, 0] ** 2 + M[3, 0] ** 2) / sympy.Abs(M00))
        if _definitely_true(D > 1):
            warnings.append(f"Diattenuation > 1 ({D})")
        if _definitely_true(P > 1):
            warnings.append(f"Polarizance > 1 ({P})")

    lines.append(f"Diattenuation = {sympy.simplify(D)}")
    lines.append(f"Polarizance = {sympy.simplify(P)}")

    if warnings:
        lines.append("WARNING: matrix is not physically admissible under these checks:")
        for warning in warnings:
            lines.append(f"  - {warning}")
    else:
        lines.append("No violations found in necessary symbolic checks.")
        lines.append("Note: this is not a complete physical-realizability proof.")

    return "\n".join(lines)


def interpret(S):
    """
    Interpret a symbolic Stokes vector.

    If a 4x4 matrix is passed, report symbolic Mueller-matrix admissibility checks.

    Args:
        S: Stokes vector with shape `(4, 1)` or `(1, 4)`, or Mueller matrix with
            shape `(4, 4)`.

    Returns:
        Human-readable interpretation string.
    """
    M, error = _to_real_sympy_matrix(S)
    if error is not None:
        print(error)
        return error

    if M.shape == (4, 4):
        summary = _interpret_mueller_matrix(M)
        print(summary)
        return summary

    if M.shape == (1, 4):
        V = M.T
    elif M.shape == (4, 1):
        V = M
    else:
        message = (
            "Malformed input: expected Stokes vector shape (4,1) or (1,4), "
            f"or Mueller matrix shape (4,4), got shape {M.shape}"
        )
        print(message)
        return message

    S0, S1, S2, S3 = [sympy.simplify(V[i, 0]) for i in range(4)]
    pnorm2 = sympy.simplify(S1**2 + S2**2 + S3**2)

    if _definitely_true(S0 < 0):
        message = f"Physically impossible Stokes vector: intensity I must be >= 0, got I = {S0}"
        print(message)
        return message

    if _definitely_true(sympy.Eq(S0, 0)) and _definitely_true(pnorm2 > 0):
        message = "Physically impossible Stokes vector: non-zero polarization components with zero intensity"
        print(message)
        return message

    if _definitely_true(pnorm2 > S0**2):
        message = (
            "Physically impossible Stokes vector: sqrt(Q^2+U^2+V^2) exceeds I "
            f"(sqrt(Q^2+U^2+V^2)^2 = {pnorm2}, I^2 = {sympy.simplify(S0**2)})"
        )
        print(message)
        return message

    if _definitely_true(sympy.Eq(S0, 0)):
        dop = sympy.Integer(0)
    else:
        dop = sympy.simplify(sympy.sqrt(pnorm2) / S0)

    lines = [
        f"I = {S0}",
        f"Q = {S1}",
        f"U = {S2}",
        f"V = {S3}",
        f"Degree of polarization = {dop}",
    ]

    if _definitely_true(sympy.Eq(S0, 0)):
        lines.append("No light (zero intensity)")
    elif _definitely_true(sympy.Eq(pnorm2, 0)):
        lines.append("Unpolarized light")
    else:
        if _definitely_true(sympy.Eq(pnorm2, S0**2)):
            lines.append("Fully polarized light")
        elif _definitely_true(pnorm2 < S0**2):
            lines.append("Partially polarized light")
        else:
            lines.append("Polarization type is symbolically undetermined")

        psi = sympy.simplify(ellipse_orientation(V))
        chi = sympy.simplify(ellipse_ellipticity(V))
        lines.append(f"Ellipse orientation = {psi}")
        lines.append(f"Ellipticity angle = {chi}")

    summary = "\n".join(lines)
    print(summary)
    return summary


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
    if _definitely_true(sympy.Eq(S[0], 0)):
        return sympy.Matrix([0, 0])

    # Calculate the degree of polarization
    p = sympy.sqrt(S[1] ** 2 + S[2] ** 2 + S[3] ** 2) / S[0]

    # Normalize the Stokes parameters (first one will be 1, of course)
    Q = S[1] / (S[0] * p)
    U = S[2] / (S[0] * p)
    V = S[3] / (S[0] * p)

    # the Jones components
    A = sympy.sqrt((1 + Q) / 2)
    if A == 0:
        B = 1
    else:
        B = (U - sympy.I * V) / (2 * A)

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
    A = sympy.Matrix.zeros(2, 2)
    A[0, 0] = sympy.sqrt((M[0, 0] + M[0, 1] + M[1, 0] + M[1, 1]) / 2)
    A[0, 1] = sympy.sqrt((M[0, 0] + M[0, 1] - M[1, 0] - M[1, 1]) / 2)
    A[1, 0] = sympy.sqrt((M[0, 0] - M[0, 1] + M[1, 0] - M[1, 1]) / 2)
    A[1, 1] = sympy.sqrt((M[0, 0] - M[0, 1] - M[1, 0] + M[1, 1]) / 2)

    theta = sympy.Matrix.zeros(2, 2)
    theta[0, 0] = 0
    theta[0, 1] = -sympy.atan2(M[0, 3] + M[1, 3], M[0, 2] + M[1, 2])
    theta[1, 0] = sympy.atan2(M[3, 0] + M[3, 1], M[2, 0] + M[2, 1])
    theta[1, 1] = sympy.atan2(M[3, 2] - M[2, 3], M[2, 2] + M[3, 3])

    phase = theta.applyfunc(lambda x: sympy.exp(sympy.I * x))
    return A.multiply_elementwise(phase)
