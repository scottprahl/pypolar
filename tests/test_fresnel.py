"""Unit tests for the numeric Fresnel functions in :mod:`pypolar.fresnel`.

These tests check basic angle relations, amplitude and power coefficients,
energy conservation for lossless interfaces, and vectorization behavior.
"""

import unittest
import numpy as np
from pypolar import fresnel

RT_TOL = 1e-12  # tolerance for R + T = 1 tests
ANGLE_TOL = 1e-12  # tolerance for angle comparisons


class TestFresnelBasicAngles(unittest.TestCase):
    """Tests for Brewster and critical angles computed by the Fresnel module."""

    def test_brewster_angle_radians_and_degrees(self):
        """Check Brewster angle in radians and degrees and Rp ≈ 0 at that angle."""
        n_i = 1.0
        m = 1.5  # real dielectric

        # Brewster angle: tan(theta_B) = m / n_i
        expected_rad = np.atan2(m, n_i)
        expected_deg = np.degrees(expected_rad)

        theta_B_rad = fresnel.brewster(m, n_i=n_i, deg=False)
        theta_B_deg = fresnel.brewster(m, n_i=n_i, deg=True)

        self.assertAlmostEqual(theta_B_rad, expected_rad, delta=ANGLE_TOL)
        self.assertAlmostEqual(theta_B_deg, expected_deg, delta=ANGLE_TOL)

        # At Brewster angle, R_par should vanish (for lossless media)
        R_p = fresnel.R_par(m, theta_B_rad, n_i=n_i, deg=False)
        self.assertAlmostEqual(R_p, 0.0, places=12)

    def test_critical_angle_radians_and_degrees(self):
        """Check critical angle in radians and degrees for total internal reflection."""
        # total internal reflection: incident from higher to lower index
        n_i = 1.5
        m = 1.0  # transmitted medium

        # critical angle: sin(theta_c) = m / n_i
        expected_rad = np.asin(m / n_i)
        expected_deg = np.degrees(expected_rad)

        theta_c_rad = fresnel.critical(m, n_i=n_i, deg=False)
        theta_c_deg = fresnel.critical(m, n_i=n_i, deg=True)

        self.assertAlmostEqual(theta_c_rad, expected_rad, delta=ANGLE_TOL)
        self.assertAlmostEqual(theta_c_deg, expected_deg, delta=ANGLE_TOL)


class TestFresnelAmplitudes(unittest.TestCase):
    """Tests for s- and p-polarized Fresnel field amplitudes."""

    def test_normal_incidence_symmetry_real_indices(self):
        """At normal incidence, check analytic rs, rp and equality of ts, tp."""
        n_i = 1.0
        m = 1.5
        theta = 0.0

        rp = fresnel.r_par_amplitude(m, theta, n_i=n_i)
        rs = fresnel.r_per_amplitude(m, theta, n_i=n_i)
        tp = fresnel.t_par_amplitude(m, theta, n_i=n_i)
        ts = fresnel.t_per_amplitude(m, theta, n_i=n_i)

        # Analytic Fresnel values at normal incidence
        r_s_expected = (n_i - m) / (n_i + m)
        r_p_expected = (m - n_i) / (m + n_i)  # = -r_s_expected

        self.assertAlmostEqual(rs, r_s_expected, places=14)
        self.assertAlmostEqual(rp, r_p_expected, places=14)

        # r_p should be the negative of r_s
        self.assertAlmostEqual(rp, -rs, places=14)

        # The transmitted amplitudes are equal at normal incidence
        self.assertAlmostEqual(tp, ts, places=14)

    def test_degrees_vs_radians_consistency(self):
        """Ensure deg=True and deg=False give identical amplitudes."""
        n_i = 1.0
        m = 1.5
        theta_deg = 37.0
        theta_rad = np.radians(theta_deg)

        rp_rad = fresnel.r_par_amplitude(m, theta_rad, n_i=n_i, deg=False)
        rp_deg = fresnel.r_par_amplitude(m, theta_deg, n_i=n_i, deg=True)

        rs_rad = fresnel.r_per_amplitude(m, theta_rad, n_i=n_i, deg=False)
        rs_deg = fresnel.r_per_amplitude(m, theta_deg, n_i=n_i, deg=True)

        self.assertAlmostEqual(rp_rad, rp_deg, places=14)
        self.assertAlmostEqual(rs_rad, rs_deg, places=14)

    def test_complex_index_equal_power_reflectance_at_normal_incidence(self):
        """For complex m at normal incidence, check Rs = Rp and Ts = Tp."""
        n_i = 1.0
        m = 1.5 - 0.2j
        theta = 0.0

        rp = fresnel.r_par_amplitude(m, theta, n_i=n_i)
        rs = fresnel.r_per_amplitude(m, theta, n_i=n_i)

        R_p = abs(rp) ** 2
        R_s = abs(rs) ** 2

        self.assertAlmostEqual(R_p, R_s, places=12)

        T_p = fresnel.T_par(m, theta, n_i=n_i)
        T_s = fresnel.T_per(m, theta, n_i=n_i)
        self.assertAlmostEqual(T_p, T_s, places=12)


class TestFresnelPowerConservation(unittest.TestCase):
    """Tests for power reflectance/transmittance and energy conservation."""

    def test_energy_conservation_lossless_normal_incidence(self):
        """Check Rp+Tp = 1 and Rs+Ts = 1 at normal incidence for real m."""
        n_i = 1.0
        m = 1.5  # real, lossless
        theta = 0.0

        Rp = fresnel.R_par(m, theta, n_i=n_i)
        Rs = fresnel.R_per(m, theta, n_i=n_i)
        Tp = fresnel.T_par(m, theta, n_i=n_i)
        Ts = fresnel.T_per(m, theta, n_i=n_i)

        # s and p should be identical at normal incidence
        self.assertAlmostEqual(Rp, Rs, places=14)
        self.assertAlmostEqual(Tp, Ts, places=14)

        # For lossless media, R + T = 1
        self.assertAlmostEqual(Rp + Tp, 1.0, delta=RT_TOL)
        self.assertAlmostEqual(Rs + Ts, 1.0, delta=RT_TOL)

    def test_energy_conservation_lossless_oblique_incidence(self):
        """Check Rp+Tp = 1 and Rs+Ts = 1 at oblique incidence for real m."""
        n_i = 1.0
        m = 1.5  # real, lossless
        theta = np.radians(45.0)  # below critical, no TIR

        Rp = fresnel.R_par(m, theta, n_i=n_i)
        Rs = fresnel.R_per(m, theta, n_i=n_i)
        Tp = fresnel.T_par(m, theta, n_i=n_i)
        Ts = fresnel.T_per(m, theta, n_i=n_i)

        self.assertAlmostEqual(Rp + Tp, 1.0, delta=RT_TOL)
        self.assertAlmostEqual(Rs + Ts, 1.0, delta=RT_TOL)

    def test_unpolarized_reflection_and_transmission(self):
        """Verify unpolarized R,T are averages of s and p and obey R+T=1."""
        n_i = 1.0
        m = 1.5
        theta = np.radians(30.0)

        Rs = fresnel.R_per(m, theta, n_i=n_i)
        Rp = fresnel.R_par(m, theta, n_i=n_i)
        Ts = fresnel.T_per(m, theta, n_i=n_i)
        Tp = fresnel.T_par(m, theta, n_i=n_i)

        R_unpol = fresnel.R_unpolarized(m, theta, n_i=n_i)
        T_unpol = fresnel.T_unpolarized(m, theta, n_i=n_i)

        # By definition, R_unpol and T_unpol are averages of s and p
        self.assertAlmostEqual(R_unpol, 0.5 * (Rs + Rp), places=14)
        self.assertAlmostEqual(T_unpol, 0.5 * (Ts + Tp), places=14)

        # Energy conservation for unpolarized light as well
        self.assertAlmostEqual(R_unpol + T_unpol, 1.0, delta=RT_TOL)

    def test_brewster_zero_parallel_reflectance(self):
        """Confirm that Rp ≈ 0 at Brewster angle for real dielectric m."""
        n_i = 1.0
        m = 1.5  # real dielectric

        theta_B = fresnel.brewster(m, n_i=n_i, deg=False)
        Rp = fresnel.R_par(m, theta_B, n_i=n_i)

        self.assertAlmostEqual(Rp, 0.0, places=12)

    def test_total_internal_reflection_reflectance_near_one(self):
        """Check Rp and Rs ≈ 1 above the critical angle (TIR)."""
        # Incident from n_i > m, angle above critical
        n_i = 1.5
        m = 1.0
        theta_c = fresnel.critical(m, n_i=n_i, deg=False)
        theta = theta_c + np.radians(5.0)

        Rp = fresnel.R_par(m, theta, n_i=n_i)
        Rs = fresnel.R_per(m, theta, n_i=n_i)

        # Reflection should be ~1 for both polarizations under TIR
        self.assertTrue(Rp > 0.999)
        self.assertTrue(Rs > 0.999)


class TestVectorization(unittest.TestCase):
    """Tests for vectorized handling of angle arrays in power coefficients."""

    def test_array_input_for_angles(self):
        """Verify array inputs work and that R+T≈1 elementwise for real m."""
        n_i = 1.0
        m = 1.5
        theta = np.linspace(0.0, np.radians(80.0), 5)

        Rp = fresnel.R_par(m, theta, n_i=n_i)
        Rs = fresnel.R_per(m, theta, n_i=n_i)
        Tp = fresnel.T_par(m, theta, n_i=n_i)
        Ts = fresnel.T_per(m, theta, n_i=n_i)

        # All should have same shape as input
        self.assertEqual(Rp.shape, theta.shape)
        self.assertEqual(Rs.shape, theta.shape)
        self.assertEqual(Tp.shape, theta.shape)
        self.assertEqual(Ts.shape, theta.shape)

        # R + T should be ~1 elementwise for lossless case
        self.assertTrue(np.allclose(Rp + Tp, 1.0, atol=1e-10))
        self.assertTrue(np.allclose(Rs + Ts, 1.0, atol=1e-10))

    def test_array_input_for_refractive_index(self):
        """Verify array refractive-index inputs work for scalar angle input."""
        n_i = 1.0
        m = np.array([1.3, 1.5 - 0.1j, 2.0])
        theta = np.radians(35.0)

        rp = fresnel.r_par_amplitude(m, theta, n_i=n_i)
        rs = fresnel.r_per_amplitude(m, theta, n_i=n_i)

        self.assertEqual(rp.shape, m.shape)
        self.assertEqual(rs.shape, m.shape)

        for i in range(len(m)):
            self.assertAlmostEqual(rp[i], fresnel.r_par_amplitude(m[i], theta, n_i=n_i), places=12)
            self.assertAlmostEqual(rs[i], fresnel.r_per_amplitude(m[i], theta, n_i=n_i), places=12)

    def test_broadcast_input_for_refractive_index_and_angle(self):
        """Verify broadcast-compatible m and theta arrays return broadcasted outputs."""
        n_i = 1.0
        m = np.array([[1.3], [1.5 - 0.1j]])
        theta = np.radians(np.array([10.0, 30.0, 50.0]))

        Rp = fresnel.R_par(m, theta, n_i=n_i)
        Tp = fresnel.T_par(m, theta, n_i=n_i)

        self.assertEqual(Rp.shape, (2, 3))
        self.assertEqual(Tp.shape, (2, 3))

        for i in range(2):
            self.assertTrue(np.allclose(Rp[i], fresnel.R_par(m[i, 0], theta, n_i=n_i)))
            self.assertTrue(np.allclose(Tp[i], fresnel.T_par(m[i, 0], theta, n_i=n_i)))


class TestFresnelInputHandling(unittest.TestCase):
    """Tests for Fresnel input normalization and validation helpers."""

    def test_positive_imaginary_index_is_conjugated(self):
        """Positive-imaginary refractive indices should map to conjugated convention."""
        n_i = 1.0
        theta = np.radians(40.0)
        m_pos = 1.6 + 0.2j
        m_neg = np.conjugate(m_pos)

        funcs = (
            fresnel.r_par_amplitude,
            fresnel.r_per_amplitude,
            fresnel.t_par_amplitude,
            fresnel.t_per_amplitude,
            fresnel.R_par,
            fresnel.R_per,
            fresnel.T_par,
            fresnel.T_per,
            fresnel.R_unpolarized,
            fresnel.T_unpolarized,
            fresnel.ellipsometry_rho,
        )
        for func in funcs:
            self.assertTrue(np.allclose(func(m_pos, theta, n_i=n_i), func(m_neg, theta, n_i=n_i)))

    def test_out_of_range_angles_raise_value_error(self):
        """Angles outside physical incidence range should raise ValueError."""
        with self.assertRaises(ValueError):
            fresnel.r_par_amplitude(1.5, -0.1)

        with self.assertRaises(ValueError):
            fresnel.r_par_amplitude(1.5, np.pi)

        with self.assertRaises(ValueError):
            fresnel.r_par_amplitude(1.5, -1.0, deg=True)

        with self.assertRaises(ValueError):
            fresnel.r_par_amplitude(1.5, 120.0, deg=True)

        with self.assertRaises(ValueError):
            fresnel.r_par_amplitude(1.5, np.array([0.1, np.pi / 3, np.pi]))

        with self.assertRaises(ValueError):
            fresnel.ellipsometry_index(0.1 + 0.2j, 95.0, deg=True)


class TestFresnelEllipsometryHelpers(unittest.TestCase):
    """Tests for Fresnel ellipsometry helper routines."""

    def test_ellipsometry_rho_matches_reflection_ratio(self):
        """Result should match direct rp/rs ratio in both radian and degree modes."""
        m = 1.5 - 0.1j
        n_i = 1.2
        theta = np.radians(70.0)

        rho = fresnel.ellipsometry_rho(m, theta, n_i=n_i)
        rp = fresnel.r_par_amplitude(m, theta, n_i=n_i)
        rs = fresnel.r_per_amplitude(m, theta, n_i=n_i)
        rho_expected = rp / rs

        self.assertAlmostEqual(rho.real, rho_expected.real, places=12)
        self.assertAlmostEqual(rho.imag, rho_expected.imag, places=12)

        theta_deg = np.degrees(theta)
        rho_deg = fresnel.ellipsometry_rho(m, theta_deg, n_i=n_i, deg=True)
        self.assertAlmostEqual(rho_deg.real, rho_expected.real, places=12)
        self.assertAlmostEqual(rho_deg.imag, rho_expected.imag, places=12)

    def test_ellipsometry_index_inverts_rho(self):
        """Round-trip m -> rho -> m should recover the refractive index up to sign."""
        m_true = 1.5 - 0.1j
        n_i = 1.2
        theta = np.radians(70.0)

        rho = fresnel.ellipsometry_rho(m_true, theta, n_i=n_i)
        m_est = fresnel.ellipsometry_index(rho, theta, n_i=n_i)
        m_est_deg = fresnel.ellipsometry_index(rho, np.degrees(theta), n_i=n_i, deg=True)

        self.assertLess(min(abs(m_est - m_true), abs(m_est + m_true)), 1e-6)
        self.assertLess(min(abs(m_est_deg - m_true), abs(m_est_deg + m_true)), 1e-6)


if __name__ == "__main__":
    unittest.main()
