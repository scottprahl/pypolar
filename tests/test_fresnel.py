import math
import unittest

import numpy as np

from pypolar import fresnel


RT_TOL = 1e-12  # tight tolerance for R + T = 1 tests
ANGLE_TOL = 1e-12


class TestFresnelBasicAngles(unittest.TestCase):
    def test_brewster_angle_radians_and_degrees(self):
        n_i = 1.0
        m = 1.5  # real dielectric

        # Brewster angle: tan(theta_B) = m / n_i
        expected_rad = math.atan2(m, n_i)
        expected_deg = math.degrees(expected_rad)

        theta_B_rad = fresnel.brewster(m, n_i=n_i, deg=False)
        theta_B_deg = fresnel.brewster(m, n_i=n_i, deg=True)

        self.assertAlmostEqual(theta_B_rad, expected_rad, delta=ANGLE_TOL)
        self.assertAlmostEqual(theta_B_deg, expected_deg, delta=ANGLE_TOL)

        # At Brewster angle, R_par should vanish (for lossless media)
        R_p = fresnel.R_par(m, theta_B_rad, n_i=n_i, deg=False)
        self.assertAlmostEqual(R_p, 0.0, places=12)

    def test_critical_angle_radians_and_degrees(self):
        # total internal reflection: incident from higher to lower index
        n_i = 1.5
        m = 1.0   # transmitted medium

        # critical angle: sin(theta_c) = m / n_i
        expected_rad = math.asin(m / n_i)
        expected_deg = math.degrees(expected_rad)

        theta_c_rad = fresnel.critical(m, n_i=n_i, deg=False)
        theta_c_deg = fresnel.critical(m, n_i=n_i, deg=True)

        self.assertAlmostEqual(theta_c_rad, expected_rad, delta=ANGLE_TOL)
        self.assertAlmostEqual(theta_c_deg, expected_deg, delta=ANGLE_TOL)


class TestFresnelAmplitudes(unittest.TestCase):
    def test_normal_incidence_symmetry_real_indices(self):
        # At normal incidence with real indices, |r_s| = |r_p| and t_s = t_p
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
        n_i = 1.0
        m = 1.5
        theta_deg = 37.0
        theta_rad = math.radians(theta_deg)

        rp_rad = fresnel.r_par_amplitude(m, theta_rad, n_i=n_i, deg=False)
        rp_deg = fresnel.r_par_amplitude(m, theta_deg, n_i=n_i, deg=True)

        rs_rad = fresnel.r_per_amplitude(m, theta_rad, n_i=n_i, deg=False)
        rs_deg = fresnel.r_per_amplitude(m, theta_deg, n_i=n_i, deg=True)

        self.assertAlmostEqual(rp_rad, rp_deg, places=14)
        self.assertAlmostEqual(rs_rad, rs_deg, places=14)

    def test_complex_index_equal_power_reflectance_at_normal_incidence(self):
        # With a complex index at normal incidence, power reflectance
        # and transmittance should match for s and p.
        n_i = 1.0
        m = 1.5 - 0.2j
        theta = 0.0

        rp = fresnel.r_par_amplitude(m, theta, n_i=n_i)
        rs = fresnel.r_per_amplitude(m, theta, n_i=n_i)
        tp = fresnel.t_par_amplitude(m, theta, n_i=n_i)
        ts = fresnel.t_per_amplitude(m, theta, n_i=n_i)

        R_p = abs(rp) ** 2
        R_s = abs(rs) ** 2

        # Power reflectance for s and p should be identical at normal incidence
        self.assertAlmostEqual(R_p, R_s, places=12)

        # Likewise for power transmission
        T_p = fresnel.T_par(m, theta, n_i=n_i)
        T_s = fresnel.T_per(m, theta, n_i=n_i)
        self.assertAlmostEqual(T_p, T_s, places=12)


class TestFresnelPowerConservation(unittest.TestCase):
    def test_energy_conservation_lossless_normal_incidence(self):
        n_i = 1.0
        m = 1.5   # real, lossless
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
        n_i = 1.0
        m = 1.5  # real, lossless
        theta = math.radians(45.0)  # below critical, no TIR

        Rp = fresnel.R_par(m, theta, n_i=n_i)
        Rs = fresnel.R_per(m, theta, n_i=n_i)
        Tp = fresnel.T_par(m, theta, n_i=n_i)
        Ts = fresnel.T_per(m, theta, n_i=n_i)

        self.assertAlmostEqual(Rp + Tp, 1.0, delta=RT_TOL)
        self.assertAlmostEqual(Rs + Ts, 1.0, delta=RT_TOL)

    def test_unpolarized_reflection_and_transmission(self):
        n_i = 1.0
        m = 1.5
        theta = math.radians(30.0)

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
        n_i = 1.0
        m = 1.5  # real dielectric

        theta_B = fresnel.brewster(m, n_i=n_i, deg=False)

        Rp = fresnel.R_par(m, theta_B, n_i=n_i)
        # Allow a little numerical noise
        self.assertAlmostEqual(Rp, 0.0, places=12)

    def test_total_internal_reflection_reflectance_near_one(self):
        # Incident from n_i > m, angle above critical
        n_i = 1.5
        m = 1.0
        theta_c = fresnel.critical(m, n_i=n_i, deg=False)
        theta = theta_c + math.radians(5.0)

        Rp = fresnel.R_par(m, theta, n_i=n_i)
        Rs = fresnel.R_per(m, theta, n_i=n_i)

        # Reflection should be ~1 for both polarizations under TIR
        self.assertTrue(Rp > 0.999)
        self.assertTrue(Rs > 0.999)


class TestVectorization(unittest.TestCase):
    def test_array_input_for_angles(self):
        n_i = 1.0
        m = 1.5
        theta = np.linspace(0.0, math.radians(80.0), 5)

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


if __name__ == "__main__":
    unittest.main()
