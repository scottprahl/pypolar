"""Unit tests for the symbolic Fresnel functions in :mod:`pypolar.sym_fresnel`.

These tests compare symbolic results to the numeric Fresnel implementation,
check energy conservation in lossless cases, and verify ellipsometric helpers.
"""

import unittest
import numpy as np  # noqa: F401  # may be useful for future tests
import sympy as sp

from pypolar import fresnel, sym_fresnel

RT_TOL = 1e-10
VAL_TOL = 1e-10


def _sym_to_complex(expr):
    """Convert a SymPy expression to a Python complex number."""
    return complex(sp.N(expr))


class TestSymFresnelSurface(unittest.TestCase):
    """Tests for parity between numeric and symbolic Fresnel utility surfaces."""

    def test_public_api_matches_numeric_module(self):
        """Symbolic and numeric Fresnel modules should export the same public names."""
        self.assertEqual(set(sym_fresnel.__all__), set(fresnel.__all__))

    def test_brewster_and_critical_match_numeric(self):
        """Symbolic Brewster/critical angle helpers should match numeric values."""
        m = 1.5
        n_i = 1.2

        b_sym = float(sp.N(sym_fresnel.brewster(m, n_i=n_i)))
        b_num = fresnel.brewster(m, n_i=n_i)
        self.assertAlmostEqual(b_sym, b_num, places=12)

        b_sym_deg = float(sp.N(sym_fresnel.brewster(m, n_i=n_i, deg=True)))
        b_num_deg = fresnel.brewster(m, n_i=n_i, deg=True)
        self.assertAlmostEqual(b_sym_deg, b_num_deg, places=12)

        c_sym = float(sp.N(sym_fresnel.critical(1.0, n_i=1.5)))
        c_num = fresnel.critical(1.0, n_i=1.5)
        self.assertAlmostEqual(c_sym, c_num, places=12)

        c_sym_deg = float(sp.N(sym_fresnel.critical(1.0, n_i=1.5, deg=True)))
        c_num_deg = fresnel.critical(1.0, n_i=1.5, deg=True)
        self.assertAlmostEqual(c_sym_deg, c_num_deg, places=12)


class TestSymFresnelAmplitudes(unittest.TestCase):
    """Tests for symbolic Fresnel amplitude coefficients."""

    def test_matches_numeric_at_normal_incidence_real_index(self):
        """Compare symbolic and numeric amplitudes at normal incidence, real m."""
        m = 1.5
        theta = 0.0

        # sympy
        rp_sym = _sym_to_complex(sym_fresnel.r_par_amplitude(m, theta))
        rs_sym = _sym_to_complex(sym_fresnel.r_per_amplitude(m, theta))
        tp_sym = _sym_to_complex(sym_fresnel.t_par_amplitude(m, theta))
        ts_sym = _sym_to_complex(sym_fresnel.t_per_amplitude(m, theta))

        # numeric
        rp_num = fresnel.r_par_amplitude(m, theta)
        rs_num = fresnel.r_per_amplitude(m, theta)
        tp_num = fresnel.t_par_amplitude(m, theta)
        ts_num = fresnel.t_per_amplitude(m, theta)

        self.assertAlmostEqual(rp_sym, rp_num, places=12)
        self.assertAlmostEqual(rs_sym, rs_num, places=12)
        self.assertAlmostEqual(tp_sym, tp_num, places=12)
        self.assertAlmostEqual(ts_sym, ts_num, places=12)

    def test_matches_numeric_oblique_incidence_real_index(self):
        """Compare symbolic and numeric amplitudes at oblique incidence, real m."""
        m = 1.5
        theta = np.radians(37.0)

        rp_sym = _sym_to_complex(sym_fresnel.r_par_amplitude(m, theta))
        rs_sym = _sym_to_complex(sym_fresnel.r_per_amplitude(m, theta))
        tp_sym = _sym_to_complex(sym_fresnel.t_par_amplitude(m, theta))
        ts_sym = _sym_to_complex(sym_fresnel.t_per_amplitude(m, theta))

        rp_num = fresnel.r_par_amplitude(m, theta)
        rs_num = fresnel.r_per_amplitude(m, theta)
        tp_num = fresnel.t_par_amplitude(m, theta)
        ts_num = fresnel.t_per_amplitude(m, theta)

        self.assertAlmostEqual(rp_sym, rp_num, places=12)
        self.assertAlmostEqual(rs_sym, rs_num, places=12)
        self.assertAlmostEqual(tp_sym, tp_num, places=12)
        self.assertAlmostEqual(ts_sym, ts_num, places=12)

    def test_complex_index_matches_numeric_at_normal_incidence(self):
        """Compare symbolic and numeric amplitudes at normal incidence, complex m."""
        m = 1.5 - 0.2j
        theta = 0.0

        rp_sym = _sym_to_complex(sym_fresnel.r_par_amplitude(m, theta))
        rs_sym = _sym_to_complex(sym_fresnel.r_per_amplitude(m, theta))
        tp_sym = _sym_to_complex(sym_fresnel.t_par_amplitude(m, theta))
        ts_sym = _sym_to_complex(sym_fresnel.t_per_amplitude(m, theta))

        rp_num = fresnel.r_par_amplitude(m, theta)
        rs_num = fresnel.r_per_amplitude(m, theta)
        tp_num = fresnel.t_par_amplitude(m, theta)
        ts_num = fresnel.t_per_amplitude(m, theta)

        self.assertAlmostEqual(rp_sym, rp_num, places=12)
        self.assertAlmostEqual(rs_sym, rs_num, places=12)
        self.assertAlmostEqual(tp_sym, tp_num, places=12)
        self.assertAlmostEqual(ts_sym, ts_num, places=12)

    def test_nonunity_incident_index_and_degree_mode_match_numeric(self):
        """Symbolic amplitudes should match numeric values for n_i != 1 and deg=True."""
        m = 1.5 - 0.1j
        n_i = 1.2
        theta_deg = 37.0

        rp_sym = _sym_to_complex(sym_fresnel.r_par_amplitude(m, theta_deg, n_i=n_i, deg=True))
        rs_sym = _sym_to_complex(sym_fresnel.r_per_amplitude(m, theta_deg, n_i=n_i, deg=True))
        tp_sym = _sym_to_complex(sym_fresnel.t_par_amplitude(m, theta_deg, n_i=n_i, deg=True))
        ts_sym = _sym_to_complex(sym_fresnel.t_per_amplitude(m, theta_deg, n_i=n_i, deg=True))

        rp_num = fresnel.r_par_amplitude(m, theta_deg, n_i=n_i, deg=True)
        rs_num = fresnel.r_per_amplitude(m, theta_deg, n_i=n_i, deg=True)
        tp_num = fresnel.t_par_amplitude(m, theta_deg, n_i=n_i, deg=True)
        ts_num = fresnel.t_per_amplitude(m, theta_deg, n_i=n_i, deg=True)

        self.assertAlmostEqual(rp_sym.real, rp_num.real, places=10)
        self.assertAlmostEqual(rp_sym.imag, rp_num.imag, places=10)
        self.assertAlmostEqual(rs_sym.real, rs_num.real, places=10)
        self.assertAlmostEqual(rs_sym.imag, rs_num.imag, places=10)
        self.assertAlmostEqual(tp_sym.real, tp_num.real, places=10)
        self.assertAlmostEqual(tp_sym.imag, tp_num.imag, places=10)
        self.assertAlmostEqual(ts_sym.real, ts_num.real, places=10)
        self.assertAlmostEqual(ts_sym.imag, ts_num.imag, places=10)


class TestSymFresnelPower(unittest.TestCase):
    """Tests for symbolic power reflectance/transmittance and unpolarized cases."""

    def test_power_coefficients_match_numeric_real_index(self):
        """Compare symbolic and numeric power coefficients for real m."""
        m = 1.5
        theta = np.radians(30.0)

        Rp_sym = float(sp.N(sym_fresnel.R_par(m, theta)))
        Rs_sym = float(sp.N(sym_fresnel.R_per(m, theta)))
        Tp_sym = float(sp.N(sym_fresnel.T_par(m, theta)))
        Ts_sym = float(sp.N(sym_fresnel.T_per(m, theta)))

        Rp_num = fresnel.R_par(m, theta)
        Rs_num = fresnel.R_per(m, theta)
        Tp_num = fresnel.T_par(m, theta)
        Ts_num = fresnel.T_per(m, theta)

        self.assertAlmostEqual(Rp_sym, Rp_num, places=10)
        self.assertAlmostEqual(Rs_sym, Rs_num, places=10)
        self.assertAlmostEqual(Tp_sym, Tp_num, places=10)
        self.assertAlmostEqual(Ts_sym, Ts_num, places=10)

    def test_energy_conservation_lossless(self):
        """Check Rp+Tp = 1 and Rs+Ts = 1 for real, lossless m."""
        m = 1.5
        theta = np.radians(40.0)

        Rp = float(sp.N(sym_fresnel.R_par(m, theta)))
        Rs = float(sp.N(sym_fresnel.R_per(m, theta)))
        Tp = float(sp.N(sym_fresnel.T_par(m, theta)))
        Ts = float(sp.N(sym_fresnel.T_per(m, theta)))

        self.assertAlmostEqual(Rp + Tp, 1.0, delta=RT_TOL)
        self.assertAlmostEqual(Rs + Ts, 1.0, delta=RT_TOL)

    def test_unpolarized_reflection_and_transmission(self):
        """Check unpolarized symbolic R,T against s/p averages and R+T=1."""
        m = 1.5
        theta = np.radians(25.0)

        Rp = float(sp.N(sym_fresnel.R_par(m, theta)))
        Rs = float(sp.N(sym_fresnel.R_per(m, theta)))
        Tp = float(sp.N(sym_fresnel.T_par(m, theta)))
        Ts = float(sp.N(sym_fresnel.T_per(m, theta)))

        R_un = float(sp.N(sym_fresnel.R_unpolarized(m, theta)))
        T_un = float(sp.N(sym_fresnel.T_unpolarized(m, theta)))

        # By definition, unpolarized is the average of s and p
        self.assertAlmostEqual(R_un, 0.5 * (Rp + Rs), places=10)
        self.assertAlmostEqual(T_un, 0.5 * (Tp + Ts), places=10)

        # And R_un + T_un = 1 in lossless case
        self.assertAlmostEqual(R_un + T_un, 1.0, delta=RT_TOL)

    def test_matches_numeric_unpolarized(self):
        """Compare symbolic and numeric unpolarized R and T."""
        m = 1.5
        theta = np.radians(33.0)

        R_un_sym = float(sp.N(sym_fresnel.R_unpolarized(m, theta)))
        T_un_sym = float(sp.N(sym_fresnel.T_unpolarized(m, theta)))

        R_un_num = fresnel.R_unpolarized(m, theta)
        T_un_num = fresnel.T_unpolarized(m, theta)

        self.assertAlmostEqual(R_un_sym, R_un_num, places=10)
        self.assertAlmostEqual(T_un_sym, T_un_num, places=10)


class TestSymFresnelEllipsometry(unittest.TestCase):
    """Tests for symbolic ellipsometric quantities rho and m."""

    def test_ellipsometry_rho_matches_numeric_ratio(self):
        """Check rho = rp/rs from sym_fresnel matches numeric Fresnel ratio."""
        m = 1.5 - 0.1j
        theta = np.radians(70.0)

        rho_sym = _sym_to_complex(sym_fresnel.ellipsometry_rho(m, theta))

        rp = fresnel.r_par_amplitude(m, theta)
        rs = fresnel.r_per_amplitude(m, theta)
        rho_num = rp / rs

        self.assertAlmostEqual(rho_sym.real, rho_num.real, places=10)
        self.assertAlmostEqual(rho_sym.imag, rho_num.imag, places=10)

    def test_ellipsometry_index_inverts_rho(self):
        """Round-trip test: m → rho → m_est using ellipsometry_index."""
        m_true = 1.5 - 0.1j
        theta = np.radians(70.0)

        rho = sym_fresnel.ellipsometry_rho(m_true, theta)
        m_est_expr = sym_fresnel.ellipsometry_index(rho, theta)
        m_est = _sym_to_complex(m_est_expr)

        # Because of possible branch choices, allow for +/- ambiguity
        diff1 = abs(m_est - m_true)
        diff2 = abs(m_est + m_true)

        self.assertTrue(
            min(diff1, diff2) < 1e-6,
            msg=f"ellipsometry_index did not recover m: m_true={m_true}, m_est={m_est}",
        )


if __name__ == "__main__":
    unittest.main()
