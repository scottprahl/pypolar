import math
import unittest

import numpy as np
import sympy as sp

from pypolar import fresnel, sym_fresnel


RT_TOL = 1e-10
VAL_TOL = 1e-10


def _sym_to_complex(expr):
    """Convert a SymPy expression to a Python complex."""
    return complex(sp.N(expr))


class TestSymFresnelAmplitudes(unittest.TestCase):
    def test_matches_numeric_at_normal_incidence_real_index(self):
        # Real, lossless interface at normal incidence: compare against fresnel
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
        m = 1.5
        theta = math.radians(37.0)

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


class TestSymFresnelPower(unittest.TestCase):
    def test_power_coefficients_match_numeric_real_index(self):
        m = 1.5
        theta = math.radians(30.0)

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
        # real, lossless case: R + T = 1 for both polarizations
        m = 1.5
        theta = math.radians(40.0)

        Rp = float(sp.N(sym_fresnel.R_par(m, theta)))
        Rs = float(sp.N(sym_fresnel.R_per(m, theta)))
        Tp = float(sp.N(sym_fresnel.T_par(m, theta)))
        Ts = float(sp.N(sym_fresnel.T_per(m, theta)))

        self.assertAlmostEqual(Rp + Tp, 1.0, delta=RT_TOL)
        self.assertAlmostEqual(Rs + Ts, 1.0, delta=RT_TOL)

    def test_unpolarized_reflection_and_transmission(self):
        m = 1.5
        theta = math.radians(25.0)

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
        # cross-check unpolarized against numeric fresnel module
        m = 1.5
        theta = math.radians(33.0)

        R_un_sym = float(sp.N(sym_fresnel.R_unpolarized(m, theta)))
        T_un_sym = float(sp.N(sym_fresnel.T_unpolarized(m, theta)))

        R_un_num = fresnel.R_unpolarized(m, theta)
        T_un_num = fresnel.T_unpolarized(m, theta)

        self.assertAlmostEqual(R_un_sym, R_un_num, places=10)
        self.assertAlmostEqual(T_un_sym, T_un_num, places=10)


class TestSymFresnelEllipsometry(unittest.TestCase):
    def test_ellipsometry_rho_matches_numeric_ratio(self):
        # check that rho = r_p / r_s matches numeric fresnel module
        m = 1.5 - 0.1j
        theta = math.radians(70.0)

        rho_sym = _sym_to_complex(sym_fresnel.ellipsometry_rho(m, theta))

        rp = fresnel.r_par_amplitude(m, theta)
        rs = fresnel.r_per_amplitude(m, theta)
        rho_num = rp / rs

        self.assertAlmostEqual(rho_sym.real, rho_num.real, places=10)
        self.assertAlmostEqual(rho_sym.imag, rho_num.imag, places=10)

    def test_ellipsometry_index_inverts_rho(self):
        # round-trip: m -> rho -> m_est
        m_true = 1.5 - 0.1j
        theta = math.radians(70.0)

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
