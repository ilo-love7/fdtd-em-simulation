"""解析理论与频谱分析函数单测（无仿真）。"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.spectrum import (
    band_edges_from_transmission,
    bragg_wavelength,
    fresnel_R,
    fresnel_r,
    fresnel_t,
    quarter_wave_thickness,
    rel_err,
    single_slit_intensity,
    single_slit_minima,
    windowed_fft_ratio,
)


class TestFresnel:
    def test_normal_incidence_values(self):
        assert fresnel_r(1.0, 1.0) == pytest.approx(0.0)
        assert fresnel_r(1.0, 3.0) == pytest.approx(-0.5)
        assert fresnel_t(1.0, 3.0) == pytest.approx(0.5)

    def test_energy_conservation(self):
        n1, n2 = 1.0, 2.5
        r, t = fresnel_r(n1, n2), fresnel_t(n1, n2)
        # R + (n2/n1) T_amp² = 1（功率守恒：R + (n2/n1)t² = 1）
        assert fresnel_R(n1, n2) + (n2 / n1) * t**2 == pytest.approx(1.0)
        assert r**2 == pytest.approx(fresnel_R(n1, n2))


class TestSingleSlit:
    def test_zeroth_max(self):
        assert single_slit_intensity(0.0, 2.5) == pytest.approx(1.0)

    def test_minima_positions(self):
        s = single_slit_minima(a_over_lambda=2.5)
        np.testing.assert_allclose(s, [0.4, 0.8], rtol=1e-12)
        # 极小值处理论强度为 0
        assert single_slit_intensity(s, 2.5).max() < 1e-24

    def test_minima_limit(self):
        # a/λ = 0.6 → 仅 m=1 可见（sinθ=1.67 > 1 丢弃）
        s = single_slit_minima(a_over_lambda=0.6)
        np.testing.assert_allclose(s, [1.0 / 0.6] if 1.0 / 0.6 <= 1.0 else [], rtol=1e-12)
        assert s.size <= 1


class TestBragg:
    def test_quarter_wave_stack_center(self):
        lam0 = 1000e-9
        for n1, n2 in [(1.5, 2.5), (1.0, 2.0), (2.2, 1.4)]:
            d1 = quarter_wave_thickness(n1, lam0)
            d2 = quarter_wave_thickness(n2, lam0)
            assert bragg_wavelength(n1, d1, n2, d2) == pytest.approx(lam0, rel=1e-12)

    def test_thickness_formula(self):
        assert quarter_wave_thickness(2.0, 800e-9) == pytest.approx(100e-9)


class TestBandEdges:
    def test_symmetric_gap(self):
        lam = np.linspace(700e-9, 1300e-9, 61)
        c0 = 299_792_458.0
        f = c0 / lam
        T = np.where((lam > 900e-9) & (lam < 1100e-9), 0.01, 0.9)
        f_lo, f_hi = band_edges_from_transmission(f, T, threshold=0.5)
        lam_lo, lam_hi = c0 / max(f_lo, f_hi), c0 / min(f_lo, f_hi)
        assert lam_lo == pytest.approx(900e-9, rel=0.02)
        assert lam_hi == pytest.approx(1100e-9, rel=0.02)

    def test_no_gap(self):
        f = np.linspace(1e9, 2e9, 10)
        T = np.full(10, 0.8)
        f_lo, f_hi = band_edges_from_transmission(f, T)
        assert f_lo != f_lo and f_hi != f_hi  # NaN


class TestMetrics:
    def test_rel_err(self):
        assert rel_err(110.0, 100.0) == pytest.approx(10.0)
        assert rel_err(100.0, 0.0) == pytest.approx(100.0)  # 绝对差口径

    def test_windowed_fft_ratio_phase(self):
        # 同频正弦，反射窗相位偏移 90°，幅值比 0.5
        n = 512
        t = np.arange(n) * 1.0
        f0 = 1.0 / 64.0
        y_inc = np.sin(2 * np.pi * f0 * t)
        y_ref = 0.5 * np.sin(2 * np.pi * f0 * t + np.pi / 2)
        res = windowed_fft_ratio(t, y_inc, y_ref, n_center_inc=128,
                                 n_center_ref=384, half_width=64, f_eval=f0)
        assert res["r_abs"] == pytest.approx(0.5, rel=0.05)
        assert abs(res["r_phase_deg"] - 90.0) < 10.0
