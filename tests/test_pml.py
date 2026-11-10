"""PML 匹配有耗层单测（v1.1）。"""
from __future__ import annotations

import numpy as np

from src.fdtd_core.pml import lossy_coeffs, sigma_max, sigma_profile_1d, sigma_profile_2d
from src.fdtd_core.yee1d import FDTD1D


class TestSigmaProfile:
    def test_zero_when_disabled(self):
        assert np.all(sigma_profile_1d(50, 0, 1e-3) == 0.0)

    def test_max_at_outer_edge_zero_at_interface(self):
        cells = 12
        sig = sigma_profile_1d(100, cells, 1e-3, m=3)
        smax = sigma_max(1e-3, 3)
        assert sig[0] == smax and sig[-1] == smax
        assert sig[cells - 1] == 0.0 and sig[100 - cells] == 0.0
        # 单调：左侧由外到内递减
        assert np.all(np.diff(sig[:cells]) <= 1e-12)
        assert np.all(np.diff(sig[100 - cells:]) >= -1e-12)

    def test_2d_map_corners(self):
        sig = sigma_profile_2d(40, 30, 6, 6, 1e-3, m=3)
        assert sig.shape == (40, 30)
        assert sig[0, 0] > 0 and sig[-1, -1] > 0
        assert sig[20, 15] == 0.0


class TestLossyCoeffs:
    def test_positive_decay(self):
        sig = sigma_profile_1d(50, 10, 1e-3, m=3)
        eps = 8.8541878128e-12 * np.ones(50)
        dt = 0.65 * 1e-3 / 299792458.0
        decay, drive = lossy_coeffs(sig, dt, eps)
        assert np.all(decay > 0) and np.all(decay <= 1.0)
        assert np.all(drive > 0) and np.all(drive <= 1.0 + 1e-12)

    def test_sigma_zero_limit(self):
        eps = 8.8541878128e-12 * np.ones(4)
        dt = 1e-12
        decay, drive = lossy_coeffs(np.zeros(4), dt, eps)
        np.testing.assert_allclose(decay, 1.0, rtol=1e-12)
        np.testing.assert_allclose(drive, 1.0, rtol=1e-12)


class TestSolverPML:
    def test_energy_decays(self):
        # 平滑高斯初值（单格尖峰含不传播的高-k 网格模，永远到不了 PML）
        sim = FDTD1D(nx=200, dx=1e-3, courant=0.65, pml_cells=16, pml_m=3)
        x = np.arange(200)
        sim.Ez = np.exp(-((x - 100.0) ** 2) / (2 * 8.0**2))
        e0 = float(np.sum(sim.Ez**2))
        for _ in range(1500):
            sim.step_h()
            sim.step_e()
        e1 = float(np.sum(sim.Ez**2) + np.sum(sim.Hy**2))
        assert e1 < 1e-3 * e0, f"PML 残余能量过大: {e1 / e0:.2e}"

    def test_no_nan_2d(self):
        from src.fdtd_core.yee2d import FDTD2D
        sim = FDTD2D(nx=48, ny=48, dx=1e-3, courant=0.65,
                     pml_cells_x=10, pml_cells_y=10, pml_m=3)
        for n in range(400):
            sim.step_h()
            sim.step_e(source=float(np.exp(-((n - 30) ** 2) / 200.0) * 1e-3),
                       source_index=(24, 24))
        assert np.all(np.isfinite(sim.Ez)) and np.all(np.isfinite(sim.Hx))

    def test_pml_reduces_echo_vs_mur(self):
        """S=0.65 宽带脉冲：PML 回波不应高于 Mur（同量级或更小）。"""
        def echo(pml_cells):
            sim = FDTD1D(nx=300, dx=1e-3, courant=0.65, pml_cells=pml_cells, pml_m=3)
            rec = np.empty(1200)
            for n in range(1200):
                sim.step_h()
                sim.step_e(source=float(-((n - 40) / 8.0)
                                        * np.exp(-(((n - 40) / 8.0) ** 2) / 2.0) / (8.0 * sim.dt)),
                           source_index=150)
                rec[n] = sim.Ez[50]
            n_inc = int(np.argmax(np.abs(rec[:350])))
            later = rec.copy()
            later[: n_inc + 120] = 0.0
            return float(np.abs(later).max() / abs(rec[n_inc]))
        assert echo(16) <= max(0.05, 1.5 * echo(0))
