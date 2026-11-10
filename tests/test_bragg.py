"""fdtd 库封装单测：Bragg 网格几何、逆介电常数修复、CW 幅值。

不跑完整透射谱（太慢），只验证构造正确性与最小可运行性。
"""
from __future__ import annotations

import numpy as np
import pytest

fd = pytest.importorskip("fdtd")

from src.analysis.spectrum import bragg_wavelength, quarter_wave_thickness
from src.fdtd_lib.bragg import build_bragg_grid, cw_amplitude


class TestBuildGrid:
    def test_geometry(self):
        lam0 = 1000e-9
        grid, meta = build_bragg_grid(n_layers=2, lam0_design_m=lam0,
                                      n1=1.5, n2=2.5, dx=5e-9, pad_cells=20)
        d1 = quarter_wave_thickness(1.5, lam0)
        d2 = quarter_wave_thickness(2.5, lam0)
        assert meta["d1_m"] == pytest.approx(d1, rel=1e-12)
        assert meta["d2_m"] == pytest.approx(d2, rel=1e-12)
        assert meta["lam_bragg_analytic_m"] == pytest.approx(
            bragg_wavelength(1.5, d1, 2.5, d2), rel=1e-12)
        assert isinstance(meta["period_steps"], int) and meta["period_steps"] > 0
        assert meta["total_cells"] == 20 * 2 + 2 * (meta["n1_cells"] + meta["n2_cells"])

    def test_inverse_permittivity_fix(self):
        # 库的对称化会把 z=-1 行逆介电常数重置为背景 → 修复后须等于 1/eps
        lam0 = 1000e-9
        grid, meta = build_bragg_grid(n_layers=1, lam0_design_m=lam0,
                                      n1=1.5, n2=2.5, dx=20e-9, pad_cells=16)
        obj = grid.objects[0]
        inv = np.asarray(obj.inverse_permittivity)  # (Nx, Ny, Nz, 3)
        eps = np.asarray(obj.permittivity, dtype=float)  # (Nx, Ny, Nz, 1)
        np.testing.assert_allclose(inv, np.broadcast_to(1.0 / eps, inv.shape), rtol=1e-12)
        # z=-1 行（最后一行）不再全为背景（1/1）
        assert not np.allclose(inv[:, :, -1, 2], 1.0)

    def test_pml_and_object_no_overlap(self):
        grid, meta = build_bragg_grid(n_layers=1, dx=20e-9, pad_cells=16)
        total = meta["total_cells"]
        obj = grid.objects[0]
        # Object 从 y=8 到 y=total-8（避开两端 PML）
        assert obj.Ny == total - 16


class TestCW:
    def test_reference_amplitude_finite(self):
        # n_layers=0（真空参考）短跑：幅值须为有限正数
        grid, meta = build_bragg_grid(n_layers=0, dx=20e-9, pad_cells=16,
                                      lam_run_m=1000e-9)
        amp = cw_amplitude(grid, total_periods=3.0, measure_periods=1.0)
        assert np.isfinite(amp) and amp > 0
