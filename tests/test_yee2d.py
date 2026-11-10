"""二维 TMz Yee FDTD 单测：CFL 守卫、稳定性、y 均匀性、缝屏掩码。"""
from __future__ import annotations

import numpy as np
import pytest

from src.fdtd_core.yee2d import FDTD2D


class TestGuard:
    def test_cfl_reject(self):
        with pytest.raises(ValueError):
            FDTD2D(nx=10, ny=10, dx=1e-3, courant=0.71)  # > 1/√2
        with pytest.raises(ValueError):
            FDTD2D(nx=10, ny=10, dx=1e-3, courant=0.0)

    def test_eps_shape(self):
        with pytest.raises(ValueError):
            FDTD2D(nx=8, ny=8, dx=1e-3, eps_r=np.ones((4, 4)))


class TestMasks:
    def test_box_pec(self):
        m = FDTD2D.box_pec(20, 20, 5, 10, 3, 8)
        assert m[5:10, 3:8].all() and m.sum() == 5 * 5
        assert not m[4, 3] and not m[5, 2]

    def test_slit_screen(self):
        m = FDTD2D.slit_screen(50, 40, x_slit=20, thickness=2, y_slit0=15, y_slit1=25)
        assert m[20:22, 15:25].sum() == 0          # 缝区留空
        assert m[20:22, :15].all() and m[20:22, 25:].all()  # 其余金属
        assert m[19].sum() == 0 and m[22].sum() == 0


class TestStability:
    def test_point_source_bounded(self):
        sim = FDTD2D(nx=60, ny=60, dx=1e-3, courant=0.65)
        for n in range(400):
            sim.step_h()
            sim.step_e(source=np.sin(2 * np.pi * n / 30.0),
                       source_index=(20, 30))
        assert np.all(np.isfinite(sim.Ez))
        assert np.abs(sim.Ez).max() < 1e3

    def test_pec_wall_reflects(self):
        # 行源 + PEC 墙：墙左侧场应显著大于墙右侧（透不过金属）
        nx, ny = 120, 40
        sim = FDTD2D(nx=nx, ny=ny, dx=1e-3, courant=0.65)
        pec = FDTD2D.box_pec(nx, ny, 80, 82, 0, ny)
        for n in range(500):
            sim.step_h()
            sim.step_e_line(source=np.sin(2 * np.pi * n / 30.0), line_source=10,
                            pec_mask=pec, y_neumann=True)
        left = np.abs(sim.Ez[20:60, :]).mean()
        right = np.abs(sim.Ez[100:, :]).mean()
        assert left > 3 * right, f"PEC 透波：left={left:.3e} right={right:.3e}"


class TestYNeumann:
    def test_line_source_uniform_in_y(self):
        # y_neumann 行源：y 向各列场应保持均匀（平面波）
        nx, ny = 100, 16
        sim = FDTD2D(nx=nx, ny=ny, dx=1e-3, courant=0.65)
        for n in range(300):
            sim.step_h()
            sim.step_e_line(source=np.sin(2 * np.pi * n / 24.0),
                            line_source=15, y_neumann=True)
        row = sim.Ez[60, :]
        # 边界列受 Neumann 拷贝影响，测内部列
        interior = row[2:-2]
        assert interior.std() < 0.05 * max(abs(interior).max(), 1e-30), (
            f"y 不均匀：std={interior.std():.2e} max={np.abs(interior).max():.2e}")

    def test_mur_default_finite(self):
        sim = FDTD2D(nx=50, ny=50, dx=1e-3)
        for n in range(200):
            sim.step_h()
            sim.step_e(source=np.sin(2 * np.pi * n / 20.0), source_index=(25, 25))
        assert np.all(np.isfinite(sim.Ez))
