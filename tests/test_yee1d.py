"""一维 Yee FDTD 单测：CFL 守卫、传播速度、PEC 反射、Fresnel 界面。"""
from __future__ import annotations

import numpy as np
import pytest

from src.fdtd_core.yee1d import C0, FDTD1D, gaussian_derivative


def run_pulse(nx=400, dx=1e-3, courant=0.95, eps_r=None, pec_from=None,
              src=40, probe=160, n_steps=700, t0=25.0, w=6.0):
    sim = FDTD1D(nx=nx, dx=dx, courant=courant, eps_r=eps_r)
    pec = None
    if pec_from is not None:
        pec = np.zeros(nx, dtype=bool)
        pec[pec_from:] = True
    rec = np.empty(n_steps)
    for n in range(n_steps):
        sim.step_h()
        t = n * sim.dt
        s = gaussian_derivative(t, t0 * sim.dt, w * sim.dt)
        sim.step_e(source=s, source_index=src, pec_mask=pec)
        rec[n] = sim.Ez[probe]
    return sim.dt, rec


class TestGuard:
    def test_cfl_reject(self):
        with pytest.raises(ValueError):
            FDTD1D(nx=10, dx=1e-3, courant=1.01)
        with pytest.raises(ValueError):
            FDTD1D(nx=10, dx=1e-3, courant=0.0)

    def test_eps_shape(self):
        with pytest.raises(ValueError):
            FDTD1D(nx=10, dx=1e-3, eps_r=np.ones(5))

    def test_no_nan_after_steps(self):
        sim = FDTD1D(nx=100, dx=1e-3)
        for n in range(200):
            # 源乘 w·dt 归一到 O(1) 幅值，stability 判据才有意义
            sim.step_h()
            sim.step_e(source=gaussian_derivative(n * sim.dt, 30 * sim.dt, 5 * sim.dt) * (5 * sim.dt),
                       source_index=10)
        assert np.all(np.isfinite(sim.Ez)) and np.all(np.isfinite(sim.Hy))
        assert np.abs(sim.Ez).max() < 1e6


class TestPropagation:
    def test_speed(self):
        # 双探针包络峰到达时刻 → 数值相速度
        nx, dx, s0 = 500, 1e-3, 0.95
        sim = FDTD1D(nx=nx, dx=dx, courant=s0)
        pa, pb = 150, 350
        ra = np.empty(1000)
        rb = np.empty(1000)
        for n in range(1000):
            sim.step_h()
            sim.step_e(source=gaussian_derivative(n * sim.dt, 30 * sim.dt, 6 * sim.dt),
                       source_index=50)
            ra[n] = sim.Ez[pa]
            rb[n] = sim.Ez[pb]
        na = int(np.argmax(np.abs(ra)))
        nb = int(np.argmax(np.abs(rb)))
        v_sim = (pb - pa) * dx / ((nb - na) * sim.dt)
        err = abs(v_sim / C0 - 1.0) * 100
        assert err < 1.5, f"速度误差 {err:.2f}%"


class TestReflection:
    def test_pec_reflection(self):
        # PEC 墙：反射脉冲与入射等幅反相（双极性脉冲符号易抓错 → 时窗 RMS + 相关性）
        _, rec = run_pulse(nx=400, pec_from=300, src=40, probe=160, n_steps=700)
        s = 0.95
        n_arr = int(round(25 + (160 - 40) / s))              # 入射到达
        n_back = int(round(n_arr + 2 * (300 - 160) / s))     # 反射返回
        w = 40
        inc = rec[n_arr - w: n_arr + w]
        ref = rec[n_back - w: n_back + w]
        r_rms = float(np.sqrt((ref**2).sum() / (inc**2).sum()))
        corr = float(np.corrcoef(inc, -ref)[0, 1])
        assert abs(r_rms - 1.0) < 0.15, f"PEC |r|_rms={r_rms:.3f}"
        assert corr > 0.95, f"反射应与入射反相，corr(inc,-ref)={corr:.3f}"

    def test_mur_absorption(self):
        # Mur 吸收边界：无反射墙时探针扰动应快速衰减
        _, rec = run_pulse(nx=400, pec_from=None, src=40, probe=200, n_steps=700)
        tail = np.abs(rec[500:]).max()
        assert tail < 0.05 * np.abs(rec).max(), "Mur 边界回波过大"


class TestFresnel:
    @pytest.mark.parametrize("n2", [1.5, 2.0])
    def test_fresnel_r(self, n2):
        from src.analysis.spectrum import fresnel_r, windowed_fft_ratio

        nx = 600
        dx = 1e-3
        i0 = 300
        eps = np.ones(nx)
        eps[i0:] = n2**2
        sim = FDTD1D(nx=nx, dx=dx, courant=0.95, eps_r=eps)
        src, probe = 50, 200
        rec = np.empty(1100)
        for n in range(1100):
            sim.step_h()
            sim.step_e(source=gaussian_derivative(n * sim.dt, 30 * sim.dt, 8 * sim.dt),
                       source_index=src)
            rec[n] = sim.Ez[probe]
        dt = sim.dt
        t = np.arange(1100) * dt
        s = 0.95
        n_inc = 30 + (probe - src) / s
        n_ref = 30 + (probe + (i0 - probe) * 2 - src) / s
        res = windowed_fft_ratio(t, rec, rec, int(n_inc), int(n_ref), half_width=70)
        r_theory = fresnel_r(1.0, n2)
        r_sim = res["r_complex"].real
        assert abs(r_sim - r_theory) < 0.03, (
            f"n2={n2}: r_sim={r_sim:.3f} vs theory {r_theory:.3f}")
