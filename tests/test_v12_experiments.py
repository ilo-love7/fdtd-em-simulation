"""v1.2 单测：Drude ADE / TFSF 连接边界 / 数值色散解析关系。"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.dispersive import (drude_eps_r, drude_n, group_index,
                                     slab_rt)
from src.analysis.dispersion import (anisotropy_2d, group_velocity_1d,
                                     k_discrete_1d, k_discrete_2d,
                                     omega_from_k, phase_velocity_1d)
from src.fdtd_core.drude import FDTD1D_Drude, cw_ramp
from src.fdtd_core.tfsf import FDTD1D_TFSF
from src.fdtd_core.yee1d import C0


# ---------------- Drude 解析 ----------------
def test_drude_eps_loss_sign():
    """γ>0, ω>0 ⇒ Im ε > 0（损耗）。"""
    eps = drude_eps_r(0.2, wp2=0.01, gamma=0.01)
    assert eps.imag > 0


def test_drude_n_evanescent_below_cutoff():
    """ω<ω_p（γ=0）⇒ ε<0 ⇒ n 纯虚（Im>0）。"""
    n = drude_n(0.05, wp2=0.01, gamma=0.0)
    assert n.real == pytest.approx(0.0, abs=1e-12)
    assert n.imag > 0


def test_drude_n_propagating_above_cutoff():
    n = drude_n(0.2, wp2=0.01, gamma=0.0)
    assert 0 < n.real < 1  # n<1 冷等离子体，v_ph>c


def test_slab_rt_below_cutoff_total_reflection():
    """无损、ω<ω_p ⇒ |r|=1, |t|≈0（倏逝隧穿为有限厚度的指数小量）。"""
    r, t = slab_rt(0.06, wp2=0.01, gamma=0.0, d=60)
    assert abs(r) == pytest.approx(1.0, abs=0.05)
    assert abs(t) < 0.1


def test_slab_rt_above_cutoff_low_reflection():
    r, t = slab_rt(0.5, wp2=0.01, gamma=0.0, d=60)
    assert abs(r) < 0.1
    assert abs(t) > 0.9


def test_group_index_gt_phase_index():
    omega, wp2 = 0.3, 0.01
    n = float(np.real(drude_n(omega, wp2, 0.0)))
    assert group_index(omega, wp2, 0.0) > n  # n_g > n（正常色散）


# ---------------- Drude FDTD ----------------
def test_drude_vacuum_limit_matches_plain_yee():
    """wp2=0 ⇒ J≡0，退化为普通 FDTD（脉冲传播不衰）。"""
    sim = FDTD1D_Drude(nx=400, courant=0.9)
    src = lambda t: cw_ramp(t, 0.3, 10.0)
    rec = sim.run_record(500, source_fn=src, source_index=100,
                         record_indices=(250,))
    assert np.max(np.abs(rec["p0"])) > 1e-3
    assert np.all(np.isfinite(rec["p0"]))


def test_drude_ade_stable():
    """有损 Drude 长时间步进不发散。"""
    ei, w2, ga = FDTD1D_Drude.drude_box(600, 200, 400, wp2=0.01, gamma=0.02)
    sim = FDTD1D_Drude(nx=600, courant=0.9, eps_inf=ei, wp2=w2, gamma=ga)
    src = lambda t: cw_ramp(t, 0.2, 30.0)
    rec = sim.run_record(3000, source_fn=src, source_index=100,
                         record_indices=(500, 150))
    assert np.all(np.isfinite(rec["p0"])) and np.all(np.isfinite(rec["p1"]))
    assert np.max(np.abs(rec["p0"])) < 100


def test_drude_box_material_mask():
    ei, w2, ga = FDTD1D_Drude.drude_box(100, 20, 50, 0.04, 0.01)
    assert np.all(w2[20:50] == 0.04) and np.all(w2[:20] == 0)
    assert np.all(ga[20:50] == 0.01)


def test_drude_invalid_courant():
    with pytest.raises(ValueError):
        FDTD1D_Drude(nx=100, courant=1.5)


# ---------------- TFSF ----------------
def test_tfsf_no_scatterer_scattered_field_near_zero():
    sim = FDTD1D_TFSF(nx=1600, i_b=400, courant=0.95, omega=0.3)
    rec = sim.run(1600, record=(300, 450))
    scat = np.sqrt(np.mean(rec["p0"][1200:]**2))
    total = np.sqrt(np.mean(rec["p1"][1200:]**2))
    assert scat / total < 0.05  # <5% 泄漏


def test_tfsf_no_scatterer_total_field_matches_incident():
    sim = FDTD1D_TFSF(nx=1600, i_b=400, courant=0.95, omega=0.3)
    idx = 450
    rec = sim.run(1600, record=(idx,))
    t = (np.arange(1200, 1600) + 0.5) * sim.dt
    e_ana = sim.e_inc(idx, t)
    seg = rec["p0"][1200:]
    assert np.sqrt(np.mean((seg - e_ana)**2)) < 0.2 * np.sqrt(np.mean(e_ana**2))


def test_tfsf_ablation_no_correction_leaks():
    full = FDTD1D_TFSF(nx=1600, i_b=400, courant=0.95, omega=0.3)
    no_h = FDTD1D_TFSF(nx=1600, i_b=400, courant=0.95, omega=0.3,
                       correct_h=False)
    rf = full.run(1600, record=(300,))["p0"][1200:]
    rn = no_h.run(1600, record=(300,))["p0"][1200:]
    assert np.sqrt(np.mean(rn**2)) > 10 * np.sqrt(np.mean(rf**2))


def test_tfsf_reflection_matches_fresnel_fine_resolution():
    n2 = 2.0
    eps = np.ones(2000)
    eps[800:] = n2**2
    sim = FDTD1D_TFSF(nx=2000, i_b=350, courant=0.95, omega=0.12, eps_r=eps)
    rec = sim.run(2800, record=(200,))
    seg = rec["p0"][2400:]
    r_meas = np.sqrt(2 * np.mean(seg**2))
    r_ana = abs((1 - n2) / (1 + n2))
    assert abs(r_meas / r_ana - 1) < 0.05


def test_tfsf_analytic_incident_plane_wave():
    sim = FDTD1D_TFSF(nx=500, i_b=200, omega=0.3, amp=2.0)
    assert sim.e_inc(0, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert sim.h_inc(10, 1.0) == -sim.e_inc(10, 1.0)  # Hy = -Ez


def test_tfsf_invalid_boundary():
    with pytest.raises(ValueError):
        FDTD1D_TFSF(nx=100, i_b=1)


# ---------------- 数值色散解析 ----------------
def test_dispersion_1d_roundtrip():
    for lodx in [10, 20, 40]:
        dx = 1e-3
        omega = 2 * np.pi * C0 / (lodx * dx)
        k = k_discrete_1d(omega, dx, 0.85)
        assert omega_from_k(k, dx, 0.85, 0.0) == pytest.approx(omega, rel=1e-10)


def test_dispersion_phase_gt_c_group_lt_c():
    dx = 1e-3
    omega = 2 * np.pi * C0 / (12 * dx)
    vp = phase_velocity_1d(omega, dx, 0.85)
    vg = group_velocity_1d(omega, dx, 0.85)
    # 1D Yee 真空离散色散（S<1）：v_ph 与 v_g 均略低于 c0
    assert C0 * 0.9 < vp < C0
    assert vg < vp < C0  # 群速度 < 相速度（正常数值色散）


def test_dispersion_phase_approaches_physical_with_fine_grid():
    dx_coarse, dx_fine = 1e-3, 2.5e-4
    omega = 2 * np.pi * C0 / (20 * dx_coarse)
    e_coarse = abs(phase_velocity_1d(omega, dx_coarse, 0.9) / C0 - 1)
    e_fine = abs(phase_velocity_1d(omega, dx_fine, 0.9) / C0 - 1)
    assert e_fine < e_coarse / 5


def test_dispersion_2d_axis_reduces_to_1d():
    dx = 1e-3
    omega = 2 * np.pi * C0 / (16 * dx)
    k1 = k_discrete_1d(omega, dx, 0.65)
    k2 = k_discrete_2d(omega, dx, 0.65, 0.0)
    assert k1 == pytest.approx(k2, rel=1e-4)


def test_dispersion_2d_anisotropy_sign_and_scaling():
    """沿轴相速度 > 对角相速度（aniso>0? 实测为负：对角更快），随 Ω² 增长。"""
    a03 = anisotropy_2d(0.3, 1.0, 0.65, 1.0)
    a08 = anisotropy_2d(0.8, 1.0, 0.65, 1.0)
    assert a03["vp_anisotropy"] < 0  # v_ph(axis) < v_ph(diag)
    assert abs(a08["vp_anisotropy"]) > abs(a03["vp_anisotropy"]) * 3


def test_dispersion_2d_no_optimal_courant_for_phase_anisotropy():
    """负结果：vp 各向异性在 S∈[0.3,1/√2] 内近似不变（无内点最优）。"""
    a = [anisotropy_2d(0.5, 1.0, S, 1.0)["vp_anisotropy"]
         for S in [0.3, 0.5, 0.65, 1 / np.sqrt(2)]]
    assert max(a) - min(a) < 1e-4  # 近常数


def test_dispersion_group_anisotropy_grows_with_courant():
    """vg 各向异性随 S 增大（与 vp 不同）：两者机制可区分。"""
    lo = anisotropy_2d(0.8, 1.0, 0.3, 1.0)["vg_anisotropy"]
    hi = anisotropy_2d(0.8, 1.0, 1 / np.sqrt(2), 1.0)["vg_anisotropy"]
    assert abs(hi) > abs(lo) * 2
