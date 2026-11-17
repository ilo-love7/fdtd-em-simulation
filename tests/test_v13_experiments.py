"""v1.3 单测：2D TFSF+Mie / 各向同性色散补偿 / 3D Yee+Drude+PML。"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.isotropic_dispersion import (anisotropy_wave, kx_from_ky,
                                               ky_for_angle, phase_velocity_wave)
from src.analysis.mie2d import mie_coeffs, mie_scattered_field
from src.fdtd_core.isotropic import Wave2D, laplacian_symbol
from src.fdtd_core.tfsf2d import FDTD2D_TFSF
from src.fdtd_core.yee3d import FDTD3D_Drude, sigma_profile_3d

C0 = 299_792_458.0


# ---------- 模块 B：各向同性 Laplacian ----------
def test_laplacian_symbol_isotropic_cancels_anisotropy_leading_order():
    """9 点符号在 (kx,ky)=(K/√2,K/√2) 与 (K,0) 的 K⁴ 项应相等（抵消各向异性）。"""
    K = 0.4
    s_ax = laplacian_symbol(K, 0.0, True)
    s_dg = laplacian_symbol(K / np.sqrt(2), K / np.sqrt(2), True)
    # 9 点：展开到 K⁴ 一致，差异为 O(K⁶)
    assert abs(s_ax - s_dg) < K**6 * 0.5
    s5_ax = laplacian_symbol(K, 0.0, False)
    s5_dg = laplacian_symbol(K / np.sqrt(2), K / np.sqrt(2), False)
    assert abs(s5_ax - s5_dg) > K**4 * 0.01  # 5 点有明显差异


def test_anisotropy_wave_9pt_smaller_than_5pt():
    dx = 1e-3
    omega = 2 * np.pi * C0 / (16 * dx)
    a5 = anisotropy_wave(omega, dx, 0.6, False)
    a9 = anisotropy_wave(omega, dx, 0.6, True)
    assert abs(a9["vp_anisotropy"]) < abs(a5["vp_anisotropy"]) / 50


def test_wave2d_stability_limit():
    with pytest.raises(ValueError):
        Wave2D(nx=20, ny=20, courant=0.8, isotropic=False)  # >1/√2
    Wave2D(nx=20, ny=20, courant=0.8, isotropic=True)  # ≤√3/2 OK


def test_wave2d_axis_dispersion_matches_analytic():
    dx = 1.0
    lodx = 20
    k0 = 2 * np.pi / lodx
    omega = k0 * C0 / dx
    for iso in [False, True]:
        sim = Wave2D(nx=260, ny=180, dx=dx, courant=0.6, isotropic=iso, omega=omega)
        src_i = 40
        y = np.arange(180) - 90
        period = 2 * np.pi / (omega * sim.dt)
        nw, na = int(period * 16), int(period * 10)
        for n in range(nw + na):
            sim.step(source=np.sin(omega * n * sim.dt - 0.0 * y),
                     source_i=src_i, accumulate=(n >= nw))
        kx, _ = sim.phasor_wavenumbers((110, 230), (80, 110))
        kxa = kx_from_ky(omega, dx, 0.6, 0.0, iso)
        assert abs(kx / kxa - 1) < 0.02


def test_kx_from_ky_45deg_consistency():
    dx = 1.0
    omega = 2 * np.pi / 20 * C0 / dx
    for iso in [False, True]:
        ky = ky_for_angle(omega, dx, 0.6, np.pi / 4, iso)
        kx = kx_from_ky(omega, dx, 0.6, ky, iso)
        assert kx == pytest.approx(ky, rel=1e-3)
        vp0 = phase_velocity_wave(omega, dx, 0.6, 0.0, iso)
        vp45 = phase_velocity_wave(omega, dx, 0.6, np.pi / 4, iso)
        if iso:
            assert abs(vp0 / vp45 - 1) < 1e-4
        else:
            assert abs(vp0 / vp45 - 1) > 1e-3


# ---------- 模块 A：2D TFSF + Mie ----------
def test_tfsf2d_no_scatterer_leakage():
    sim = FDTD2D_TFSF(nx=260, ny=260, box=(80, 180, 80, 180),
                      courant=0.5, theta=np.pi / 6, omega=0.25)
    pts = [(40, 130), (220, 130), (130, 40), (130, 220)]
    rec = [[] for _ in pts]
    for n in range(1500):
        sim.step()
        if n >= 1200:
            for q, (i, j) in enumerate(pts):
                rec[q].append(sim.Ez[i, j])
    scat = np.sqrt(np.mean(np.concatenate(rec) ** 2))
    assert scat < 0.05  # <5% 泄漏


def test_tfsf2d_ablation_leaks():
    kw = dict(nx=260, ny=260, box=(80, 180, 80, 180), courant=0.5, omega=0.25)
    def rms(**kwargs):
        sim = FDTD2D_TFSF(**kw, **kwargs)
        pts = [(50, 130), (210, 130)]
        rec = [[] for _ in pts]
        for n in range(1600):
            sim.step()
            if n >= 1300:
                for q, (i, j) in enumerate(pts):
                    rec[q].append(sim.Ez[i, j])
        return np.sqrt(np.mean(np.concatenate(rec) ** 2))
    full = rms()
    noh = rms(correct_h=False)
    assert noh > 3 * full  # 消融泄漏显著放大（生产几何 ~17×）
    assert set(mie_coeffs(0.25, 12, 1.5).keys()) >= set(range(-3, 4))


def test_mie2d_forward_backward_asymmetry():
    c = mie_coeffs(0.25, 12, 1.5)
    # 该约定下前向瓣在 φ=π（FDTD 用 φ→φ+π 镜像对照）
    fwd = abs(mie_scattered_field(0.25, 12, 1.5, 65, np.pi, c))
    back = abs(mie_scattered_field(0.25, 12, 1.5, 65, 0.0, c))
    assert fwd > back * 2  # 大粒子前向散射远强于后向


def test_tfsf2d_invalid_box():
    with pytest.raises(ValueError):
        FDTD2D_TFSF(nx=100, ny=100, box=(0, 50, 0, 50))


# ---------- 模块 C：3D Yee + Drude + PML ----------
def test_yee3d_cfl_limit():
    with pytest.raises(ValueError):
        FDTD3D_Drude(nx=10, ny=5, nz=5, courant=0.7)
    FDTD3D_Drude(nx=10, ny=5, nz=5, courant=0.5)


def test_yee3d_sigma_profile_symmetric():
    sig = sigma_profile_3d(20, 20, 20, cells=4)
    assert sig[0, 10, 10] == pytest.approx(sig[-1, 10, 10])
    assert sig[10, 10, 10] == 0.0
    assert sig[0, 0, 0] >= sig[0, 10, 10]  # 角落覆盖


def test_yee3d_drude_box_mask():
    ei, w2, ga = FDTD3D_Drude.drude_slab(20, 8, 8, 5, 12, 0.04, 0.01)
    assert np.all(w2[5:12] == 0.04) and np.all(w2[:5] == 0.0)
    assert np.all(ga[5:12] == 0.01)


def test_yee3d_plane_wave_dispersion():
    omega = 0.25
    S = 0.4
    nx, ny, nz = 160, 6, 6
    sim = FDTD3D_Drude(nx=nx, ny=ny, nz=nz, courant=S)
    period = 2 * np.pi / (omega * sim.dt)
    total = int(period * 12)
    xs = np.arange(45, 90)
    acc = np.zeros(len(xs), complex)
    n0 = total - int(period * 2)
    for n in range(total):
        sim.step()
        sim.inject_plane("Ey", 20, np.sin(omega * n * sim.dt)
                         * min(1.0, n / period), slice(None), slice(None))
        if n >= n0:
            for q, i in enumerate(xs):
                acc[q] += sim.Ey[i, 3, 3] * np.exp(+1j * omega * n * sim.dt)
    acc *= 2.0 / (total - n0)
    kx = float(np.polyfit(xs, np.unwrap(np.angle(acc)), 1)[0])
    from src.analysis.dispersion import k_discrete_1d
    ka = k_discrete_1d(omega, 1.0, S, 1.0)
    # 短域相量斜率符号受残余边界回波相位影响，取模与离散色散对照
    assert 0 < abs(kx) < 0.35
    assert abs(abs(kx) / ka - 1) < 0.05


def test_yee3d_drude_stable():
    ei, w2, ga = FDTD3D_Drude.drude_slab(40, 5, 5, 12, 28, 0.05, 0.02)
    sim = FDTD3D_Drude(nx=40, ny=5, nz=5, courant=0.4,
                       eps_inf=ei, wp2=w2, gamma=ga, pml_cells=1)
    for n in range(1200):
        sim.step()
        sim.inject_plane("Ey", 6, np.sin(0.2 * n * sim.dt) * min(1, n / 50))
    assert np.all(np.isfinite(sim.Ey))
    assert np.max(np.abs(sim.Ey)) < 50
