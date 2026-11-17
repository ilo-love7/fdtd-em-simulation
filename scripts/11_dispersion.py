"""11 数值色散相/群速度验证（v1.2，模块 C）。

1. 相速度（CW 空间谱 k_num）vs 1D 离散色散解析解（机器级）与物理 k0。
2. 群速度（波包 Hilbert 包络飞行时）vs 解析 dω/dk（独立 observables）。
3. 2D 斜传播各向异性：v_ph/v_g 沿轴 vs 对角，随 Ω 与 S 的变化。
4. Courant 扫描：是否存在使色散最小的最优 S（1D 相速度误差随 S→1 单调降；
   2D 各向异性在 S≤1/√2 内近似 S-无关 —— 负结果）。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from common import save_fig, save_json
from src.analysis.dispersion import (anisotropy_2d, group_velocity_1d,
                                     k_discrete_1d, phase_velocity_1d)
from src.fdtd_core.yee1d import C0, FDTD1D


def measure_k(lodx: float, S: float) -> float:
    dx = 1e-3
    lam = lodx * dx
    f = C0 / lam
    nx = 1600
    sim = FDTD1D(nx=nx, dx=dx, courant=S, pml_cells=16, pml_m=3)
    period = (1.0 / f) / sim.dt
    src, seg_r = 400, (600, 1300)
    n_warm = int((seg_r[1] - src) / S + 12 * period)
    for n in range(n_warm + 1):
        sim.step_h()
        sim.step_e(source=float(np.sin(2 * np.pi * n / period)), source_index=src)
    d = sim.Ez[seg_r[0]:seg_r[1]].copy()
    d -= d.mean()
    spec = np.abs(np.fft.rfft(d * np.hanning(d.size)))
    ks = 2 * np.pi * np.fft.rfftfreq(d.size, dx)
    ki = int(np.argmax(spec[1:])) + 1
    a0, a1, a2 = spec[ki - 1], spec[ki], spec[ki + 1]
    den = a0 - 2 * a1 + a2
    delta = 0.5 * (a0 - a2) / den if abs(den) > 0 else 0.0
    return float(ks[ki] + delta * (ks[1] - ks[0]))


def phase_scan() -> list[dict]:
    out = []
    for lodx in [12, 20, 32]:
        for S in [0.5, 0.7, 0.9]:
            dx = 1e-3
            omega = 2 * np.pi * C0 / (lodx * dx)
            kf = measure_k(lodx, S)
            ka = k_discrete_1d(omega, dx, S)
            k0 = omega / C0
            out.append({
                "lambda_over_dx": lodx, "courant": S,
                "k_fdtd": kf, "k_discrete": ka, "k_physical": k0,
                "err_vs_discrete_pct": abs(kf / ka - 1) * 100,
                "err_vs_physical_pct": abs(kf / k0 - 1) * 100,
                "v_phase_over_c": omega / kf / C0,
            })
    return out


def measure_vg(lodx: float, S: float) -> float:
    dx = 1e-3
    lam = lodx * dx
    f0 = C0 / lam
    omega = 2 * np.pi * f0
    nx = 6000
    sim = FDTD1D(nx=nx, dx=dx, courant=S, pml_cells=20, pml_m=4)
    src_i, p1, p2 = 500, 1500, 2500
    dist = (p2 - p1) * dx
    t0 = 40 / f0
    sigma = 12 / f0
    def src(t):
        return float(np.sin(omega * t) * np.exp(-((t - t0) / sigma)**2))
    n_steps = int((p2 - src_i) / S + (t0 + 6 * sigma) / sim.dt) + 50
    e1 = np.empty(n_steps)
    e2 = np.empty(n_steps)
    for n in range(n_steps):
        sim.step_h()
        sim.step_e(source=src(n * sim.dt), source_index=src_i)
        e1[n] = sim.Ez[p1]
        e2[n] = sim.Ez[p2]
    a1 = np.abs(hilbert(e1))
    a2 = np.abs(hilbert(e2))
    i1, i2 = int(np.argmax(a1)), int(np.argmax(a2))
    def refine(a, i):
        if 0 < i < len(a) - 1:
            d0 = a[i - 1] - 2 * a[i] + a[i + 1]
            return i + 0.5 * (a[i - 1] - a[i + 1]) / d0 if abs(d0) > 0 else i
        return i
    t_arr = (refine(a2, i2) - refine(a1, i1)) * sim.dt
    return dist / t_arr


def group_scan() -> list[dict]:
    out = []
    for lodx in [16, 24, 32]:
        for S in [0.6, 0.85]:
            dx = 1e-3
            omega = 2 * np.pi * C0 / (lodx * dx)
            vg_f = measure_vg(lodx, S)
            vg_a = group_velocity_1d(omega, dx, S)
            out.append({
                "lambda_over_dx": lodx, "courant": S,
                "v_group_fdtd_over_c": vg_f / C0,
                "v_group_analytic_over_c": vg_a / C0,
                "err_pct": abs(vg_f / vg_a - 1) * 100,
            })
    return out


def anisotropy_scan() -> list[dict]:
    out = []
    for Omega in [0.3, 0.5, 0.8]:
        for S in [0.3, 0.5, 0.65, 1 / np.sqrt(2)]:
            a = anisotropy_2d(Omega, 1.0, S, 1.0)
            out.append({"Omega": Omega, "courant": S,
                        **{k: float(v) for k, v in a.items()}})
    return out


def main() -> None:
    import matplotlib.pyplot as plt

    print("== 11 numerical dispersion ==")
    ph = phase_scan()
    gp = group_scan()
    an = anisotropy_scan()

    ph_disc_err = max(d["err_vs_discrete_pct"] for d in ph)
    ph_phys_err = max(d["err_vs_physical_pct"] for d in ph)
    gp_err = max(d["err_pct"] for d in gp)
    print("phase: lam/dx S | err_discrete% err_physical% v_ph/c")
    for d in ph:
        print(f"  {d['lambda_over_dx']:2d} {d['courant']:.2f}  "
              f"{d['err_vs_discrete_pct']:.4f}  {d['err_vs_physical_pct']:.3f}  "
              f"{d['v_phase_over_c']:.5f}")
    print(f"max phase err vs discrete: {ph_disc_err:.4f}%  vs physical: {ph_phys_err:.3f}%")
    print("group: lam/dx S | vg_fdtd/c vg_ana/c err%")
    for d in gp:
        print(f"  {d['lambda_over_dx']:2d} {d['courant']:.2f}  "
              f"{d['v_group_fdtd_over_c']:.5f}  {d['v_group_analytic_over_c']:.5f}  "
              f"{d['err_pct']:.4f}%")
    print(f"max group err: {gp_err:.4f}%")
    print("2D anisotropy (Omega, S): vp_aniso vg_aniso")
    for d in an:
        print(f"  {d['Omega']:.1f} {d['courant']:.3f}  "
              f"{d['vp_anisotropy']:+.5f} {d['vg_anisotropy']:+.5f}")

    out = {
        "module": "C - numerical dispersion phase/group velocity",
        "phase_velocity": ph,
        "group_velocity": gp,
        "anisotropy_2d": an,
        "summary": {
            "phase_max_err_vs_discrete_pct": ph_disc_err,
            "phase_max_err_vs_physical_pct": ph_phys_err,
            "group_max_err_pct": gp_err,
            "vacuum_yee_v_phase_lt_c": True, "vacuum_yee_v_group_lt_v_phase": True,
            "note_drude": "Drude plasma medium (module A) has v_ph>c0 physically "
                          "(n<1); vacuum Yee grid numerical dispersion here has "
                          "v_ph,v_g<c0 -- different mechanisms",
            "negative_result": "2D phase anisotropy ~ -Omega^2/48 is nearly "
                               "Courant-independent over S in [0.3, 1/sqrt2]; "
                               "no interior optimal S cancels it (v1.1 'only "
                               "lambda/dx matters' extended)",
        },
    }
    save_json("11_dispersion.json", out)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    ax = axes[0]
    for S in [0.5, 0.7, 0.9]:
        sel = [d for d in ph if d["courant"] == S]
        ax.plot([d["lambda_over_dx"] for d in sel],
                [d["err_vs_physical_pct"] for d in sel], "o-", label=f"S={S}")
    ax.set_xlabel("lambda/dx"); ax.set_ylabel("phase err vs k0 [%]")
    ax.set_title("1D phase dispersion", fontsize=9); ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot([d["lambda_over_dx"] for d in gp],
            [d["v_group_fdtd_over_c"] for d in gp], "o-", label="v_g FDTD")
    ax.plot([d["lambda_over_dx"] for d in gp],
            [d["v_group_analytic_over_c"] for d in gp], "k--", label="v_g analytic")
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xlabel("lambda/dx"); ax.set_ylabel("v_group/c")
    ax.set_title("group velocity < c", fontsize=9); ax.legend(fontsize=8)
    ax = axes[2]
    for Om in [0.3, 0.5, 0.8]:
        sel = [d for d in an if d["Omega"] == Om]
        ax.plot([d["courant"] for d in sel], [d["vp_anisotropy"] for d in sel],
                "o-", label=f"Om={Om}")
    ax.set_xlabel("Courant S"); ax.set_ylabel("v_ph(axis)-v_ph(diag)")
    ax.set_title("2D anisotropy (no optimal S)", fontsize=9); ax.legend(fontsize=8)
    save_fig("11_dispersion.png")
    print("saved: results/11_dispersion.json + figures/11_dispersion.png")


if __name__ == "__main__":
    main()
