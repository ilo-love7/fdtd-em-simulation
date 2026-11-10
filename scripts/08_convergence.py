"""08 收敛阶研究（v1.1）。

1. 1D Fresnel 界面反射误差 vs dx：完全复刻 v1.0 01 的几何/脉冲/窗口配方
   （物理域固定，步数与窗口随 dx 等比缩放），拟合误差斜率。
2. 1D 数值色散：定频 CW 空间谱（抛物线插值谱峰）→ k_num，
   对照离散色散关系解析解（应机器吻合）与物理波数 k0（误差 ~ (dx/λ)²）。
3. 2D 单缝第一极小 spot check（两级分辨率；暖机 ≥ 波传播路程）。
"""
from __future__ import annotations

import numpy as np

from common import save_fig, save_json
from src.analysis.spectrum import fresnel_r, rel_err, windowed_fft_ratio
from src.fdtd_core.yee1d import C0, FDTD1D, gaussian_derivative
from src.fdtd_core.yee2d import FDTD2D

# v1.0 01 基准配方（dx=1e-3）：域 0.6 m，源 0.05，探针 0.2，界面 0.3
REF_DX = 1e-3


def fresnel_error(dx: float) -> float:
    """n=2 界面反射 |r| 相对解析的误差（%）。几何固定、步数随 dx 缩放。"""
    n2 = 2.0
    s = REF_DX / dx          # 缩放因子
    nx = int(round(0.6 / dx))
    src = int(round(0.05 / dx))
    probe = int(round(0.2 / dx))
    i_wall = int(round(0.3 / dx))
    eps = np.ones(nx)
    eps[i_wall:] = n2**2
    sim = FDTD1D(nx=nx, dx=dx, courant=0.95, eps_r=eps)
    t0 = 30.0 * s            # 物理脉冲形状固定
    w = 8.0 * s
    n_steps = int(900 * s)
    rec = np.empty(n_steps)
    for n in range(n_steps):
        sim.step_h()
        sim.step_e(source=gaussian_derivative(n * sim.dt, t0 * sim.dt, w * sim.dt),
                   source_index=src)
        rec[n] = sim.Ez[probe]
    # 窗口按 v1.0 手工区间等比缩放
    i_inc = int(np.argmax(np.abs(rec[int(100 * s):int(280 * s)])) + int(100 * s))
    i_ref = int(np.argmax(np.abs(rec[int(320 * s):int(480 * s)])) + int(320 * s))
    res = windowed_fft_ratio(np.arange(n_steps) * sim.dt, rec, rec,
                             i_inc, i_ref, half_width=int(60 * s))
    r_theory = fresnel_r(1.0, n2)
    return rel_err(float(np.real(res["r_complex"])), r_theory)


def dispersion_error(dx: float, lam_over_dx: float) -> dict:
    """定频 CW 空间谱（抛物线插值）测 k_num。"""
    lam = lam_over_dx * dx
    f = C0 / lam
    omega = 2 * np.pi * f
    nx = 1600
    sim = FDTD1D(nx=nx, dx=dx, courant=0.95, pml_cells=16, pml_m=3)
    period = (1.0 / f) / sim.dt
    src = 400
    seg_lo, seg_hi = 600, 1300
    # 暖机必须覆盖波前到达谱窗末端 + 若干周期
    n_warm = int((seg_hi - src) / 0.95 + 12 * period)
    for n in range(n_warm + 1):
        sim.step_h()
        sim.step_e(source=float(np.sin(2 * np.pi * n / period)), source_index=src)
    seg = sim.Ez[seg_lo:seg_hi].copy()
    win = np.hanning(seg.size)
    spec = np.abs(np.fft.rfft(seg * win))
    ks = 2 * np.pi * np.fft.rfftfreq(seg.size, d=dx)
    k_i = int(np.argmax(spec[1:])) + 1
    # 抛物线插值谱峰
    a0, a1, a2 = spec[k_i - 1], spec[k_i], spec[k_i + 1]
    denom = (a0 - 2 * a1 + a2)
    delta = 0.5 * (a0 - a2) / denom if abs(denom) > 0 else 0.0
    k_num = float(ks[k_i] + delta * (ks[1] - ks[0]))
    s = np.sin(omega * sim.dt / 2) / (C0 * sim.dt / 2)
    k_exact = 2 * np.arcsin(np.clip(s * dx / 2, -1, 1)) / dx
    k0 = omega / C0
    return {"dx": dx, "lam_over_dx": lam_over_dx,
            "k_num": k_num, "k_discrete_exact": float(k_exact), "k0": float(k0),
            "err_vs_discrete_pct": rel_err(k_num, float(k_exact)),
            "err_vs_physical_pct": rel_err(k_num, float(k0))}


def slit_profile(dx_frac: float):
    """2D 单缝角分布（返回 sinθ 网格与归一化幅度）。"""
    lam = 1.0
    a_slit = 2.5 * lam
    dx = dx_frac * lam
    nx = int(round(70 * lam / dx))
    ny = int(round(56 * lam / dx))
    sim = FDTD2D(nx=nx, ny=ny, dx=dx, courant=0.65,
                 pml_cells_x=10, pml_cells_y=10, pml_m=3)
    slit0 = int(round(ny / 2 - a_slit / (2 * dx)))
    slit1 = int(round(ny / 2 + a_slit / (2 * dx)))
    x_scr = int(round(22 * lam / dx))
    pec = sim.slit_screen(nx, ny, x_scr, max(2, int(round(0.3 * lam / dx))), slit0, slit1)
    period = (lam / dx) / 0.65
    src_x = int(round(8 * lam / dx))
    R = 25 * lam
    dist = (8 + 22 + 25) * lam
    n_warm = int((dist / lam + 8) * period)
    n_meas = int(2 * period)
    amp_hi = np.full((nx, ny), -np.inf)
    amp_lo = np.full((nx, ny), np.inf)
    for n in range(n_warm + n_meas):
        sim.step_h()
        s = float(np.sin(2 * np.pi * n / period))
        sim.step_e_line(source=s, line_source=src_x, pec_mask=pec)
        if n >= n_warm:
            np.maximum(amp_hi, sim.Ez, out=amp_hi)
            np.minimum(amp_lo, sim.Ez, out=amp_lo)
    amp = (amp_hi - amp_lo) / 2
    cx, cy = x_scr * dx, (ny / 2) * dx
    th = np.linspace(-1.1, 1.1, 221)
    vals = []
    for t in th:
        x = cx + R * np.cos(t)
        y = cy + R * np.sin(t)
        i, j = int(round(x / dx)), int(round(y / dx))
        vals.append(amp[i, j] if 1 <= i < nx - 1 and 1 <= j < ny - 1 else np.nan)
    vals = np.asarray(vals)
    ok = np.isfinite(vals)
    th, vals = th[ok], vals[ok]
    v = vals / np.nanmax(vals)
    sin_th = np.sin(th)
    order = np.argsort(sin_th)
    return sin_th[order], v[order]


def slit_spot(sin_th, v) -> float:
    """第一极小 sinθ 误差（Fraunhofer 口径，如实标注 a=2.5λ 的近场偏差）。"""
    target = 1.0 / 2.5
    lo, hi = target - 0.12, target + 0.12
    m = (sin_th > lo) & (sin_th < hi)
    st, vv = sin_th[m], v[m]
    return rel_err(float(st[int(np.argmin(vv))]), target)


def main() -> None:
    import matplotlib.pyplot as plt

    print("== 08 convergence ==")
    out: dict = {}

    # 1. Fresnel 误差 vs dx
    dxs = [2e-3, 1e-3, 5e-4]
    errs = []
    for dx in dxs:
        e = fresnel_error(dx)
        errs.append(e)
        print(f"Fresnel dx={dx:.4e}  err={e:.4f}%")
    errs = np.asarray(errs)
    good = np.isfinite(errs) & (errs > 0)
    p = (float(np.polyfit(np.log(np.asarray(dxs)[good]), np.log(errs[good]), 1)[0])
         if good.sum() >= 2 else float("nan"))
    out["fresnel"] = {"dx": dxs, "err_pct": errs.tolist(), "observed_order": p}

    # 2. 数值色散
    disp = [dispersion_error(1e-3, 8.0), dispersion_error(1e-3, 12.0),
            dispersion_error(1e-3, 20.0), dispersion_error(5e-4, 20.0)]
    out["dispersion"] = disp
    for d in disp:
        print(f"dispersion lam/dx={d['lam_over_dx']:.0f} dx={d['dx']:.2e}: "
              f"vs discrete {d['err_vs_discrete_pct']:.4f}%, "
              f"vs physical {d['err_vs_physical_pct']:.4f}%")

    # 3. 2D slit：Fraunhofer 极小偏差（如实标注近场偏移）+ 双分辨率自收敛 RMS
    st1, v1 = slit_profile(1 / 10)
    st2, v2 = slit_profile(1 / 20)
    v2i = np.interp(st1, st2, v2)
    self_rms = float(np.sqrt(np.mean((v1 - v2i) ** 2)))
    s1 = slit_spot(st1, v1)
    s2 = slit_spot(st2, v2)
    out["slit_spot"] = {"dx_lam_1_10_err_pct": s1, "dx_lam_1_20_err_pct": s2,
                        "self_convergence_rms": self_rms,
                        "note": "Fraunhofer sinθ=λ/a 在 a=2.5λ 有固有近场偏移"
                                "（v1.0 单一分辨率亦偏 5%）；收敛性以自收敛 RMS 为准"}
    print(f"2D slit first-min err: dx=λ/10 → {s1:.3f}%, dx=λ/20 → {s2:.3f}%; "
          f"self-convergence RMS = {self_rms:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    ax = axes[0]
    ax.loglog(dxs, np.maximum(errs, 1e-9), "o-", label="measured")
    if good.any():
        ref_line = errs[good][0] * (np.asarray(dxs) / dxs[0]) ** 2
        ax.loglog(dxs, ref_line, "k--", label="O(dx^2)")
    ax.set_xlabel("dx [m]"); ax.set_ylabel("Fresnel |r| err [%]")
    ax.set_title(f"interface error (order≈{p:.2f})", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1]
    labels = [f"λ/{int(d['lam_over_dx'])}" for d in disp]
    ax.bar(labels, [d["err_vs_physical_pct"] for d in disp], color="#1f77b4")
    ax.set_title("dispersion err vs k0 [%]", fontsize=9)
    ax = axes[2]
    ax.bar(["λ/10", "λ/20"], [s1, s2], color="#ff7f0e")
    ax.set_title("2D slit first-min err [%]", fontsize=9)
    save_fig("08_convergence.png")
    save_json("08_convergence.json", out)
    print("saved: results/08_convergence.json + figures/08_convergence.png")


if __name__ == "__main__":
    main()
