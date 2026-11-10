"""06 PML 吸收层验证（v1.1）。

v1.0 病灶：一阶 Mur 回波系数 |coef|=0.212（S=0.65）填充驻波波节，
2D 金属反射隐含 |r| 只有 0.922（1D 干净口径 0.996）。
v1.1 引入匹配有耗层（σ_m/μ0=σ_e/ε0，E/H 双场指数衰减）后重测：

1. 1D 脉冲回波：探针时窗峰值比 echo/incident（PML vs Mur）
2. 1D 包络 VSWR：PEC 墙锚点（|r|→1）vs PML 开放（|r|→0）——测量方法自证
3. 2D 驻波（PML x-lo + y Neumann + PEC 墙）：波节间距 + 波节深度隐含 |r|
"""
from __future__ import annotations

import numpy as np

from common import save_fig, save_json
from src.analysis.spectrum import rel_err
from src.fdtd_core.yee1d import FDTD1D, gaussian_derivative
from src.fdtd_core.yee2d import FDTD2D


def echo_ratio_1d(pml_cells: int, courant: float = 0.65) -> float:
    """脉冲在探针处的回波/入射峰值比（左边界开口）。

    courant=0.65 与 v1.0 的 2D 巡航数一致——该处 Mur 回波系数
    |coef|=|(S−1)/(S+1)|=0.212，对比才有意义（S=0.95 时 Mur 本身只有 0.026）。
    """
    nx = 400
    sim = FDTD1D(nx=nx, dx=1e-3, courant=courant, pml_cells=pml_cells, pml_m=3)
    src, probe = 200, 60
    rec = np.empty(1600)
    for n in range(1600):
        sim.step_h()
        sim.step_e(source=gaussian_derivative(n * sim.dt, 40 * sim.dt, 8 * sim.dt),
                   source_index=src)
        rec[n] = sim.Ez[probe]
    n_inc = int(np.argmax(np.abs(rec[:400])))
    inc = abs(rec[n_inc])
    later = rec.copy()
    later[: n_inc + 150] = 0.0
    echo = float(np.abs(later).max())
    return echo / inc if inc > 0 else float("nan")


def envelope_r_1d(kind: str, pml_cells: int = 12) -> dict:
    """时间包络 VSWR：kind='pec'（墙在右侧测量窗内）或 'pml'（开放）。"""
    nx = 400
    sim = FDTD1D(nx=nx, dx=1e-3, courant=0.95,
                 pml_cells=(pml_cells if kind == "pml" else 0), pml_m=3)
    lam_cells = 20.0
    period = lam_cells / 0.95
    n_warm, n_meas = int(30 * period), int(3 * period)
    pec = np.zeros(nx, dtype=bool)
    src = 100
    if kind == "pec":
        pec[320:] = True
        seg = slice(src + 30, 300)   # 源与墙之间
    else:
        seg = slice(20, 90)          # 源左侧行波区
    rec = []
    for n in range(n_warm + n_meas):
        sim.step_h()
        sim.step_e(source=float(np.sin(2 * np.pi * n / period)),
                   source_index=src, pec_mask=pec)
        if n >= n_warm:
            rec.append(sim.Ez[seg].copy())
    rec = np.asarray(rec)
    amp = (rec.max(axis=0) - rec.min(axis=0)) / 2.0
    a, mn = float(amp.max()), float(amp.min())
    r = (a - mn) / (a + mn) if (a + mn) > 0 else 1.0
    return {"r_envelope": r, "amp_max": a, "amp_min": mn, "n_points": int(amp.size)}


def standing_wave_2d(pml_x: int = 12) -> dict:
    """2D 驻波：行源 + PEC 墙 + PML(x-lo) + y Neumann。

    测量口径（v1.1 修正）：稳态复振幅 E(x) 在源频 DFT 提取，
    空间拟合 E(x)=A e^{−ikx}+B e^{+ikx} → r=B/A。包络深度法会被
    PML 微反射的行波分量填充波节（v1.0 教训的延续），拟合把
    行波污染归入 B/A 的实部/虚部而不影响幅度分离。
    """
    nx, ny = 220, 32
    dx = 1e-3
    src, wall = 40, 170
    lam_cells = 20.0
    sim = FDTD2D(nx=nx, ny=ny, dx=dx, courant=0.65,
                 pml_cells_x=pml_x, pml_cells_y=0, pml_m=3)
    period = lam_cells / 0.65
    n_warm, n_meas = int(35 * period), int(4 * period)
    pec = np.zeros((nx, ny), dtype=bool)
    pec[wall:, :] = True
    x_idx = np.arange(src + 12, wall - 8)
    rec = np.empty((n_meas, x_idx.size))
    for n in range(n_warm + n_meas):
        sim.step_h()
        s = float(np.sin(2 * np.pi * n / period))
        sim.step_e_line(source=s, line_source=src, pec_mask=pec, y_neumann=True)
        if n >= n_warm:
            rec[n - n_warm] = sim.Ez[x_idx, ny // 2]
    # 源频复振幅
    t = np.arange(n_meas)
    demod = np.exp(-2j * np.pi * t / period)
    E = rec.T @ demod * (2.0 / n_meas)   # (nx_seg,)
    # 波节间距（包络 |E(x)| 极小，抛物线插值）→ 实测 k（网格色散修正）
    env = np.abs(E)
    nodes = []
    for i in range(1, env.size - 1):
        if env[i] < env[i - 1] and env[i] < env[i + 1]:
            d = (env[i - 1] - env[i + 1]) / (2 * (env[i - 1] - 2 * env[i] + env[i + 1]) + 1e-30)
            nodes.append(i + float(d))
    spacing = float(np.mean(np.diff(nodes))) if len(nodes) >= 2 else float("nan")
    k = np.pi / spacing if np.isfinite(spacing) and spacing > 0 else 2 * np.pi / lam_cells
    # 拟合 A e^{-ikx} + B e^{+ikx}
    X = x_idx.astype(float)
    M = np.stack([np.exp(-1j * k * X), np.exp(1j * k * X)], axis=1)
    coef, *_ = np.linalg.lstsq(M, E, rcond=None)
    A, B = coef[0], coef[1]
    r = B / A if abs(A) > 0 else complex(np.nan, np.nan)
    a, mn = float(env.max()), float(env.min())
    return {"node_spacing_cells": spacing,
            "lambda_half_cells": lam_cells / 2.0,
            "spacing_err_pct": rel_err(spacing, lam_cells / 2.0),
            "r_abs": float(abs(r)), "r_phase_deg": float(np.degrees(np.angle(r))),
            "r_envelope_depth": (a - mn) / (a + mn) if (a + mn) > 0 else 1.0,
            "n_nodes": len(nodes), "amp_max": a, "amp_min": mn,
            "fit_residual": float(np.linalg.norm(M @ coef - E) / (np.linalg.norm(E) + 1e-30))}


def energy_residual_1d() -> float:
    """平滑脉冲注入后 1300 步的残余能量比。"""
    sim = FDTD1D(nx=300, dx=1e-3, courant=0.95, pml_cells=16)
    peak = 0.0
    for n in range(1300):
        sim.step_h()
        sim.step_e(source=float(np.exp(-((n - 30) ** 2) / (2 * 12.0**2))), source_index=150)
        e = float(np.sum(sim.Ez**2) + np.sum(sim.Hy**2))
        peak = max(peak, e)
    e = float(np.sum(sim.Ez**2) + np.sum(sim.Hy**2))
    return e / peak if peak > 0 else float("nan")


def point_source_residual_2d(pml_cells: int) -> float:
    """2D 点源脉冲，2500 步后的残余能量比（Mur 的失效域：斜入射/2D）。"""
    nx = ny = 120
    sim = FDTD2D(nx=nx, ny=ny, dx=1e-3, courant=0.65,
                 pml_cells_x=pml_cells, pml_cells_y=pml_cells, pml_m=3)
    peak = 0.0
    for n in range(2500):
        sim.step_h()
        s = float(np.exp(-((n - 40) ** 2) / (2 * 12.0**2)))
        sim.step_e(source=s, source_index=(nx // 2, ny // 2))
        e = float(np.sum(sim.Ez**2) + np.sum(sim.Hx**2) + np.sum(sim.Hy**2))
        peak = max(peak, e)
    e = float(np.sum(sim.Ez**2) + np.sum(sim.Hx**2) + np.sum(sim.Hy**2))
    return e / peak if peak > 0 else float("nan")


def main() -> None:
    import matplotlib.pyplot as plt

    print("== 06 PML validation ==")
    out: dict = {"config": {"pml_cells_1d": 16, "pml_cells_2d": 16,
                            "pml_m": 3, "sigma_max_rule": "0.8(m+1)/(eta0 dx)",
                            "damping_ratio_m_over_e": 1.0}}

    # 1. 1D 回波（S=0.65）。注意口径：一阶 Mur 在 1D 正入射近乎精确
    # （这是它残留回波小的原因），PML 的价值在 2D/斜入射与宽带吸收。
    r_pml = echo_ratio_1d(16, courant=0.65)
    r_mur = echo_ratio_1d(0, courant=0.65)
    out["echo_1d"] = {"pml": r_pml, "mur": r_mur, "courant": 0.65,
                      "mur_update_coef": abs((0.65 - 1) / (0.65 + 1)),
                      "note": "Mur 1st order ~exact at 1D normal incidence; "
                              "its failure mode is oblique/2D (see point-source test)"}
    print(f"1D echo/incident : PML={r_pml:.4f}  Mur={r_mur:.4f}")

    # 2. 包络 VSWR（方法自证）
    pec = envelope_r_1d("pec")
    pml = envelope_r_1d("pml")
    out["envelope_vswr"] = {"pec_anchor": pec, "pml_open": pml}
    print(f"envelope |r|     : PEC anchor={pec['r_envelope']:.4f}  "
          f"PML open={pml['r_envelope']:.4f}")

    # 3. 2D 驻波（复振幅拟合）
    sw = standing_wave_2d(16)
    out["standing_wave_2d"] = sw
    print(f"2D standing wave : node spacing={sw['node_spacing_cells']:.3f} cells "
          f"(lambda/2={sw['lambda_half_cells']}, err {sw['spacing_err_pct']:.2f}%), "
          f"fit |r|={sw['r_abs']:.4f} ∠{sw['r_phase_deg']:.1f}° "
          f"(envelope depth {sw['r_envelope_depth']:.4f}, resid {sw['fit_residual']:.2e})")

    # 3b. 2D 点源残余能量（Mur vs PML 的判别性口径）
    res_pml = point_source_residual_2d(16)
    res_mur = point_source_residual_2d(0)
    out["point_source_2d"] = {"pml": res_pml, "mur": res_mur}
    print(f"2D point-source residual energy: PML={res_pml:.3e}  Mur={res_mur:.3e}")

    # 4. 能量残余
    er = energy_residual_1d()
    out["energy_residual"] = er
    print(f"smooth-pulse energy residual = {er:.3e}")

    # 图
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    ax = axes[0]
    ax.bar(["PML", "Mur"], [res_pml, res_mur], color=["#1f77b4", "#ff7f0e"])
    ax.set_yscale("log")
    ax.set_title("2D point-source residual energy", fontsize=9)
    ax = axes[1]
    ax.bar(["PEC wall", "PML open"], [pec["r_envelope"], pml["r_envelope"]],
           color=["#d62728", "#2ca02c"])
    ax.set_title("envelope VSWR |r| (method anchor)", fontsize=9)
    ax = axes[2]
    ax.bar(["PML", "Mur"], [r_pml, r_mur], color=["#1f77b4", "#ff7f0e"])
    ax.set_yscale("log")
    ax.set_title("1D echo ratio (log)", fontsize=9)
    save_fig("06_pml_validation.png")
    save_json("06_pml_validation.json", out)
    print("saved: results/06_pml_validation.json + figures/06_pml_validation.png")


if __name__ == "__main__":
    main()
