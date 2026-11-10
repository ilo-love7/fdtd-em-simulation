"""03 二维 FDTD：单缝衍射（金属屏 + 缝）。

平面波入射 PEC 屏（缝宽 a），在半圆弧上测 |Ez| 角分布，
与 Fraunhofer 单缝公式 I(θ)=sinc²(π a sinθ/λ) 对比，验证极小值位置 sinθ=mλ/a。
"""
from __future__ import annotations

import numpy as np

from common import save_fig, save_json
from src.analysis.spectrum import rel_err, single_slit_intensity, single_slit_minima
from src.fdtd_core.yee2d import C0, FDTD2D

NX, NY = 260, 200
DX = 1e-3
COURANT = 0.65
SRC = 25
SCREEN_X = 70
SCREEN_TH = 2
LAM_CELLS = 16
SLIT_HALF = 20          # 缝半宽（格）→ a = 40 格
SLIT_C = NY // 2
ARC_R = 85.0            # 弧半径（格）


def main() -> None:
    import matplotlib.pyplot as plt

    sim = FDTD2D(nx=NX, ny=NY, dx=DX, courant=COURANT)
    pec = FDTD2D.slit_screen(NX, NY, SCREEN_X, SCREEN_TH,
                             SLIT_C - SLIT_HALF, SLIT_C + SLIT_HALF)
    period_steps = LAM_CELLS / COURANT
    n_total = 1800
    n_ramp = int(5 * period_steps)

    def src_fn(n: int) -> float:
        ramp = min(1.0, n / n_ramp) if n_ramp > 0 else 1.0
        return float(ramp * np.sin(2 * np.pi * n / period_steps))

    # 弧上采样点（以缝中心为圆心，屏幕下游侧）
    thetas = np.linspace(-np.pi / 2 * 0.92, np.pi / 2 * 0.92, 121)
    ax_c = SCREEN_X + SCREEN_TH + 1
    ay_c = SLIT_C
    arc_i = np.clip((ax_c + ARC_R * np.cos(thetas)).astype(int), 1, NX - 2)
    arc_j = np.clip((ay_c + ARC_R * np.sin(thetas)).astype(int), 1, NY - 2)
    arc_max = np.zeros(thetas.size)

    for n in range(n_total):
        sim.step_h()
        sim.step_e_line(source=src_fn(n), line_source=SRC, pec_mask=pec)
        if n > n_total - 400:
            arc_max = np.maximum(arc_max, np.abs(sim.Ez[arc_i, arc_j]))

    I_sim = arc_max**2
    I_sim = I_sim / I_sim.max() if I_sim.max() > 0 else I_sim
    a_over_lam = (2 * SLIT_HALF) / LAM_CELLS
    I_th = single_slit_intensity(np.sin(thetas), a_over_lam)

    # 极小值位置（sim：弧上 |E| 局部极小；theory：mλ/a）
    from scipy.signal import argrelextrema
    imin = argrelextrema(arc_max, np.less_equal, order=4)[0]
    # 只取 θ>0 一侧
    imin_pos = imin[thetas[imin] > 0.05]
    sim_sin = np.sin(thetas[imin_pos])
    th_sin = single_slit_minima(a_over_lam, m_max=4)

    min_errs = []
    for s_th in th_sin:
        if sim_sin.size:
            d = np.min(np.abs(sim_sin - s_th))
            min_errs.append(float(d))
    min_err_mean = float(np.mean(min_errs)) if min_errs else float("nan")

    out = {
        "config": {"nx": NX, "ny": NY, "dx_m": DX, "courant": COURANT,
                   "src": SRC, "screen_x": SCREEN_X, "lam_cells": LAM_CELLS,
                   "slit_half_cells": SLIT_HALF, "arc_R_cells": ARC_R},
        "lam_m": LAM_CELLS * DX,
        "slit_width_m": 2 * SLIT_HALF * DX,
        "a_over_lambda": a_over_lam,
        "theory_minima_sin_theta": th_sin,
        "sim_minima_sin_theta": sim_sin,
        "minima_mean_abs_err_sin_theta": min_err_mean,
        "pattern_rms_err_vs_theory": float(np.sqrt(np.mean((I_sim - I_th) ** 2))),
    }
    print(f"[03] a/λ={a_over_lam:.2f}  theory minima sinθ={th_sin}  sim={np.round(sim_sin, 4)}")
    print(f"[03] minima |Δsinθ| mean={min_err_mean:.4f}  pattern RMS={out['pattern_rms_err_vs_theory']:.4f}")

    # 图
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    ax = axes[0, 0]
    im = ax.imshow(sim.Ez.T, aspect='auto', origin='lower', cmap='RdBu_r',
                   extent=[0, NX * DX * 1e3, 0, NY * DX * 1e3])
    ax.plot((arc_i * DX) * 1e3, (arc_j * DX) * 1e3, 'k.', ms=1, alpha=0.3, label='arc')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title(f'Slit diffraction (a={2*SLIT_HALF} cells, λ={LAM_CELLS} cells)')
    ax.legend(loc='upper right')
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    ax.plot(np.degrees(thetas), I_sim, 'o-', ms=3, lw=1, label='FDTD |E|² (arc)')
    ax.plot(np.degrees(thetas), I_th, '-', lw=1.2, label='Fraunhofer sinc²')
    for s in th_sin:
        ax.axvline(np.degrees(np.arcsin(min(s, 1.0))), color='r', ls=':', lw=0.8)
    ax.set_xlabel('θ (deg)'); ax.set_ylabel('I / I0')
    ax.set_title(f'Angular pattern (minima err Δsinθ={min_err_mean:.4f})')
    ax.legend()

    ax = axes[1, 0]
    ax.plot(np.sin(thetas), I_sim, 'o-', ms=3, lw=1, label='FDTD')
    ax.plot(np.sin(thetas), I_th, '-', lw=1.2, label='theory')
    for s in th_sin:
        ax.axvline(s, color='r', ls=':', lw=0.8)
    ax.set_xlabel('sin θ'); ax.set_ylabel('I / I0')
    ax.set_title('Pattern vs sinθ (red: theory minima)')
    ax.legend()

    ax = axes[1, 1]
    ax.axis('off')
    ax.text(0.02, 0.95, f"a/λ = {a_over_lam:.3f}", va='top', fontsize=9)
    ax.text(0.02, 0.85, f"theory minima sinθ = {np.round(th_sin, 4)}", va='top', fontsize=8)
    ax.text(0.02, 0.75, f"sim minima sinθ   = {np.round(sim_sin, 4)}", va='top', fontsize=8)
    ax.text(0.02, 0.62, f"mean |Δsinθ| = {min_err_mean:.4f}", va='top', fontsize=8)
    ax.text(0.02, 0.52, f"pattern RMS err = {out['pattern_rms_err_vs_theory']:.4f}", va='top', fontsize=8)
    save_fig("03_2d_slit_diffraction.png")

    save_json("03_2d_slit_diffraction.json", out)
    print("[03] done -> 03_2d_slit_diffraction.json")


if __name__ == "__main__":
    main()
