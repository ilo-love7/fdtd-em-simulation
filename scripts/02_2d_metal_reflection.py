"""02 二维 FDTD：金属板反射与驻波验证。

平面波（行源，y Neumann 边界保持 y 均匀）正入射 PEC 墙 → 驻波。
定量：波节间距 = λ/2（抛物线插值）。
半定量：波节深度 → 隐含 |r|；残余填充与一阶 Mur 回波系数 |coef| 定量对照（诚实标注）。
（干净的 |r| 定量见 01 的 1D FFT 时窗法：|r|=0.996。）
"""
from __future__ import annotations

import numpy as np

from common import save_fig, save_json
from src.analysis.spectrum import rel_err
from src.fdtd_core.yee2d import C0, FDTD2D

NX, NY = 220, 32
DX = 1e-3
COURANT = 0.65
SRC = 40
WALL = 170
LAM_CELLS = 20


def main() -> None:
    import matplotlib.pyplot as plt
    from scipy.signal import argrelextrema

    sim = FDTD2D(nx=NX, ny=NY, dx=DX, courant=COURANT)
    pec = np.zeros((NX, NY), dtype=bool)
    pec[WALL, :] = True
    period_steps = LAM_CELLS / COURANT
    n_ramp = int(5 * period_steps)
    n_total = 1200

    def src_fn(n: int) -> float:
        ramp = min(1.0, n / n_ramp) if n_ramp > 0 else 1.0
        return float(ramp * np.sin(2 * np.pi * n / period_steps))

    prof_max = np.zeros(NX)
    for n in range(n_total):
        sim.step_h()
        sim.step_e_line(source=src_fn(n), line_source=SRC, pec_mask=pec, y_neumann=True)
        if n > n_total - 200:
            prof_max = np.maximum(prof_max, np.abs(sim.Ez[:, NY // 2]))

    x = np.arange(NX) * DX
    i0, i1 = SRC + 3 * LAM_CELLS, WALL - int(1.5 * LAM_CELLS)
    seg = prof_max[i0:i1]
    xs = x[i0:i1]
    depth = float(seg.min() / seg.max()) if seg.max() > 0 else float("nan")
    imin = argrelextrema(seg, np.less_equal, order=3)[0]
    node_x = []
    for k in imin:
        if 0 < k < seg.size - 1:
            y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
            denom = y0 - 2 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-30 else 0.0
            node_x.append(xs[k] + delta * DX)
    node_x = np.array(node_x)
    spacing = float(np.mean(np.diff(node_x) / DX)) if node_x.size >= 2 else float("nan")
    imax = argrelextrema(seg, np.greater_equal, order=3)[0]

    # 一阶 Mur 对正入射平面波的回波系数（源的左行波在 x=0 边界的反射，污染驻波）
    mur_r = abs((C0 * sim.dt - DX) / (C0 * sim.dt + DX))
    out = {
        "config": {"nx": NX, "ny": NY, "dx_m": DX, "courant": COURANT,
                   "src": SRC, "wall": WALL, "lam_cells": LAM_CELLS, "y_bc": "neumann"},
        "lam_m": LAM_CELLS * DX,
        "period_steps": period_steps,
        "node_spacing_cells_sim": spacing,
        "node_spacing_cells_theory": LAM_CELLS / 2,
        "node_spacing_err_pct": rel_err(spacing, LAM_CELLS / 2),
        "node_depth_ratio": depth,
        "implied_abs_r_vswr": (1.0 - depth) / (1.0 + depth),
        "mur1_backreflection_coef": mur_r,
        "depth_explained_by_mur": (
            f"波节填充 ~ 源左行波经 x=0 一阶 Mur 边界回波 |coef|={mur_r:.3f} 的相干叠加；"
            "墙本身 r=-1（1D FFT 法已定 |r|=0.996）"
        ),
        "n_nodes": int(imin.size), "n_antinodes": int(imax.size),
    }
    print(f"[02] node spacing {spacing:.3f} cells (theory {LAM_CELLS/2})  err={out['node_spacing_err_pct']:.3f}%")
    print(f"[02] node depth {depth:.4f}  implied |r|={out['implied_abs_r_vswr']:.3f}  "
          f"(Mur 回波 coef={mur_r:.3f} 可解释填充)")

    # 图
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    ax = axes[0]
    im = ax.imshow(sim.Ez.T, aspect='auto', origin='lower', cmap='RdBu_r',
                   extent=[0, NX * DX * 1e3, 0, NY * DX * 1e3])
    ax.axvline(WALL * DX * 1e3, color='k', lw=2, label='PEC wall')
    ax.axvline(SRC * DX * 1e3, color='g', lw=1, ls='--', label='source')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title('2D plane wave reflecting off PEC wall (Ez, y-Neumann BC)')
    ax.legend(loc='upper right')
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    ax.plot(xs * 1e3, seg, lw=1.2, label='|Ez| envelope')
    if node_x.size:
        ax.plot(node_x * 1e3, np.interp(node_x, x, prof_max), 'v', ms=5,
                label=f'nodes (depth={depth:.3f}, spacing={spacing:.2f} cells)')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('|Ez|')
    ax.set_title(f'Standing wave: node spacing {spacing:.3f} vs {LAM_CELLS/2} cells '
                 f'({out["node_spacing_err_pct"]:.2f}%)')
    ax.legend()
    save_fig("02_2d_metal_reflection.png")

    save_json("02_2d_metal_reflection.json", out)
    print("[02] done -> 02_2d_metal_reflection.json")


if __name__ == "__main__":
    main()
