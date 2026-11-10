"""fdtd（flaport/fdtd 0.3.5）库封装：一维 Bragg 反射镜透射谱。

方法：逐波长 CW 正弦激励（库的 pulse=False 分支，周期以**整数步数**传入
—— float 会被 `_handle_time` 当秒换算，这是库的坑），稳态幅值比 → T(λ)。

库边界情况（如实记录）：
- 单点 LineSource 的 profile 因 vect.max()=0 除零得 NaN → 用 ≥2 点线源
  并把 profile 手工置为均匀（保持 y 均匀性）。
- 驱动分量是 E[z]（分量下标 2），探测器记录 (Ex,Ey,Ez)。
"""
from __future__ import annotations

import numpy as np


def build_bragg_grid(
    n_layers: int = 8,
    lam0_design_m: float = 1000e-9,
    n1: float = 1.5,
    n2: float = 2.5,
    dx: float = 5e-9,
    pad_cells: int = 60,
    lam_run_m: float | None = None,
):
    """构造 Bragg 镜 fdtd 网格（四分之一波长堆栈，y 方向 3 格均匀）。

    lam_run_m: 实际激励波长（周期换算用）；None 则用 lam0_design_m。
    """
    import fdtd as fd

    lam_src = lam_run_m if lam_run_m is not None else lam0_design_m
    d1 = lam0_design_m / (4.0 * n1)
    d2 = lam0_design_m / (4.0 * n2)
    n1_cells = max(1, int(round(d1 / dx)))
    n2_cells = max(1, int(round(d2 / dx)))
    stack_cells = n_layers * (n1_cells + n2_cells)
    total = pad_cells * 2 + stack_cells
    ny = 3

    # y 均匀的相对介电常数（3 列相同）
    eps = np.ones((1, total, ny))
    x0 = pad_cells
    for _ in range(n_layers):
        eps[0, x0:x0 + n1_cells, :] = n1**2
        x0 += n1_cells
        eps[0, x0:x0 + n2_cells, :] = n2**2
        x0 += n2_cells

    # 几何约定：形状 (Nx=1, Ny=total, Nz=3)，传播沿 y（层堆叠方向），z 为横向
    grid = fd.Grid(shape=(1, total, ny), grid_spacing=dx)
    # y 向 PML 吸收层（传播方向两端；无 PML 时端面形成谐振腔，透射测量失真）
    grid[:, 0:8, :] = fd.PML()
    grid[:, -8:, :] = fd.PML()
    # Object 只铺非 PML 区（铺满会覆盖 PML 单元，破坏吸收）
    grid[0, 8:-8, :] = fd.Object(permittivity=eps[:, 8:-8, :], name="stack")
    # 库的「对称化」把 object 的 z=-1 行 Ez 逆介电常数重置为背景 → 波从该行绕过堆栈。
    # 手工恢复：整列逆介电常数按 eps 重建（含 z=-1 行）。
    obj = grid.objects[0]
    eps_obj = np.asarray(obj.permittivity, dtype=float)   # (1,Ny,3,1)
    obj.inverse_permittivity = np.ones((obj.Nx, obj.Ny, obj.Nz, 3)) / eps_obj

    src_x = pad_cells // 2
    det_x = total - pad_cells // 2
    dt = grid.time_step
    period_steps = int(round(lam_src / fd.constants.c / dt))  # 整数步数（库约定）
    grid[0, src_x, :] = fd.LineSource(period=period_steps, name="src")  # 定 y，跨 z 3 点
    # 修复单线源的高斯 apodization：置均匀 profile 保持 z 均匀场
    grid.sources[0].profile = np.ones(ny) * grid.sources[0].amplitude
    grid[0, det_x, ny // 2] = fd.LineDetector(name="det")

    lam_bragg = 2.0 * (n1 * d1 + n2 * d2)
    meta = {
        "n_layers": n_layers, "lam0_design_m": lam0_design_m,
        "lam_run_m": lam_src, "d1_m": d1, "d2_m": d2, "n1": n1, "n2": n2,
        "dx": dx, "n1_cells": n1_cells, "n2_cells": n2_cells,
        "total_cells": total, "src_x": src_x, "det_x": det_x,
        "lam_bragg_analytic_m": lam_bragg,
        "dt_s": dt, "period_steps": period_steps,
    }
    return grid, meta


def cw_amplitude(grid, total_periods: float = 25.0, measure_periods: float = 8.0) -> float:
    """跑 total_periods 个周期，返回稳态段源频处 FFT 幅值（抗驻波纹波）。"""
    det = [d for d in grid.detectors if d.name == "det"][0]
    period_steps = grid.sources[0].period  # 整数步数
    n_total = int(total_periods * period_steps)
    n_meas = int(measure_periods * period_steps)
    grid.run(n_total, progress_bar=False)
    arr = np.array(det.E, dtype=float)  # (T, 1, 3)
    e = arr[-n_meas:, 0, 2]             # Ez 分量（源驱动分量下标 2）
    # FFT 在源频（f = 1/(period_steps·dt)）处取幅值
    dt = grid.time_step
    Y = np.fft.rfft(e - e.mean())
    f = np.fft.rfftfreq(e.size, dt)
    f_src = 1.0 / (period_steps * dt)
    k = int(np.argmin(np.abs(f - f_src)))
    return float(2.0 * np.abs(Y[k]) / e.size)   # 单边幅值


def transmission_spectrum(
    lam_list_m: np.ndarray,
    n_layers: int = 8,
    lam0_design_m: float = 1000e-9,
    n1: float = 1.5,
    n2: float = 2.5,
    dx: float = 5e-9,
    total_periods: float = 25.0,
    verbose: bool = False,
) -> dict:
    """对每个波长跑 Bragg 与真空参考，得 T(λ) = (A_bragg/A_ref)²。"""
    from ..analysis.spectrum import band_edges_from_transmission, rel_err

    lam_list_m = np.atleast_1d(np.asarray(lam_list_m, dtype=float))
    T = np.zeros(lam_list_m.size)
    amp_b = np.zeros(lam_list_m.size)
    amp_r = np.zeros(lam_list_m.size)

    for k, lam in enumerate(lam_list_m):
        gb, _ = build_bragg_grid(n_layers=n_layers, lam0_design_m=lam0_design_m,
                                 n1=n1, n2=n2, dx=dx, lam_run_m=float(lam))
        ab = cw_amplitude(gb, total_periods=total_periods)
        gr, _ = build_bragg_grid(n_layers=0, lam0_design_m=lam0_design_m,
                                 n1=n1, n2=n2, dx=dx, lam_run_m=float(lam))
        ar = cw_amplitude(gr, total_periods=total_periods)
        amp_b[k], amp_r[k] = ab, ar
        T[k] = (ab / ar) ** 2 if ar > 0 else float("nan")
        if verbose:
            print(f"    λ={lam*1e9:7.1f} nm  A_bragg={ab:.4e}  A_ref={ar:.4e}  T={T[k]:.4f}")

    lam_bragg = lam0_design_m  # 四分之一波长设计下解析带隙中心 = lam0_design
    # 带隙：T<0.5 的连续区间（按频率找边缘再换回波长）
    c0 = 299_792_458.0
    f = c0 / lam_list_m
    f_lo, f_hi = band_edges_from_transmission(f, T, threshold=0.5)
    # f 数组是降序（λ 升序），band_edges 返回的 (f_lo,f_hi) 可能次序颠倒 → 用 min/max 归正
    lam_lo = c0 / max(f_lo, f_hi) if f_lo == f_lo else float("nan")
    lam_hi = c0 / min(f_lo, f_hi) if f_lo == f_lo else float("nan")
    lam_center = float(np.sqrt(lam_lo * lam_hi)) if lam_lo == lam_lo else float("nan")
    if verbose:
        print(f"    analytic bandgap center {lam_bragg*1e9:.1f} nm; "
              f"sim {lam_lo*1e9:.1f}–{lam_hi*1e9:.1f} nm center {lam_center*1e9:.1f} nm")

    return {
        "lam_m": lam_list_m, "T": T, "amp_bragg": amp_b, "amp_ref": amp_r,
        "lam_bragg_analytic_m": lam_bragg,
        "lam_bandgap_lo_m": lam_lo, "lam_bandgap_hi_m": lam_hi,
        "lam_bandgap_center_m": lam_center,
        "bandgap_center_err_pct": rel_err(lam_center, lam_bragg),
        "T_min_in_gap": float(T[(lam_list_m > lam_lo) & (lam_list_m < lam_hi)].min())
            if lam_lo == lam_lo else float("nan"),
    }
