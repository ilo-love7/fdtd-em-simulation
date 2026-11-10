"""07 二维光子晶体带隙（正方晶格介质柱，TMz）。

晶格常数 a、半径 r=0.2a、柱介电常数 εr=11.7（硅）背景空气。
CW 行源照射 N 行晶格薄板，逐波长稳态透射谱 T(λ)=(A_slab/A_ref)²。

定量口径（离线，不引文献数）：
1. **带隙自检**：晶格常数 a→2a 后 T(λ/a) 应重合（光子晶体无本征尺度定理）
2. **周期性归因**：同体积分数均匀介质膜（无周期）无禁带 → 带隙源于周期性
3. 报告 T_min、带隙边缘（T<0.5 连续区）与归一化频率 a/λ

参考场 A_ref(λ) 按 (λ, 网格) 缓存复用（晶格/均匀对照共用），逐段存盘防丢。
"""
from __future__ import annotations

import numpy as np

from common import RESULTS, save_fig, save_json
from src.fdtd_core.yee2d import C0, FDTD2D

# 几何（以晶格常数 a 为单位）
DX_FRAC = 1 / 20          # dx = a/20
N_ROWS = 8                # 晶格行数
NX_A, NY_A = 16, 10       # 域尺寸 [a]
R_FRAC = 0.2              # 柱半径 r/a
EPS_ROD = 11.7
PML = 10
LAM_OVER_A = np.linspace(1.6, 6.0, 15)

_REF_CACHE: dict = {}


def build_eps(a: float, nx: int, ny: int, dx: float, uniform: bool = False) -> np.ndarray:
    """介电常数图：正方晶格介质柱（x 为传播向），或同体积分数均匀膜。"""
    eps = np.ones((nx, ny))
    r = R_FRAC * a
    x0 = int(round(4 * a / dx))
    if uniform:
        x1 = x0 + int(round(N_ROWS * a / dx))
        area_frac = np.pi * r**2 / a**2
        eps[x0:x1, :] = 1.0 + (EPS_ROD - 1.0) * area_frac
        return eps
    xs = np.arange(nx) * dx
    ys = np.arange(ny) * dx
    for col in range(N_ROWS):
        cx = 4 * a + col * a
        for row in range(int(np.ceil(ny * dx / a)) + 1):
            cy = row * a
            mask = (xs[:, None] - cx) ** 2 + (ys[None, :] - cy) ** 2 <= r**2
            eps[mask] = EPS_ROD
    return eps


def cw_amplitude_2d(eps: np.ndarray, lam: float, a: float, nx: int, ny: int,
                    dx: float) -> float:
    """CW 稳态透射幅：探测行 |Ez| 稳态峰峰值/2。"""
    courant = 0.65
    sim = FDTD2D(nx=nx, ny=ny, dx=dx, courant=courant,
                 eps_r=eps, pml_cells_x=PML, pml_cells_y=PML)
    period_steps = int(round(lam / (C0 * sim.dt)))
    n_warm = int(20 * period_steps)
    n_meas = int(3 * period_steps)
    src_x = int(round(2.0 * a / dx))
    det_x = min(int(round((4.0 + N_ROWS + 1.5) * a / dx)), nx - PML - 2)
    rec = []
    for n in range(n_warm + n_meas):
        sim.step_h()
        s = float(np.sin(2 * np.pi * n / period_steps))
        sim.step_e_line(source=s, line_source=src_x)
        if n >= n_warm:
            rec.append(float(sim.Ez[det_x, ny // 2]))
    arr = np.asarray(rec)
    return float((arr.max() - arr.min()) / 2.0)


def ref_amplitude(lam: float, a: float, nx: int, ny: int, dx: float) -> float:
    key = (round(lam, 12), nx, ny, round(dx, 15))
    if key not in _REF_CACHE:
        _REF_CACHE[key] = cw_amplitude_2d(np.ones((nx, ny)), lam, a, nx, ny, dx)
    return _REF_CACHE[key]


def transmission(a: float, lam_over_a: np.ndarray, uniform: bool = False) -> np.ndarray:
    dx = DX_FRAC * a
    nx = int(round(NX_A * a / dx))
    ny = int(round(NY_A * a / dx))
    T = []
    for la in lam_over_a:
        lam = la * a
        eps = build_eps(a, nx, ny, dx, uniform=uniform)
        amp = cw_amplitude_2d(eps, lam, a, nx, ny, dx)
        ref = ref_amplitude(lam, a, nx, ny, dx)
        T.append((amp / ref) ** 2 if ref > 0 else float("nan"))
    return np.asarray(T)


def band_edges(lam_over_a: np.ndarray, T: np.ndarray, thresh: float = 0.5):
    gap = T < thresh
    if not gap.any():
        return None
    idx = np.where(gap)[0]
    runs, start = [], idx[0]
    for i in range(1, len(idx)):
        if idx[i] != idx[i - 1] + 1:
            runs.append((start, idx[i - 1])); start = idx[i]
    runs.append((start, idx[-1]))
    s, e = max(runs, key=lambda r: r[1] - r[0])
    lam_lo, lam_hi = float(lam_over_a[s]), float(lam_over_a[e])
    return {"lam_lo_over_a": min(lam_lo, lam_hi), "lam_hi_over_a": max(lam_lo, lam_hi),
            "center_over_a": float(np.sqrt(lam_lo * lam_hi)),
            "T_min": float(T.min()), "width_over_a": abs(lam_hi - lam_lo)}


def main() -> None:
    import matplotlib.pyplot as plt

    print("== 07 photonic crystal (square lattice, TMz) ==")
    out: dict = {"config": {"r_over_a": R_FRAC, "eps_rod": EPS_ROD,
                            "n_rows": N_ROWS, "dx_over_a": DX_FRAC,
                            "lam_over_a_range": [float(LAM_OVER_A[0]), float(LAM_OVER_A[-1])],
                            "n_lambda": int(LAM_OVER_A.size)}}

    # 基准晶格 a = 1
    T1 = transmission(1.0, LAM_OVER_A)
    g1 = band_edges(LAM_OVER_A, T1)
    out["a1"] = {"T": T1.tolist(), "gap": g1}
    print(f"a=1 : gap {g1['lam_lo_over_a']:.2f}–{g1['lam_hi_over_a']:.2f} a, "
          f"center {g1['center_over_a']:.3f} a, T_min {g1['T_min']:.4f}" if g1 else "no gap")
    save_json("07_photonic_crystal.json", out)   # 增量存盘

    # 缩放自检：a=2 → T(λ/a) 应与 a=1 重合
    T2 = transmission(2.0, LAM_OVER_A)
    g2 = band_edges(LAM_OVER_A, T2)
    out["a2"] = {"T": T2.tolist(), "gap": g2}
    if g1 and g2:
        ratio = g2["center_over_a"] / g1["center_over_a"]
        out["scaling_center_ratio"] = float(ratio)
        print(f"a=2 : gap center {g2['center_over_a']:.3f} a  (λ/a ratio = {ratio:.4f}, want 1)")
    save_json("07_photonic_crystal.json", out)

    # 均匀膜对照（同体积分数，无周期）
    Tu = transmission(1.0, LAM_OVER_A, uniform=True)
    out["uniform_film"] = {"T": Tu.tolist(), "T_min": float(Tu.min())}
    print(f"uniform film T_min = {Tu.min():.4f} (no stopband expected)")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(LAM_OVER_A, T1, "o-", label="crystal a", color="#1f77b4")
    ax.plot(LAM_OVER_A, T2, "s--", label="crystal 2a (λ/a)", color="#ff7f0e", alpha=0.8)
    ax.plot(LAM_OVER_A, Tu, "^:", label="uniform film", color="#2ca02c", alpha=0.8)
    if g1:
        ax.axvspan(g1["lam_lo_over_a"], g1["lam_hi_over_a"], color="gray", alpha=0.2,
                   label=f"gap ({g1['center_over_a']:.2f}a)")
    ax.set_xlabel(r"$\lambda / a$"); ax.set_ylabel("transmission")
    ax.set_title("Photonic crystal TMz transmission", fontsize=9)
    ax.legend(fontsize=8)
    save_fig("07_photonic_crystal.png")
    save_json("07_photonic_crystal.json", out)
    print("saved: results/07_photonic_crystal.json + figures/07_photonic_crystal.png")


if __name__ == "__main__":
    main()
