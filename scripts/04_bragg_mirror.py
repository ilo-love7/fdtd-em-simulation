"""04 Bragg 反射镜透射谱（flaport/fdtd 0.3.5 库仿真）。

一维四分之一波长堆栈（n1=1.5 / n2=2.5，8 层对，设计波长 1000 nm），
逐波长 CW 稳态激励求 T(λ) = (A_bragg/A_ref)²，验证光子带隙位置
（解析中心 λ0 = 2(n1 d1 + n2 d2) = 1000 nm）。

库使用注意（详见 src/fdtd_lib/bragg.py 文档）：
- period 传**整数步数**（float 被当秒）；
- 单点 LineSource profile 除零 NaN → 线源 profile 手工置均匀；
- Object「对称化」重置 z=-1 行逆介电常数 → 注册后整体重建（否则波绕过堆栈 T≡1）；
- PML 铺传播方向两端，Object 只铺非 PML 区。
"""
from __future__ import annotations

import numpy as np

from common import save_fig, save_json
from src.analysis.spectrum import bragg_wavelength, quarter_wave_thickness
from src.fdtd_lib.bragg import transmission_spectrum

LAM0_M = 1000e-9
N1, N2 = 1.5, 2.5
N_LAYERS = 8
DX = 5e-9
LAM_NM = np.arange(700.0, 1401.0, 35.0)  # 21 波长点，覆盖通带+带隙+通带


def main() -> None:
    import matplotlib.pyplot as plt

    lam_m = LAM_NM * 1e-9
    res = transmission_spectrum(
        lam_m, n_layers=N_LAYERS, lam0_design_m=LAM0_M,
        n1=N1, n2=N2, dx=DX, total_periods=25.0, verbose=True,
    )

    d1 = quarter_wave_thickness(N1, LAM0_M)
    d2 = quarter_wave_thickness(N2, LAM0_M)
    lam_an = bragg_wavelength(N1, d1, N2, d2)
    T = np.asarray(res["T"], dtype=float)
    lam_lo = res["lam_bandgap_lo_m"] * 1e9
    lam_hi = res["lam_bandgap_hi_m"] * 1e9
    lam_c = res["lam_bandgap_center_m"] * 1e9

    # ---------- 图：T(λ) + 带隙标注 ----------
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(LAM_NM, T, "o-", color="#1f77b4", ms=4, lw=1.4, label="FDTD T(λ)")
    if lam_lo == lam_lo:
        ax.axvspan(lam_lo, lam_hi, color="#d62728", alpha=0.12,
                   label=f"bandgap {lam_lo:.0f}–{lam_hi:.0f} nm")
        ax.axvline(lam_c, color="#d62728", ls="--", lw=1.2,
                   label=f"gap center {lam_c:.1f} nm")
    ax.axvline(lam_an * 1e9, color="k", ls=":", lw=1.0,
               label=f"analytic λ0 = {lam_an*1e9:.0f} nm")
    ax.set_xlabel("wavelength λ (nm)")
    ax.set_ylabel("transmission T")
    ax.set_ylim(-0.02, 1.08)
    ax.set_title(f"1D Bragg mirror transmission ({N_LAYERS} layer pairs, "
                 f"n={N1}/{N2}, design λ0={LAM0_M*1e9:.0f} nm)")
    ax.legend(fontsize=8, loc="upper right")
    save_fig("04_bragg_mirror.png")

    out = {
        "config": {
            "n_layers": N_LAYERS, "lam0_design_nm": LAM0_M * 1e9,
            "n1": N1, "n2": N2, "dx_m": DX, "total_periods": 25.0,
            "n_wavelengths": int(LAM_NM.size),
            "solver": "flaport-fdtd 0.3.5 (CW steady-state amplitude ratio)",
        },
        "d1_nm": d1 * 1e9, "d2_nm": d2 * 1e9,
        "lam_analytic_nm": lam_an * 1e9,
        "bandgap_lo_nm": lam_lo, "bandgap_hi_nm": lam_hi,
        "bandgap_center_nm": lam_c,
        "bandgap_center_err_pct": res["bandgap_center_err_pct"],
        "T_min_in_gap": res["T_min_in_gap"],
        "lam_nm": LAM_NM,
        "T": T,
        "amp_bragg": np.asarray(res["amp_bragg"], dtype=float),
        "amp_ref": np.asarray(res["amp_ref"], dtype=float),
    }
    save_json("04_bragg_mirror.json", out)
    print(f"\nbandgap {lam_lo:.1f}–{lam_hi:.1f} nm  center {lam_c:.1f} nm  "
          f"err {res['bandgap_center_err_pct']:.3f}%  T_min {res['T_min_in_gap']:.4f}")
    print("saved: results/04_bragg_mirror.json, figures/04_bragg_mirror.png")


if __name__ == "__main__":
    main()
