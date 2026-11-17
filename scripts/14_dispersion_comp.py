"""14 各向同性 9 点 Laplacian 色散补偿（v1.3，模块 B）。

1. 解析：5 点 vs 9 点 Laplacian 的 v_ph 各向异性随 Ω、S 的变化；
   9 点把 O(K⁴) 各向异性主项精确消去，余项 O(K⁶)（压缩 100–700×）。
2. FDTD 验证：全场相位列源在若干 ky（轴到 ~37°）驱动 CW，从复相量拟合
   实测 kx，与两格式各自的离散色散关系对照；给出 v_ph(θ) 曲线，显示
   5 点随角度上扬、9 点近似水平。
3. 稳定上限：5 点 S≤1/√2、9 点 S≤√3/2（9 点允许更大 Courant）。
"""
from __future__ import annotations

import numpy as np

from common import save_fig, save_json
from src.analysis.isotropic_dispersion import (anisotropy_wave,
                                               kx_from_ky,
                                               phase_velocity_wave)
from src.fdtd_core.isotropic import Wave2D, laplacian_symbol

C0 = 299_792_458.0


def analytic_scan() -> list[dict]:
    out = []
    dx = 1e-3
    for lodx in [8, 12, 16, 20, 32]:
        omega = 2 * np.pi * C0 / (lodx * dx)
        for S in [0.4, 0.6, 0.7]:
            a5 = anisotropy_wave(omega, dx, S, False)
            a9 = anisotropy_wave(omega, dx, S, True)
            if not np.isfinite(a5["vp_anisotropy"]) or not np.isfinite(a9["vp_anisotropy"]):
                continue
            ratio = (abs(a9["vp_anisotropy"]) / max(abs(a5["vp_anisotropy"]), 1e-30)
                     if abs(a5["vp_anisotropy"]) > 1e-12 else float("nan"))
            out.append({
                "lambda_over_dx": lodx, "Omega": omega * dx / C0, "courant": S,
                "vp_aniso_5pt": a5["vp_anisotropy"] / C0,
                "vp_aniso_9pt": a9["vp_anisotropy"] / C0,
                "compression": 1.0 / ratio if np.isfinite(ratio) and ratio > 0 else float("inf"),
            })
    return out


def fdtd_oblique_scan(lodx=20, S=0.6, N=300) -> list[dict]:
    """相位列源 CW 驱动，实测 (kx,ky) 与 v_ph，对照两格式离散色散。"""
    dx = 1.0
    k0 = 2 * np.pi / lodx
    omega = k0 * C0 / dx
    dt = S * dx / C0
    rhs = (np.sin(omega * dt / 2.0) / S) ** 2
    out = []
    for iso in [False, True]:
        for ky_imp in [0.0, 0.08, 0.14, 0.16]:
            sim = Wave2D(nx=N, ny=N, dx=dx, courant=S, isotropic=iso, omega=omega)
            src_i = 48
            y = np.arange(N) - N // 2
            period = 2 * np.pi / (omega * sim.dt)
            n_warm, n_acc = int(period * 18), int(period * 14)
            for n in range(n_warm + n_acc):
                src = np.sin(omega * n * sim.dt - ky_imp * y)
                sim.step(source=src, source_i=src_i, accumulate=(n >= n_warm))
            kx, ky = sim.phasor_wavenumbers((140, 310), (110, 230))
            kmag = float(np.hypot(kx, ky))
            theta = float(np.arctan2(ky, kx))
            # 色散关系残差：实测 (kx,ky) 是否满足该格式的离散符号（独立实现验证）
            sx2 = np.sin(kx * dx / 2.0) ** 2
            sy2 = np.sin(ky * dx / 2.0) ** 2
            lhs = sx2 + sy2 - (2.0 / 3.0) * sx2 * sy2 if iso else sx2 + sy2
            residual = abs(lhs - rhs) / rhs
            kxa = kx_from_ky(omega, dx, S, ky, iso)
            out.append({
                "scheme": "9pt" if iso else "5pt",
                "ky_imposed": ky_imp,
                "kx_meas": float(kx), "ky_meas": float(ky),
                "theta_deg": float(np.degrees(theta)),
                "k_mag_meas": kmag,
                "k_analytic": float(kxa),
                "kx_err_pct": abs(kx / kxa - 1) * 100 if np.isfinite(kxa) else float("nan"),
                "dispersion_residual_pct": float(residual * 100),
                "v_phase_over_c": omega / kmag / (C0 / dx),
            })
    return out


def main() -> None:
    import matplotlib.pyplot as plt

    print("== 14 dispersion compensation (9-point Laplacian) ==")
    an = analytic_scan()
    fd = fdtd_oblique_scan()

    print("analytic anisotropy (v_ph(axis)-v_ph(diag))/c | 5pt 9pt compression")
    for d in an:
        print(f"  lodx={d['lambda_over_dx']:2d} S={d['courant']:.2f}: "
              f"{d['vp_aniso_5pt']:+.3e} {d['vp_aniso_9pt']:+.3e} "
              f"{d['compression']:.1f}x")
    comp = [d["compression"] for d in an if np.isfinite(d["compression"])]
    print(f"compression range: {min(comp):.1f}x - {max(comp):.1f}x  (median {np.median(comp):.1f}x)")
    print("FDTD oblique: scheme ky_imp theta res% kx_err% vp/c")
    for d in fd:
        print(f"  {d['scheme']:>3s} {d['ky_imposed']:.2f} {d['theta_deg']:5.1f} "
              f"{d['dispersion_residual_pct']:5.2f} {d['kx_err_pct']:6.2f} "
              f"{d['v_phase_over_c']:.5f}")

    res_max = max(d["dispersion_residual_pct"] for d in fd)
    kx_err = max(d["kx_err_pct"] for d in fd if np.isfinite(d["kx_err_pct"]))
    fd5 = [d for d in fd if d["scheme"] == "5pt"]
    fd9 = [d for d in fd if d["scheme"] == "9pt"]
    spread5 = max(d["v_phase_over_c"] for d in fd5) - min(d["v_phase_over_c"] for d in fd5)
    spread9 = max(d["v_phase_over_c"] for d in fd9) - min(d["v_phase_over_c"] for d in fd9)

    out = {
        "module": "B - isotropic 9-point Laplacian dispersion compensation",
        "analytic_scan": an,
        "fdtd_oblique_scan": fd,
        "summary": {
            "compression_median": float(np.median(comp)),
            "compression_min": float(min(comp)),
            "compression_max": float(max(comp)),
            "fdtd_kx_max_err_pct": float(kx_err),
            "fdtd_dispersion_residual_max_pct": float(res_max),
            "v_phase_spread_5pt": float(spread5),
            "v_phase_spread_9pt": float(spread9),
            "courant_limit_5pt": 1 / np.sqrt(2), "courant_limit_9pt": np.sqrt(3) / 2,
            "mechanism": "9pt symbol -4[sx^2+sy^2-(2/3)sx^2sy^2] cancels the O(K^4) "
                         "theta-dependent term of 5pt exactly; residual O(K^6)",
            "note": "FDTD vp(angle) spread limited by phasor-fit noise (~0.05%c) "
                    "below the analytic 9pt anisotropy; analytic scan is the "
                    "quantitative result, FDTD confirms the dispersion realization",
        },
    }
    save_json("14_dispersion_comp.json", out)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    ax = axes[0]
    for lodx in [8, 12, 20]:
        sel = [d for d in an if d["lambda_over_dx"] == lodx]
        ax.plot([d["Omega"] for d in sel], [abs(d["vp_aniso_5pt"]) for d in sel],
                "o-", label=f"5pt lodx={lodx}")
        ax.plot([d["Omega"] for d in sel], [abs(d["vp_aniso_9pt"]) for d in sel],
                "s--", label=f"9pt lodx={lodx}")
    ax.set_yscale("log"); ax.set_xlabel("Omega = w dx/c"); ax.set_ylabel("|v_ph aniso|/c")
    ax.set_title("9pt kills O(K^4) anisotropy", fontsize=9); ax.legend(fontsize=6)
    ax = axes[1]
    for sch, mk in [("5pt", "o"), ("9pt", "s")]:
        sel = [d for d in fd if d["scheme"] == sch]
        ax.plot([d["theta_deg"] for d in sel], [d["v_phase_over_c"] for d in sel],
                mk + "-", label=sch)
    ax.set_xlabel("propagation angle [deg]"); ax.set_ylabel("v_phase/c")
    ax.set_title("FDTD v_phase vs angle", fontsize=9); ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot([d["kx_meas"] for d in fd], [d["k_analytic"] for d in fd], "o")
    lim = [min(d["kx_meas"] for d in fd) * 0.98, max(d["kx_meas"] for d in fd) * 1.02]
    ax.plot(lim, lim, "k--")
    ax.set_xlabel("k_x measured"); ax.set_ylabel("k_x analytic")
    ax.set_title(f"dispersion check (max {kx_err:.2f}%)", fontsize=9)
    save_fig("14_dispersion_comp.png")
    print("saved: results/14_dispersion_comp.json + figures/14_dispersion_comp.png")


if __name__ == "__main__":
    main()
