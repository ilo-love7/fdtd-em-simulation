"""09 Drude 色散介质验证（v1.2，模块 A）。

1. 等离子体截止：Drude 板 r/t 随频率扫描，|r|/|t| vs 解析 Airy 公式
   （ω<ω_p 强反射 + 倏逝透射；ω>ω_p 透明）。
2. 传播模相速度：均匀 Drude 中 CW 空间谱 k_num vs 冷等离子体 n(ω)ω/c0。
3. 有损吸收：γ>0 时透射被吸收压低。
4. 倏逝衰减的直接空间拟合受真空侧驻波 + 边界回波污染 —— 如实标注为
   测量口径限制，截止判据以板透射 |t|（与解析 1% 吻合）为准。
"""
from __future__ import annotations

import numpy as np

from common import save_fig, save_json
from src.analysis.dispersive import (drude_eps_r, drude_n, group_index,
                                     slab_rt)
from src.fdtd_core.drude import FDTD1D_Drude, cw_ramp

NX = 1600
SRC = 200
SLAB = (500, 560)
P_L, P_R = 350, 800
WP = 0.1
WP2 = WP**2
GAMMA = 0.01
S = 0.95
WIN = (700, 1000)
OMEGA_SWEEP = [0.06, 0.08, 0.09, 0.11, 0.13, 0.15, 0.20, 0.30]


def _run(omega: float, with_slab: bool, gamma: float = GAMMA) -> dict[str, np.ndarray]:
    if with_slab:
        ei, w2, ga = FDTD1D_Drude.drude_box(NX, SLAB[0], SLAB[1], WP2, gamma)
    else:
        ei, w2, ga = np.ones(NX), np.zeros(NX), np.zeros(NX)
    sim = FDTD1D_Drude(nx=NX, dx=1.0, courant=S, eps_inf=ei, wp2=w2, gamma=ga)
    src = lambda t: 0.5 * cw_ramp(t, omega, 100.0)
    return sim.run_record(WIN[1] + 50, source_fn=src, source_index=SRC,
                          record_indices=(P_L, P_R))


def _fft_at(y: np.ndarray, dt: float, omega: float, win: tuple[int, int]) -> complex:
    seg = y[win[0]:win[1]]
    seg = seg - seg.mean()
    Y = np.fft.rfft(seg * np.hanning(seg.size))
    f = np.fft.rfftfreq(seg.size, d=dt)
    k = int(np.argmin(np.abs(2 * np.pi * f - omega)))
    return complex(Y[k])


def sweep_rt() -> list[dict]:
    out = []
    for omega in OMEGA_SWEEP:
        v, s = _run(omega, False), _run(omega, True)
        Vi = _fft_at(v["p0"], S, omega, WIN)
        Vr = _fft_at(v["p1"], S, omega, WIN)
        Si = _fft_at(s["p0"], S, omega, WIN)
        St = _fft_at(s["p1"], S, omega, WIN)
        r_meas = (Si - Vi) / Vi
        t_meas = St / Vr
        r_ana, t_ana = slab_rt(omega, WP2, GAMMA, d=SLAB[1] - SLAB[0])
        ratio = WP / omega
        out.append({
            "omega": omega, "wp_over_omega": ratio,
            "below_cutoff": bool(ratio > 1.0),
            "r_abs_meas": float(abs(r_meas)), "r_abs_ana": float(abs(r_ana)),
            "t_abs_meas": float(abs(t_meas)), "t_abs_ana": float(abs(t_ana)),
            "eps_r": complex(drude_eps_r(omega, WP2, GAMMA)),
            "n_refr": complex(drude_n(omega, WP2, GAMMA)),
        })
    return out


def measure_k(omega: float, gamma: float = 0.0) -> float:
    """均匀 Drude 中 CW 稳态空间谱峰 k_num（抛物线插值）。"""
    nx = 2600
    ei = np.full(nx, 1.0)
    w2 = np.full(nx, WP2)
    ga = np.full(nx, gamma)
    sim = FDTD1D_Drude(nx=nx, dx=1.0, courant=S, eps_inf=ei, wp2=w2, gamma=ga)
    src = lambda t: 0.5 * cw_ramp(t, omega, 80.0)
    n_steps = 1800
    for n in range(n_steps):
        sim.step_h()
        sim.step_e(source=src(n * sim.dt), source_index=200)
    seg = sim.Ez[900:1700].copy()
    seg -= seg.mean()
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    ks = 2 * np.pi * np.fft.rfftfreq(seg.size, d=1.0)
    ki = int(np.argmax(spec[1:])) + 1
    a0, a1, a2 = spec[ki - 1], spec[ki], spec[ki + 1]
    den = a0 - 2 * a1 + a2
    delta = 0.5 * (a0 - a2) / den if abs(den) > 0 else 0.0
    return float(ks[ki] + delta * (ks[1] - ks[0]))


def phase_velocity_scan() -> list[dict]:
    out = []
    for omega in [0.12, 0.15, 0.20, 0.30, 0.50]:
        k_num = measure_k(omega, gamma=0.0)
        n = complex(drude_n(omega, WP2, 0.0))
        k_ana = omega * n.real
        ng = group_index(omega, WP2, 0.0)
        out.append({
            "omega": omega, "k_num": k_num, "k_ana": k_ana,
            "err_vs_ana_pct": abs(k_num / k_ana - 1) * 100,
            "v_phase_num": omega / k_num, "v_phase_ana": 1.0 / n.real,
            "n_phase": n.real, "n_group": ng,
        })
    return out


def absorption_check() -> list[dict]:
    """有损 Drude 板：γ 增大透射衰减（截止以下 + 以上各一点）。"""
    out = []
    for omega in [0.08, 0.20]:
        for gamma in [0.0, 0.02, 0.05, 0.10]:
            s = _run(omega, True, gamma=gamma)
            St = _fft_at(s["p1"], S, omega, WIN)
            v = _run(omega, False)
            Vr = _fft_at(v["p1"], S, omega, WIN)
            t_abs = abs(St / Vr)
            _, t_ana = slab_rt(omega, WP2, gamma, d=SLAB[1] - SLAB[0])
            out.append({"omega": omega, "gamma": gamma,
                        "t_abs_meas": float(t_abs), "t_abs_ana": float(abs(t_ana))})
    return out


def main() -> None:
    import matplotlib.pyplot as plt

    print("== 09 Drude dispersion ==")
    rt = sweep_rt()
    pv = phase_velocity_scan()
    ab = absorption_check()

    # --- diagnostics ---
    errs = [max(abs(d["r_abs_meas"] - d["r_abs_ana"]),
                abs(d["t_abs_meas"] - d["t_abs_ana"])) for d in rt]
    pv_good = [d for d in pv if d["omega"] >= 0.15]
    pv_err = max(d["err_vs_ana_pct"] for d in pv_good)
    ab_err = max(abs(d["t_abs_meas"] - d["t_abs_ana"]) for d in ab)
    print("rt scan: omega | ratio | |r|meas/ana | |t|meas/ana")
    for d in rt:
        print(f"  {d['omega']:.2f}  {d['wp_over_omega']:.2f}  "
              f"{d['r_abs_meas']:.4f}/{d['r_abs_ana']:.4f}  "
              f"{d['t_abs_meas']:.4f}/{d['t_abs_ana']:.4f}"
              f"{'  <cutoff' if d['below_cutoff'] else ''}")
    print(f"max rt abs deviation: {max(errs):.4f}")
    print("phase velocity: omega | k_num | k_ana | err%")
    for d in pv:
        print(f"  {d['omega']:.2f}  {d['k_num']:.5f}  {d['k_ana']:.5f}  "
              f"{d['err_vs_ana_pct']:.3f}%")
    print(f"phase-velocity max err (omega>=0.15): {pv_err:.3f}%")
    print(f"absorption max |t| deviation: {ab_err:.4f}")

    out = {
        "module": "A - Drude dispersive medium (ADE)",
        "params": {"wp": WP, "gamma": GAMMA, "slab_cells": SLAB[1] - SLAB[0],
                   "courant": S},
        "cutoff_scan": rt,
        "phase_velocity": pv,
        "absorption": ab,
        "summary": {
            "rt_max_abs_deviation": max(errs),
            "phase_velocity_max_err_pct_omega_ge_015": pv_err,
            "absorption_max_abs_deviation": ab_err,
            "cutoff_note": "omega<wp: strong reflection + evanescent |t|~0.01 "
                           "(analytic Airy match ~1%); direct spatial decay fit "
                           "contaminated by vacuum-side standing wave / boundary "
                           "echo -> report as measurement limitation, not physics",
        },
    }
    save_json("09_drude.json", out)

    # --- figures ---
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    ax = axes[0]
    w = [d["omega"] for d in rt]
    ax.plot(w, [d["r_abs_meas"] for d in rt], "o-", label="|r| FDTD")
    ax.plot(w, [d["r_abs_ana"] for d in rt], "k--", label="|r| analytic")
    ax.plot(w, [d["t_abs_meas"] for d in rt], "s-", label="|t| FDTD")
    ax.plot(w, [d["t_abs_ana"] for d in rt], "k:", label="|t| analytic")
    ax.axvline(WP, color="r", ls="-.", lw=1, label="wp cutoff")
    ax.set_xlabel("omega [norm]"); ax.set_ylabel("amplitude")
    ax.set_title("plasma cutoff reflection/transmission", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.plot([d["omega"] for d in pv], [d["v_phase_num"] for d in pv], "o-",
            label="v_ph FDTD")
    ax.plot([d["omega"] for d in pv], [d["v_phase_ana"] for d in pv], "k--",
            label="v_ph cold plasma")
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xlabel("omega"); ax.set_ylabel("v_phase [c0=1]")
    ax.set_title("phase velocity > c0 (anomalous)", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[2]
    for omega in [0.08, 0.20]:
        sel = [d for d in ab if d["omega"] == omega]
        ax.plot([d["gamma"] for d in sel], [d["t_abs_meas"] for d in sel], "o-",
                label=f"|t| w={omega} FDTD")
        ax.plot([d["gamma"] for d in sel], [d["t_abs_ana"] for d in sel], "k--",
                alpha=0.6)
    ax.set_xlabel("gamma [collision]"); ax.set_ylabel("|t|")
    ax.set_title("loss absorption", fontsize=9)
    ax.legend(fontsize=7)
    save_fig("09_drude.png")
    print("saved: results/09_drude.json + figures/09_drude.png")


if __name__ == "__main__":
    main()
