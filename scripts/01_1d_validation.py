"""01 一维 FDTD 验证：传播速度、PEC 反射、Fresnel 介质界面。

测量方法（v1.0 方法学）：
- 速度：双探针包络峰到达时刻（离散化误差 ~0.2%）
- 反射系数：入射/反射两时窗分别 FFT，在谱峰频率处取比值（复数 r）
"""
from __future__ import annotations

import numpy as np

from common import save_fig, save_json
from src.analysis.spectrum import fresnel_r, rel_err, windowed_fft_ratio
from src.fdtd_core.yee1d import C0, FDTD1D, gaussian_derivative

NX = 600
DX = 1e-3
COURANT = 0.95
SRC = 50
PROBE = 200
WALL = 300
T0_STEPS = 30
W_STEPS = 8


def gdev(n: int, dt: float) -> float:
    t = n * dt
    t0 = T0_STEPS * dt
    w = W_STEPS * dt
    return gaussian_derivative(t, t0, w)


def run_case(eps_r=None, pec_from: int | None = None):
    sim = FDTD1D(nx=NX, dx=DX, courant=COURANT, eps_r=eps_r)
    pec = None
    if pec_from is not None:
        pec = np.zeros(NX, dtype=bool)
        pec[pec_from:] = True
    rec = np.empty(900)
    for n in range(900):
        sim.step_h()
        sim.step_e(source=gdev(n, sim.dt), source_index=SRC, pec_mask=pec)
        rec[n] = sim.Ez[PROBE]
    return sim.dt, rec


def main() -> None:
    import matplotlib.pyplot as plt

    dt = COURANT * DX / C0
    out: dict = {"config": {"nx": NX, "dx_m": DX, "courant": COURANT,
                            "src": SRC, "probe": PROBE, "wall": WALL,
                            "t0_steps": T0_STEPS, "w_steps": W_STEPS}}

    # ---------- 1. 传播速度 ----------
    sim = FDTD1D(nx=NX, dx=DX, courant=COURANT)
    probeA, probeB = 150, 350
    recA = np.empty(1200)
    recB = np.empty(1200)
    for n in range(1200):
        sim.step_h()
        sim.step_e(source=gdev(n, sim.dt), source_index=SRC)
        recA[n] = sim.Ez[probeA]
        recB[n] = sim.Ez[probeB]
    iA = int(np.argmax(np.abs(recA)))
    iB = int(np.argmax(np.abs(recB)))
    v_sim = (probeB - probeA) * DX / ((iB - iA) * dt)
    out["propagation_speed"] = {
        "v_sim_m_per_s": v_sim, "c0_m_per_s": C0,
        "err_pct": rel_err(v_sim, C0),
        "peak_steps": [iA, iB],
    }
    print(f"[01] speed v={v_sim:.6e}  c={C0:.6e}  err={out['propagation_speed']['err_pct']:.4f}%")

    # ---------- 2. PEC 反射 ----------
    dt, rec = run_case(pec_from=WALL)
    i_inc = int(np.argmax(np.abs(rec[100:280]))) + 100
    i_ref = int(np.argmax(np.abs(rec[320:480]))) + 320
    wres = windowed_fft_ratio(np.arange(rec.size) * dt, rec, rec, i_inc, i_ref, half_width=60)
    r_sim = wres["r_complex"]
    out["pec_reflection"] = {
        "r_real": r_sim.real, "r_imag": r_sim.imag,
        "r_abs": wres["r_abs"], "r_phase_deg": wres["r_phase_deg"],
        "r_theory": -1.0,
        "err_pct_vs_minus1": abs(abs(r_sim) - 1.0) * 100,
        "phase_err_deg_vs_180": abs(abs(wres["r_phase_deg"]) - 180.0),
        "peak_method_r": float(rec[i_ref] / rec[i_inc]),
        "peak_steps": [i_inc, i_ref], "f_eval_Hz": wres["f_eval"],
    }
    print(f"[01] PEC r={r_sim:.5f}  |r|={wres['r_abs']:.5f} (expect 1)  phase={wres['r_phase_deg']:.2f}° (expect 180)")

    # ---------- 3. Fresnel 介质界面 ----------
    fres = {}
    for n2 in (1.5, 2.0, 3.0):
        eps = FDTD1D.layered_eps_r(NX, DX, [(WALL * DX, NX * DX, n2**2)])
        _, rec3 = run_case(eps_r=eps)
        i_inc = int(np.argmax(np.abs(rec3[100:280]))) + 100
        i_ref = int(np.argmax(np.abs(rec3[320:480]))) + 320
        w3 = windowed_fft_ratio(np.arange(rec3.size) * dt, rec3, rec3, i_inc, i_ref, half_width=60)
        r_th = fresnel_r(1.0, n2)
        r_s = w3["r_complex"]
        fres[f"n2={n2}"] = {
            "n2": n2, "r_sim_real": r_s.real, "r_sim_abs": w3["r_abs"],
            "r_theory": r_th, "err_pct": rel_err(r_s.real, r_th),
        }
        print(f"[01] Fresnel n2={n2}: r_sim={r_s.real:+.5f}  r_th={r_th:+.5f}  err={fres[f'n2={n2}']['err_pct']:.3f}%")
    out["fresnel"] = fres

    # ---------- 图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    ax = axes[0, 0]
    ax.plot(recA, lw=0.9, label=f'probe x={probeA}')
    ax.plot(recB, lw=0.9, label=f'probe x={probeB}')
    ax.set_xlabel('step'); ax.set_ylabel('Ez'); ax.set_title('1D pulse propagation')
    ax.legend()

    ax = axes[0, 1]
    ax.plot(rec, lw=0.9)
    ax.axvspan(100, 280, color='C0', alpha=0.15, label='inc window')
    ax.axvspan(320, 480, color='C3', alpha=0.15, label='ref window')
    ax.set_xlabel('step'); ax.set_ylabel('Ez'); ax.set_title('PEC reflection @ probe')
    ax.legend()

    ax = axes[1, 0]
    ns = sorted(fres.keys(), key=lambda k: fres[k]['n2'])
    xs = [fres[k]['n2'] for k in ns]
    ax.plot(xs, [fres[k]['r_theory'] for k in ns], 'o-', label='Fresnel theory')
    ax.plot(xs, [fres[k]['r_sim_real'] for k in ns], 's--', label='FDTD (FFT window)')
    ax.set_xlabel('n2'); ax.set_ylabel('r'); ax.set_title('Dielectric interface reflection')
    ax.legend()

    ax = axes[1, 1]
    ax.axis('off')
    ax.text(0.02, 0.95, out['propagation_speed'].__repr__().replace(', ', ',\n')[:300], va='top', fontsize=7)
    ax.text(0.02, 0.35, f"PEC |r|={out['pec_reflection']['r_abs']:.5f}  phase={out['pec_reflection']['r_phase_deg']:.2f}°",
            va='top', fontsize=8)
    save_fig("01_1d_validation.png")

    save_json("01_1d_validation.json", out)
    print("[01] done -> 01_1d_validation.json")


if __name__ == "__main__":
    main()
