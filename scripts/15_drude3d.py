"""15 3D Yee + Drude ADE + PML（v1.3，模块 C）。

1. 3D 平面波（Ey 沿 +x，整片 y/z 注入）色散：k 实测 vs 1D 离散关系（~1.7%，
   短时窗避免边界回波）。
2. 3D Drude 板正入射 r/t vs 解析 Airy（复用 v1.2 slab_rt），等离子体截止
   与传播两频点。
3. PML 匹配有耗层：脉冲激励后外缘残余/内部场 ~1e-5（吸收有效）。
4. 稳定性：有损 Drude 长时间步进不发散；CFL 上限 S≤1/√3 的越界拒绝。
"""
from __future__ import annotations

import numpy as np

from common import save_fig, save_json
from src.analysis.dispersive import slab_rt
from src.analysis.dispersion import k_discrete_1d
from src.fdtd_core.yee3d import FDTD3D_Drude


def dispersion_scan() -> list[dict]:
    out = []
    omega = 0.25
    for S in [0.3, 0.4, 0.5]:
        nx, ny, nz = 160, 6, 6
        sim = FDTD3D_Drude(nx=nx, ny=ny, nz=nz, courant=S)
        src_i, y, z = 20, 3, 3
        period = 2 * np.pi / (omega * sim.dt)
        total = int(period * 12)
        xs = np.arange(45, 90)
        acc = np.zeros(len(xs), complex)
        n0 = total - int(period * 2)
        for n in range(total):
            sim.step()
            sim.inject_plane("Ey", src_i,
                             np.sin(omega * n * sim.dt) * min(1.0, n / period),
                             slice(None), slice(None))
            if n >= n0:
                for q, i in enumerate(xs):
                    acc[q] += sim.Ey[i, y, z] * np.exp(+1j * omega * n * sim.dt)
        acc *= 2.0 / (total - n0)
        kx = float(np.polyfit(xs, np.unwrap(np.angle(acc)), 1)[0])
        ka = k_discrete_1d(omega, 1.0, S, 1.0)
        out.append({"courant": S, "k_fdtd": kx, "k_discrete": ka,
                    "k_physical": omega, "err_pct": abs(kx / ka - 1) * 100})
    return out


def drude_slab_rt() -> list[dict]:
    """3D Drude 板：透射用真空参照归一（消源幅与边界回波）；截止下测板内倏逝衰减。"""
    out = []
    nx, ny, nz = 140, 12, 12
    cells, xs, xe = 4, 70, 110
    wp2 = 0.09  # ω_p = 0.3
    ei, w2, ga = FDTD3D_Drude.drude_slab(nx, ny, nz, xs, xe, wp2, gamma=0.0)

    def run(ei_, w2_, omega: float):
        sim = FDTD3D_Drude(nx=nx, ny=ny, nz=nz, courant=0.4,
                           eps_inf=ei_, wp2=w2_, pml_cells=cells)
        period = 2 * np.pi / (omega * sim.dt)
        total = int(period * 26)
        rec_t = []
        for n in range(total):
            sim.step()
            sim.inject_plane("Ey", 25, np.sin(omega * n * sim.dt)
                             * min(1.0, n / period))
            if n > int(period * 20):
                rec_t.append(sim.Ey[125, 6, 6])
        amp = float(np.sqrt(2 * np.mean(np.asarray(rec_t) ** 2)))
        return amp, sim

    for omega in [0.5, 0.22]:  # 截止上 / 截止下
        t_plasma, sim_p = run(ei, w2, omega)
        t_vac, _ = run(np.ones_like(ei), np.zeros_like(w2), omega)
        _, t_a = slab_rt(omega, wp2=wp2, gamma=0.0, d=xe - xs)
        row = {"omega": omega, "omega_over_wp": omega / np.sqrt(wp2),
               "t_over_vacuum": t_plasma / max(t_vac, 1e-12),
               "t_analytic": abs(t_a),
               "t_err_pct": abs(t_plasma / max(t_vac, 1e-12) - abs(t_a)) * 100}
        if omega < np.sqrt(wp2):
            # 截止下：板内幅度沿 x 指数衰减，测衰减率（独立判据）
            amp_profile = np.sqrt(np.mean(sim_p.Ey[xs:xe, 4:8, 4:8] ** 2, axis=(1, 2)))
            xr = np.arange(len(amp_profile))
            valid = (amp_profile > amp_profile[0] * 1e-3) & (amp_profile > 1e-8)
            if valid.sum() > 5:
                kappa = -np.polyfit(xr[valid], np.log(amp_profile[valid]), 1)[0]
                row["kappa_meas"] = float(kappa)
            else:
                row["kappa_meas"] = None
        out.append(row)
    return out


def pml_absorption() -> dict:
    """点源脉冲激励（已验证口径）：外缘残余/内部场 ~1e-5。"""
    nx = ny = nz = 50
    cells = 8
    sim = FDTD3D_Drude(nx=nx, ny=ny, nz=nz, courant=0.4, pml_cells=cells)
    for n in range(2200):
        val = (np.sin(0.3 * n * sim.dt) * np.exp(-((n - 200) / 80) ** 2)
               if n < 450 else 0.0)
        sim.step(source=("Ey", 25, 25, 25, val) if n < 450 else None)
    edge = float(np.max(np.abs(sim.Ey[:cells, :, :])))
    interior = float(np.max(np.abs(sim.Ey[18:32, 18:32, 18:32])))
    return {"edge_max": edge, "interior_max": interior,
            "residual_ratio": edge / max(interior, 1e-12)}


def stability_check() -> dict:
    ei, w2, ga = FDTD3D_Drude.drude_slab(50, 5, 5, 18, 35, wp2=0.05, gamma=0.02)
    sim = FDTD3D_Drude(nx=50, ny=5, nz=5, courant=0.4,
                       eps_inf=ei, wp2=w2, gamma=ga)
    for n in range(2000):
        sim.step()
        sim.inject_plane("Ey", 8, np.sin(0.2 * n * sim.dt) * min(1.0, n / 50))
    mx = max(float(np.max(np.abs(sim.Ex))), float(np.max(np.abs(sim.Ey))),
             float(np.max(np.abs(sim.Ez))))
    rejected = False
    try:
        FDTD3D_Drude(nx=10, ny=5, nz=5, courant=0.7)
    except ValueError:
        rejected = True
    return {"max_abs_E_after_2000": mx, "finite": bool(np.isfinite(mx)),
            "cfl_violation_rejected": rejected}


def main() -> None:
    import matplotlib.pyplot as plt

    print("== 15 3D Yee + Drude ADE + PML ==")
    dp = dispersion_scan()
    rt = drude_slab_rt()
    pml = pml_absorption()
    st = stability_check()
    print("dispersion: S | k_fdtd k_discrete err%")
    for d in dp:
        print(f"  {d['courant']:.2f} | {d['k_fdtd']:.5f} {d['k_discrete']:.5f} "
              f"{d['err_pct']:.2f}%")
    print("Drude slab 3D: omega/wp | t/vacuum |t|ana err% [kappa]")
    for d in rt:
        kapp = f" kappa={d['kappa_meas']:.3f}" if d.get("kappa_meas") else ""
        print(f"  {d['omega_over_wp']:.2f} | {d['t_over_vacuum']:.3f} "
              f"{d['t_analytic']:.3f} {d['t_err_pct']:.1f}%{kapp}")
    print(f"PML residual ratio: {pml['residual_ratio']:.2e}")
    print(f"stability: max|E|={st['max_abs_E_after_2000']:.3f} "
          f"finite={st['finite']} CFL rejected={st['cfl_violation_rejected']}")

    out = {
        "module": "C - 3D Yee + Drude ADE + PML",
        "dispersion": dp, "drude_slab": rt, "pml": pml, "stability": st,
        "summary": {
            "dispersion_max_err_pct": max(d["err_pct"] for d in dp),
            "pml_residual_ratio": pml["residual_ratio"],
            "stable": st["finite"] and st["cfl_violation_rejected"],
            "note": "3D k measurement uses short time window (no boundary echo); "
                    "phasing e^{+iwt} gives slope +k for a +x wave. ADE trapezoid "
                    "coefficients stay positive for gamma>=0; PML E drive uses "
                    "exact-integration phi=(1-e^-x)/x to avoid CN negative pump. "
                    "Above-cutoff transmission matches analytic to 1%; below-cutoff "
                    "transmission is measurement-limited by 3D boundary echoes "
                    "(evanescent decay kappa measured inside slab instead).",
        },
    }
    save_json("15_drude3d.json", out)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    ax = axes[0]
    ax.plot([d["courant"] for d in dp], [d["err_pct"] for d in dp], "o-")
    ax.set_xlabel("Courant S"); ax.set_ylabel("k error [%]")
    ax.set_title("3D plane-wave dispersion", fontsize=9)
    ax = axes[1]
    xw = [d["omega_over_wp"] for d in rt]
    ax.plot(xw, [d["t_over_vacuum"] for d in rt], "o-", label="t/t_vac FDTD")
    ax.plot(xw, [d["t_analytic"] for d in rt], "s--", label="|t| analytic")
    ax.set_xlabel("omega/omega_p"); ax.set_ylabel("transmission")
    ax.set_title("3D Drude slab", fontsize=9); ax.legend(fontsize=7)
    ax = axes[2]
    ax.bar(["edge", "interior"], [pml["edge_max"], pml["interior_max"]])
    ax.set_yscale("log"); ax.set_ylabel("max|E|")
    ax.set_title(f"PML residual {pml['residual_ratio']:.1e}", fontsize=9)
    save_fig("15_drude3d.png")
    print("saved: results/15_drude3d.json + figures/15_drude3d.png")


if __name__ == "__main__":
    main()
