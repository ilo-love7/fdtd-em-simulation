"""12 v1.2 汇总：聚合 09/10/11 JSON → results/12_summary_v12.json 并打印对照。"""
from __future__ import annotations

import json

from common import RESULTS, save_json


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    d09 = load("09_drude.json")
    d10 = load("10_tfsf.json")
    d11 = load("11_dispersion.json")
    s09, s10, s11 = d09["summary"], d10["summary"], d11["summary"]

    summary = {
        "project": "N8 FDTD (v1.2 dispersive media / TFSF / numerical dispersion)",
        "results": {
            "drude_rt_max_abs_deviation": s09["rt_max_abs_deviation"],
            "drude_phase_velocity_max_err_pct": s09["phase_velocity_max_err_pct_omega_ge_015"],
            "drude_absorption_max_deviation": s09["absorption_max_abs_deviation"],
            "tfsf_leak_pct_range": s10["leak_pct_range"],
            "tfsf_reflection_err_pct_best": s10["reflection_err_pct_best"],
            "tfsf_ablation_full": d10["ablation"]["scat_rms_full"],
            "tfsf_ablation_no_h": d10["ablation"]["scat_rms_no_h_correction"],
            "dispersion_phase_max_err_discrete_pct": s11["phase_max_err_vs_discrete_pct"],
            "dispersion_group_max_err_pct": s11["group_max_err_pct"],
            "dispersion_anisotropy_omega05_S0307": [
                d["vp_anisotropy"] for d in d11["anisotropy_2d"]
                if d["Omega"] == 0.5],
        },
        "files": {
            "results": ["09_drude.json", "10_tfsf.json", "11_dispersion.json"],
            "figures": ["09_drude.png", "10_tfsf.png", "11_dispersion.png"],
        },
    }
    save_json("12_summary_v12.json", summary)

    r = summary["results"]
    print("=" * 64)
    print(f"{'N8 FDTD v1.2 深化汇总':^58}")
    print("=" * 64)
    print(f"[A Drude] r/t vs Airy max deviation      : {r['drude_rt_max_abs_deviation']:.4f}")
    print(f"[A Drude] phase velocity max err (w>=.15): {r['drude_phase_velocity_max_err_pct']:.3f}%")
    print(f"[A Drude] loss absorption max deviation  : {r['drude_absorption_max_deviation']:.4f}")
    print("-" * 64)
    print(f"[B TFSF ] scatter leak % range           : {r['tfsf_leak_pct_range'][0]:.2f}"
          f"-{r['tfsf_leak_pct_range'][1]:.2f}%")
    print(f"[B TFSF ] reflection err best vs Fresnel : {r['tfsf_reflection_err_pct_best']:.2f}%")
    print(f"[B TFSF ] ablation full vs no-H rms      : "
          f"{r['tfsf_ablation_full']:.4f} vs {r['tfsf_ablation_no_h']:.4f}")
    print("-" * 64)
    print(f"[C Disp ] phase err vs discrete max      : "
          f"{r['dispersion_phase_max_err_discrete_pct']:.4f}%")
    print(f"[C Disp ] group err vs dw/dk max         : {r['dispersion_group_max_err_pct']:.4f}%")
    print(f"[C Disp ] 2D v_ph anisotropy (Om=.5)     : {r['dispersion_anisotropy_omega05_S0307']}")
    print("=" * 64)
    print("saved: results/12_summary_v12.json")


if __name__ == "__main__":
    main()
