"""05 汇总：聚合 01–04 实验 JSON → results/00_summary.json 并打印对照表。"""
from __future__ import annotations

import json

from common import RESULTS, save_json


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    d1 = load("01_1d_validation.json")
    d2 = load("02_2d_metal_reflection.json")
    d3 = load("03_2d_slit_diffraction.json")
    d4 = load("04_bragg_mirror.json")

    summary = {
        "project": "N8 FDTD 电磁/光子学仿真",
        "solver": ["手写 1D Yee (Ez-Hy)", "手写 2D TMz Yee",
                   "flaport/fdtd 0.3.5 (Bragg)"],
        "metrics": {
            "speed_err_pct": d1["propagation_speed"]["err_pct"],
            "pec_abs_r": d1["pec_reflection"]["r_abs"],
            "pec_phase_err_deg": d1["pec_reflection"]["phase_err_deg_vs_180"],
            "fresnel_err_pct": {
                k: v["err_pct"] for k, v in d1["fresnel"].items()
            },
            "standing_wave_node_spacing_err_pct": d2["node_spacing_err_pct"],
            "standing_wave_implied_abs_r": d2["implied_abs_r_vswr"],
            "slit_minima_mean_abs_err_sin_theta": d3["minima_mean_abs_err_sin_theta"],
            "slit_pattern_rms_err": d3["pattern_rms_err_vs_theory"],
            "bragg_bandgap_center_err_pct": d4["bandgap_center_err_pct"],
            "bragg_T_min_in_gap": d4["T_min_in_gap"],
        },
        "files": {
            "results": ["01_1d_validation.json", "02_2d_metal_reflection.json",
                        "03_2d_slit_diffraction.json", "04_bragg_mirror.json"],
            "figures": ["01_1d_validation.png", "02_2d_metal_reflection.png",
                        "03_2d_slit_diffraction.png", "04_bragg_mirror.png"],
        },
    }
    save_json("00_summary.json", summary)

    m = summary["metrics"]
    print("=" * 62)
    print(f"{'N8 FDTD 验证汇总':^56}")
    print("=" * 62)
    print(f"1D 传播速度误差          : {m['speed_err_pct']:.3f} %")
    print(f"1D PEC |r|               : {m['pec_abs_r']:.4f} (phase err "
          f"{m['pec_phase_err_deg']:.1f} deg)")
    for k, v in m["fresnel_err_pct"].items():
        print(f"1D Fresnel {k:>6} 误差 : {v:.3f} %")
    print(f"2D 驻波波节间距误差      : {m['standing_wave_node_spacing_err_pct']:.2f} %"
          f"  (隐含 |r|={m['standing_wave_implied_abs_r']:.3f})")
    print(f"2D 单缝极小值 Δsinθ(均值): {m['slit_minima_mean_abs_err_sin_theta']:.4f}"
          f"  图案 RMS {m['slit_pattern_rms_err']:.4f}")
    print(f"Bragg 带隙中心误差        : {m['bragg_bandgap_center_err_pct']:.3f} %"
          f"  T_min {m['bragg_T_min_in_gap']:.4f}")
    print("=" * 62)
    print("saved: results/00_summary.json")


if __name__ == "__main__":
    main()
