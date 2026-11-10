"""05 汇总：聚合 01–08 实验 JSON → results/00_summary.json 并打印对照表。"""
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
    v11 = {}
    for name in ("06_pml_validation.json", "07_photonic_crystal.json",
                 "08_convergence.json"):
        try:
            v11[name] = load(name)
        except FileNotFoundError:
            v11[name] = None

    summary = {
        "project": "N8 FDTD 电磁/光子学仿真",
        "solver": ["手写 1D Yee (Ez-Hy)", "手写 2D TMz Yee",
                   "flaport/fdtd 0.3.5 (Bragg)", "匹配有耗 PML 层 (v1.1)"],
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
        "metrics_v11": {
            "pml_energy_residual": (v11["06_pml_validation.json"] or {}).get("energy_residual"),
            "pml_envelope_r_open": ((v11["06_pml_validation.json"] or {})
                                    .get("envelope_vswr", {})).get("pml_open", {}).get("r_envelope"),
            "pml_2d_standing_wave_abs_r": ((v11["06_pml_validation.json"] or {})
                                           .get("standing_wave_2d", {})).get("r_abs"),
            "pml_2d_node_spacing_err_pct": ((v11["06_pml_validation.json"] or {})
                                            .get("standing_wave_2d", {})).get("spacing_err_pct"),
            "photonic_gap_center_over_a": ((v11["07_photonic_crystal.json"] or {})
                                           .get("a1", {})).get("gap", {}).get("center_over_a")
            if v11["07_photonic_crystal.json"] else None,
            "fresnel_convergence_order": ((v11["08_convergence.json"] or {})
                                          .get("fresnel", {})).get("observed_order"),
        },
        "files": {
            "results": ["01_1d_validation.json", "02_2d_metal_reflection.json",
                        "03_2d_slit_diffraction.json", "04_bragg_mirror.json",
                        "06_pml_validation.json", "07_photonic_crystal.json",
                        "08_convergence.json"],
            "figures": ["01_1d_validation.png", "02_2d_metal_reflection.png",
                        "03_2d_slit_diffraction.png", "04_bragg_mirror.png",
                        "06_pml_validation.png", "07_photonic_crystal.png",
                        "08_convergence.png"],
        },
    }
    save_json("00_summary.json", summary)

    m = summary["metrics"]
    v = summary["metrics_v11"]
    print("=" * 62)
    print(f"{'N8 FDTD 验证汇总 (v1.0 + v1.1)':^56}")
    print("=" * 62)
    print(f"1D 传播速度误差          : {m['speed_err_pct']:.3f} %")
    print(f"1D PEC |r|               : {m['pec_abs_r']:.4f} (phase err "
          f"{m['pec_phase_err_deg']:.1f} deg)")
    for k, fv in m["fresnel_err_pct"].items():
        print(f"1D Fresnel {k:>6} 误差 : {fv:.3f} %")
    print(f"2D 驻波波节间距误差      : {m['standing_wave_node_spacing_err_pct']:.2f} %"
          f"  (隐含 |r|={m['standing_wave_implied_abs_r']:.3f}, v1.0 口径)")
    print(f"2D 单缝极小值 Δsinθ(均值): {m['slit_minima_mean_abs_err_sin_theta']:.4f}"
          f"  图案 RMS {m['slit_pattern_rms_err']:.4f}")
    print(f"Bragg 带隙中心误差        : {m['bragg_bandgap_center_err_pct']:.3f} %"
          f"  T_min {m['bragg_T_min_in_gap']:.4f}")
    print("-" * 62)
    print(f"[v1.1] PML 平滑脉冲残余    : {v['pml_energy_residual']:.2e}")
    print(f"[v1.1] PML 开放包络 |r|    : {v['pml_envelope_r_open']:.4f}")
    if v["pml_2d_standing_wave_abs_r"] is not None:
        print(f"[v1.1] 2D 驻波拟合 |r|     : {v['pml_2d_standing_wave_abs_r']:.4f}"
              f"  (波节间距误差 {v['pml_2d_node_spacing_err_pct']:.2f} %)")
    if v["photonic_gap_center_over_a"] is not None:
        print(f"[v1.1] 光子晶体带隙中心    : {v['photonic_gap_center_over_a']:.3f} a")
    if v["fresnel_convergence_order"] is not None:
        print(f"[v1.1] Fresnel 收敛阶      : {v['fresnel_convergence_order']:.2f}")
    print("=" * 62)
    print("saved: results/00_summary.json")


if __name__ == "__main__":
    main()
