"""16 v1.3 汇总：聚合 13/14/15 JSON → results/16_summary_v13.json 并打印对照。"""
from __future__ import annotations

import json

from common import RESULTS, save_json


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    d13 = load("13_mie_tfsf.json")
    d14 = load("14_dispersion_comp.json")
    d15 = load("15_drude3d.json")
    s13, s14, s15 = d13["summary"], d14["summary"], d15["summary"]

    summary = {
        "project": "N8 FDTD (v1.3 TFSF oblique+Mie / dispersion compensation / 3D Drude+PML)",
        "results": {
            "mie_tfsf_leak_max_pct": s13["leak_max_pct"],
            "mie_tfsf_ablation_amp_min": s13["ablation_amplification_min"],
            "mie_pattern_corr": s13["mie_corr"],
            "mie_pattern_rel_err_pct": s13["mie_rel_err_pct"],
            "disp_compression_median": s14["compression_median"],
            "disp_compression_range": [s14["compression_min"], s14["compression_max"]],
            "disp_fdtd_residual_max_pct": s14["fdtd_dispersion_residual_max_pct"],
            "drude3d_dispersion_err_max_pct": s15["dispersion_max_err_pct"],
            "drude3d_pml_residual_ratio": s15["pml_residual_ratio"],
            "drude3d_stable": s15["stable"],
        },
        "files": {
            "results": ["13_mie_tfsf.json", "14_dispersion_comp.json", "15_drude3d.json"],
            "figures": ["13_mie_tfsf.png", "14_dispersion_comp.png", "15_drude3d.png"],
        },
    }
    save_json("16_summary_v13.json", summary)

    r = summary["results"]
    print("=" * 64)
    print(f"{'N8 FDTD v1.3 深化汇总':^56}")
    print("=" * 64)
    print(f"[A Mie  ] TFSF leak max                  : {r['mie_tfsf_leak_max_pct']:.2f}%")
    print(f"[A Mie  ] ablation amplification (min)   : {r['mie_tfsf_ablation_amp_min']:.1f}x")
    print(f"[A Mie  ] 2D Mie pattern corr / rel err  : {r['mie_pattern_corr']:.4f} / "
          f"{r['mie_pattern_rel_err_pct']:.2f}%")
    print("-" * 64)
    print(f"[B Disp ] anisotropy compression median  : {r['disp_compression_median']:.0f}x "
          f"({r['disp_compression_range'][0]:.0f}-{r['disp_compression_range'][1]:.0f}x)")
    print(f"[B Disp ] FDTD dispersion residual max   : {r['disp_fdtd_residual_max_pct']:.3f}%")
    print("-" * 64)
    print(f"[C 3D   ] plane-wave dispersion err max  : {r['drude3d_dispersion_err_max_pct']:.2f}%")
    print(f"[C 3D   ] PML residual ratio             : {r['drude3d_pml_residual_ratio']:.2e}")
    print(f"[C 3D   ] stable (finite + CFL reject)   : {r['drude3d_stable']}")
    print("=" * 64)
    print("saved: results/16_summary_v13.json")


if __name__ == "__main__":
    main()
