#!/usr/bin/env bash
# N8 FDTD 一键复现：01–08 实验 → 单测 → 汇总
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

echo "=== 01 一维 FDTD 验证 ==="
"$PY" scripts/01_1d_validation.py
echo "=== 02 二维金属反射（驻波） ==="
"$PY" scripts/02_2d_metal_reflection.py
echo "=== 03 二维单缝衍射 ==="
"$PY" scripts/03_2d_slit_diffraction.py
echo "=== 04 Bragg 镜透射谱（fdtd 库，较长） ==="
"$PY" scripts/04_bragg_mirror.py
echo "=== 06 PML 吸收层验证（v1.1） ==="
"$PY" scripts/06_pml_validation.py
echo "=== 07 光子晶体带隙（v1.1，较长） ==="
"$PY" scripts/07_photonic_crystal.py
echo "=== 08 收敛阶研究（v1.1，较长） ==="
"$PY" scripts/08_convergence.py
echo "=== 09 Drude 色散介质（v1.2） ==="
"$PY" scripts/09_drude.py
echo "=== 10 TFSF 总场/散射场源（v1.2） ==="
"$PY" scripts/10_tfsf.py
echo "=== 11 数值色散相/群速度（v1.2） ==="
"$PY" scripts/11_dispersion.py
echo "=== 13 TFSF 斜入射+2D Mie（v1.3） ==="
"$PY" scripts/13_mie_tfsf.py
echo "=== 14 各向同性色散补偿（v1.3） ==="
"$PY" scripts/14_dispersion_comp.py
echo "=== 15 3D Yee+Drude+PML（v1.3） ==="
"$PY" scripts/15_drude3d.py
echo "=== pytest ==="
"$PY" -m pytest tests/ -q
echo "=== 05 汇总 ==="
"$PY" scripts/05_summary.py
echo "=== 12 v1.2 汇总 ==="
"$PY" scripts/12_summary_v12.py
echo "=== 16 v1.3 汇总 ==="
"$PY" scripts/16_summary_v13.py
echo "ALL DONE"
