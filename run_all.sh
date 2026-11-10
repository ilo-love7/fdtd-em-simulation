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
echo "=== pytest ==="
"$PY" -m pytest tests/ -q
echo "=== 05 汇总 ==="
"$PY" scripts/05_summary.py
echo "ALL DONE"
