"""二维 TMz 圆柱 Mie 级数解析解（v1.3 模块 A）。

无限长介质圆柱（半径 a、折射率 m），TMz 极化（Ez 沿柱轴），
单位幅度平面波入射。展开（e^{+ikx} 入射约定）：

    E_inc = Σ_n i^n J_n(k r) e^{i n φ}
    E_scat = Σ_n i^n a_n H_n^{(2)}(k r) e^{i n φ}
    a_n = [J_n(m x) J_n'(x) − m J_n'(m x) J_n(x)] /
          [J_n(m x) H_n^{(2)'}(x) − m J_n'(m x) H_n^{(2)}(x)],  x=k a

对应 FDTD 入射 sin(ωt−kx)（相量 ∝ e^{−ikx}）时，散射方向图相对
e^{+ikx} 约定做 φ→φ+π 镜像。
"""
from __future__ import annotations

import numpy as np
from scipy.special import h2vp, hankel2, jv, jvp


def mie_coeffs(k: float, a: float, m: float, n_max: int = 25) -> dict[int, complex]:
    """返回 a_n 系数字典（含 i^n 相位，散射 = Σ c_n H_n^{(2)}(k r) e^{inφ}）。"""
    x = k * a
    coeffs: dict[int, complex] = {}
    for n in range(-n_max, n_max + 1):
        jn = jv(n, x)
        jmn = jv(n, m * x)
        num = jmn * jvp(n, x) - m * jvp(n, m * x) * jn
        den = jmn * h2vp(n, x) - m * jvp(n, m * x) * hankel2(n, x)
        coeffs[n] = (1j) ** n * (-num / den)
    return coeffs


def mie_scattered_field(k: float, a: float, m: float, r: float, phi: float,
                        coeffs: dict[int, complex] | None = None) -> complex:
    """散射场复幅度（e^{+ikx} 入射约定）。"""
    if coeffs is None:
        coeffs = mie_coeffs(k, a, m)
    return sum(cn * hankel2(n, k * r) * np.exp(1j * n * phi)
               for n, cn in coeffs.items())
