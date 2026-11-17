"""各向同性 9 点 Laplacian 的色散补偿解析理论（v1.3 模块 B）。

波方程 FDTD 的色散关系：
    5 点：sin²(Ω/2) = S²[s_x² + s_y²]
    9 点：sin²(Ω/2) = S²[s_x² + s_y² - (2/3)s_x²s_y²]
其中 Ω=ω dx/c，S=c dt/dx，s_x=sin(k_x dx/2)，k_x=k cosθ。

各向异性（v_ph(θ=0) - v_ph(θ=45°)，同一 Ω、S）：
    5 点 ≈ -Ω⁴/192（或 O(Ω⁴)，对角更快）；9 点 ≈ O(Ω⁶)，显著压缩。
本模块给出精确解：给定 (Ω,S,θ) 解出 K=k dx，再得 v_ph=Ω/K·c。
"""
from __future__ import annotations

import numpy as np


def _solve_K(omega_dx_over_c: float, courant: float, theta: float,
             isotropic: bool) -> float:
    """对色散关系在 K∈(0,K_max) 用二分求解（K=k dx，无量纲）。"""
    Omega = omega_dx_over_c
    rhs = (np.sin(Omega * courant / 2.0) / courant) ** 2
    cx, cy = np.cos(theta), np.sin(theta)

    def lhs(K: float) -> float:
        sx2 = np.sin(K * cx / 2.0) ** 2
        sy2 = np.sin(K * cy / 2.0) ** 2
        val = sx2 + sy2 - (2.0 / 3.0) * sx2 * sy2 if isotropic else sx2 + sy2
        return val

    # 物理根（长波支）在 K∈(0,π)：lhs 从 0 单调升到峰值。
    # K=2π 时 sx=sy=0 又回到 0，会和 K=0 同号导致二分失效，故上界取 π。
    lo, hi = 0.0, np.pi
    f_lo = lhs(lo) - rhs  # = -rhs < 0
    f_hi = lhs(hi) - rhs
    if f_lo * f_hi > 0:
        # 无解（超出稳定或截止），返回 nan
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = lhs(mid) - rhs
        if f_lo * fm <= 0:
            hi, f_hi = mid, fm
        else:
            lo, f_lo = mid, fm
    return 0.5 * (lo + hi)


def k_discrete_wave(omega: float, dx: float, courant: float, theta: float,
                    isotropic: bool = True, c0: float = 299_792_458.0) -> float:
    K = _solve_K(omega * dx / c0, courant, theta, isotropic)
    return K / dx if np.isfinite(K) else float("nan")


def phase_velocity_wave(omega: float, dx: float, courant: float, theta: float,
                        isotropic: bool = True, c0: float = 299_792_458.0) -> float:
    K = _solve_K(omega * dx / c0, courant, theta, isotropic)
    return (omega * dx / c0) / K * c0 if np.isfinite(K) and K > 0 else float("nan")


def kx_from_ky(omega: float, dx: float, courant: float, ky: float,
               isotropic: bool = True, c0: float = 299_792_458.0) -> float:
    """给定横波数 ky（rad/m）与频率，精确解离散 kx。

    5 点：sx² = [sin(ωdt/2)/S]² − sy²
    9 点：sx² = ([sin(ωdt/2)/S]² − sy²) / (1 − (2/3)sy²)
    其中 sx=sin(kx dx/2)、sy=sin(ky dx/2)。这是 FDTD 相位列源注入的
    自然对照（源强加 ky，网格自选 kx）。
    """
    rhs = (np.sin(omega * courant * dx / (2.0 * c0)) / courant) ** 2
    sy2 = np.sin(ky * dx / 2.0) ** 2
    num = rhs - sy2
    den = (1.0 - (2.0 / 3.0) * sy2) if isotropic else 1.0
    sx2 = num / den
    if sx2 < 0 or sx2 > 1:
        return float("nan")
    return 2.0 * np.arcsin(np.sqrt(sx2)) / dx


def ky_for_angle(omega: float, dx: float, courant: float, theta: float,
                 isotropic: bool = True, c0: float = 299_792_458.0) -> float:
    """求使 kx_from_ky 的 kx==ky·cot(theta) 的 ky（轴对称传播角 θ 的注入 ky）。"""
    cot = 1.0 / np.tan(theta)

    def f(ky: float) -> float:
        kx = kx_from_ky(omega, dx, courant, ky, isotropic, c0)
        if not np.isfinite(kx):
            return 1e6
        return kx - ky * cot

    # 扫 ky 找符号变化（物理支）
    ks = np.linspace(1e-6, np.pi / dx * 0.95, 400)
    prev = f(ks[0])
    for k in ks[1:]:
        cur = f(k)
        if prev * cur <= 0:
            lo, hi = k - (ks[1] - ks[0]), k
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if f(lo) * f(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            return 0.5 * (lo + hi)
        prev = cur
    return float("nan")


def anisotropy_wave(omega: float, dx: float, courant: float, isotropic: bool,
                    c0: float = 299_792_458.0) -> dict:
    """同一 (Ω,S) 下轴向(θ=0)与对角(θ=45°)的 v_ph 差。"""
    vp_ax = phase_velocity_wave(omega, dx, courant, 0.0, isotropic, c0)
    vp_dg = phase_velocity_wave(omega, dx, courant, np.pi / 4, isotropic, c0)
    return {
        "vp_axis": float(vp_ax), "vp_diag": float(vp_dg),
        "vp_anisotropy": float(vp_ax - vp_dg),
    }
