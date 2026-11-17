"""Yee 网格数值色散解析理论（v1.2）：1D/2D 离散色散、相速度、群速度、各向异性。

1D Yee（Courant S=c·dt/dx）：
    sin(ω dt/2) = S sin(k dx/2)
    k_num = (2/dx) arcsin( S sin(ω dt/2) )
    v_ph/c = Ω/(k dx)，      v_g/c = cos(k dx/2)/cos(ω dt/2)
其中 Ω = ω dx/c 为无量纲频率。

2D Yee（正方网格 dx=dy，传播角 θ 相对 x 轴）：
    sin²(ω dt/2) = S²[ sin²(k_x dx/2) + sin²(k_y dx/2) ]，k_x=k cosθ, k_y=k sinθ
沿轴 θ=0 退化为 1D；沿对角 θ=45°：sin(ω dt/2)=S√2 sin(k dx/(2√2))。
各向异性 = v_ph(θ=0) − v_ph(θ=45°)（同一 Ω、S）。2D 稳定上限 S≤1/√2。
"""
from __future__ import annotations

import numpy as np

from ..fdtd_core.yee1d import C0 as _C0  # 物理光速，与 FDTD 求解器同口径


# ---------- 1D ----------
def k_discrete_1d(omega: float, dx: float, courant: float, c0: float = _C0) -> float:
    """1D Yee 离散波数 k_num（满足离散色散关系的物理解）。"""
    # Yee 1D: sin(ω dt/2) = S sin(k dx/2)  ⇒  sin(k dx/2) = sin(ω dt/2)/S
    omega_dt_half = omega * (courant * dx / c0) / 2.0
    arg = np.sin(omega_dt_half) / courant
    return 2.0 * np.arcsin(np.clip(arg, -1.0, 1.0)) / dx


def phase_velocity_1d(omega: float, dx: float, courant: float, c0: float = _C0) -> float:
    """1D 数值相速度 v_ph = ω/k_num。"""
    return omega / k_discrete_1d(omega, dx, courant, c0)


def group_velocity_1d(omega: float, dx: float, courant: float, c0: float = _C0) -> float:
    """1D 数值群速度 v_g = dω/dk = c0·cos(k dx/2)/cos(ω dt/2)。"""
    dt = courant * dx / c0
    kdx2 = k_discrete_1d(omega, dx, courant, c0) * dx / 2.0
    return c0 * np.cos(kdx2) / np.cos(omega * dt / 2.0)


# ---------- 2D ----------
def k_discrete_2d(omega: float, dx: float, courant: float, theta: float,
                  c0: float = _C0) -> float:
    """2D Yee 离散波数（给定传播角 θ，解 sin²(ωdt/2)=S²[sin²(kx dx/2)+sin²(ky dx/2)]）。

    对小 arg 用 Newton 迭代（单调可解）；k>0 物理解。
    """
    dt = courant * dx / c0
    rhs = (np.sin(omega * dt / 2.0) / courant) ** 2
    cx, cy = np.cos(theta), np.sin(theta)

    def f(k):
        return (np.sin(k * dx * cx / 2.0) ** 2 + np.sin(k * dx * cy / 2.0) ** 2) - rhs

    k_phys = omega / c0
    k = k_phys
    for _ in range(60):
        a, b = k * dx * cx / 2.0, k * dx * cy / 2.0
        fk = np.sin(a) ** 2 + np.sin(b) ** 2 - rhs
        # d/dk sin²(k dx c/2) = dx c sin(k dx c/2) cos(k dx c/2) = (dx c/2) sin(k dx c)
        dfdk = (dx * cx / 2.0) * np.sin(2 * a) + (dx * cy / 2.0) * np.sin(2 * b)
        if abs(dfdk) < 1e-14:
            break
        step = fk / dfdk
        k -= step
        if abs(step) < 1e-14:
            break
    return float(k)


def phase_velocity_2d(omega: float, dx: float, courant: float, theta: float,
                      c0: float = _C0) -> float:
    return omega / k_discrete_2d(omega, dx, courant, theta, c0)


def group_velocity_2d(omega: float, dx: float, courant: float, theta: float,
                      c0: float = _C0) -> float:
    """2D 群速度大小 v_g = |∇_k ω|（由隐函数微分）。"""
    dt = courant * dx / c0
    k = k_discrete_2d(omega, dx, courant, theta, c0)
    kx, ky = k * np.cos(theta), k * np.sin(theta)
    ax, ay = kx * dx / 2.0, ky * dx / 2.0
    # 隐微分 sin²(ωdt/2)=S²[sin²ax+sin²ay]，a=k·dx/2：
    # cos(ωdt/2)(dt/2)·2 dω = S²·dx[sin(kx dx)/2 dkx + sin(ky dx)/2 dky]
    # ⇒ dω/dkx = c0·sin(kx dx)/(2 cos(ωdt/2))；HOWEVER S²·dx/dt=c0（非 S dx/dt）
    pref = (courant**2 * dx / dt) / (2.0 * np.cos(omega * dt / 2.0))  # = c0/(2cos)
    dw_dkx = pref * np.sin(kx * dx)
    dw_dky = pref * np.sin(ky * dx)
    return float(np.hypot(dw_dkx, dw_dky))


def anisotropy_2d(omega: float, dx: float, courant: float, c0: float = _C0) -> dict:
    """同一 (Ω,S) 下沿轴与沿对角的相/群速度差。"""
    vp_ax = phase_velocity_2d(omega, dx, courant, 0.0, c0)
    vp_dg = phase_velocity_2d(omega, dx, courant, np.pi / 4, c0)
    vg_ax = group_velocity_2d(omega, dx, courant, 0.0, c0)
    vg_dg = group_velocity_2d(omega, dx, courant, np.pi / 4, c0)
    return {
        "vp_axis": vp_ax, "vp_diag": vp_dg, "vp_anisotropy": vp_ax - vp_dg,
        "vg_axis": vg_ax, "vg_diag": vg_dg, "vg_anisotropy": vg_ax - vg_dg,
    }


def omega_from_k(k: float, dx: float, courant: float, theta: float,
                 c0: float = _C0) -> float:
    """离散色散的反向关系：给定 k,θ 求 ω（用于波包群速度对照）。"""
    dt = courant * dx / c0
    s2 = (np.sin(k * dx * np.cos(theta) / 2.0) ** 2
          + np.sin(k * dx * np.sin(theta) / 2.0) ** 2)
    return 2.0 * np.arcsin(np.clip(courant * np.sqrt(s2), -1.0, 1.0)) / dt
