"""PML 吸收层（κ=1 的 CPML 退化形式：电导率渐变有耗层）。

v1.0 的一阶 Mur 边界回波系数 |coef|=|(c·dt−dx)/(c·dt+dx)| ≈ 0.212（S=0.65），
会填充驻波波节、污染 |r| 测量（2D 金属反射隐含 |r| 只有 0.922）。
本模块用多项式渐变电导率 σ(d)=σ_max·(d/d_pml)^m 吸收行波：

    ε ∂E/∂t + σ E = curl H        （κ=1、α=0 的 CPML / 有耗匹配层）

离散采用 Crank–Nicolson 平均（无条件稳定、σ=0 时精确退回标准格式）：

    E^{n+1} = [E^n (1 − σΔt/2ε) + (Δt/ε)·curl H] / (1 + σΔt/2ε)

σ_max 经典取值 0.8(m+1)/(η0·dx)（Taflove），10 格厚度、m=4 时
单程反射 ~1e-6 量级；外缘用零阶镜像闭合（波在到达外缘前已被吸收）。

离散格式注意（v1.1 教训）：Crank–Nicolson 平均在 σΔt/2ε > 1 时
系数 (1−x/2)/(1+x/2) 变负——E 场逐步反号，与蛙跳 H 耦合成参数泵浦，
能量先衰减后回升。改用**精确积分（指数）格式**：

    E^{n+1} = e^{−x} E^n + (Δt/ε)·curl H·φ(x),  x = σΔt/ε,
    φ(x) = (1 − e^{−x})/x   （x→0 时 → 1，级数展开防消去）

系数恒正、无条件稳定，σ=0 时精确退回标准格式。
"""
from __future__ import annotations

import numpy as np

ETA0 = 376.730313668  # 真空波阻抗 η0 = sqrt(μ0/ε0)


def sigma_max(dx: float, m: int = 4) -> float:
    """Taflove 经典取值：10 格厚度时目标反射 ~1e-6。"""
    return 0.8 * (m + 1) / (ETA0 * dx)


def sigma_profile_1d(nx: int, cells: int, dx: float, m: int = 4) -> np.ndarray:
    """一维两侧 PML 电导率剖面。cells=0 返回全零。

    σ 按**距接口深度**幂律渐变：接口处 σ=0，外缘 σ=σ_max，平滑单调——
    剖面若在外缘突跳（如按格心错位采样）会因阻抗失配产生大反射。
    """
    sig = np.zeros(nx)
    if cells <= 0:
        return sig
    smax = sigma_max(dx, m)
    if cells == 1:
        sig[0] = smax
        sig[-1] = smax
        return sig
    # 深度因子：接口 0 → 外缘 1
    prof = (np.arange(cells) / (cells - 1)) ** m   # 左侧翻转使用
    sig[:cells] = smax * prof[::-1]                 # 左：外缘(0)=σmax，接口=0
    sig[nx - cells:] = smax * prof                  # 右：接口=0，外缘=σmax
    return sig


def sigma_profile_2d(nx: int, ny: int, cells_x: int, cells_y: int, dx: float,
                     m: int = 4) -> np.ndarray:
    """二维 PML 电导率图（x/y 剖面取最大值，角落覆盖）。"""
    sig = np.zeros((nx, ny))
    if cells_x > 0:
        px = sigma_profile_1d(nx, cells_x, dx, m)
        sig[:, :] = np.maximum(sig, px[:, None])
    if cells_y > 0:
        py = sigma_profile_1d(ny, cells_y, dx, m)
        sig[:, :] = np.maximum(sig, py[None, :])
    return sig


def lossy_coeffs(sigma: np.ndarray, dt: float, eps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """精确积分有耗格式的系数 (decay, drive)。

    E^{n+1} = decay·E^n + drive·(Δt/(ε Δx))·ΔH   其中
      decay = e^{−x},  drive = φ(x) = (1−e^{−x})/x,  x = σΔt/ε
    """
    x = sigma * dt / eps
    decay = np.exp(-x)
    small = x < 1e-8
    phi = np.where(small, 1.0 - x / 2.0 + x**2 / 6.0, (1.0 - decay) / np.where(small, 1.0, x))
    return decay, phi


def matched_layer_note() -> str:
    """匹配条件说明：σ_m/μ0 = σ_e/ε0（归一化电导率相等）时
    η = sqrt((jωμ0+σ_m)/(jωε0+σ_e)) ≡ η0——连续介质中任意角度/频率
    无界面反射，波在层内按 e^{−σ t/ε0} 衰减。1D 即 Berenger PML 的
    等价形式。数值上 E、H 用同一衰减因子 x=σ_eΔt/ε0。
    """
    return "matched"
