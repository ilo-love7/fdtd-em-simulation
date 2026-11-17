"""色散介质解析理论：Drude 复介电常数、复 Fresnel、Fabry–Pérot 板 r/t。

符号约定：光学/物理常用 e^{−jωt}，Drude
    ε_r(ω) = eps_inf − ω_p²/(ω² + jγω)
当 γ>0, ω>0 时 Im ε_r > 0（损耗）。n = sqrt(ε_r) 取主支（Im n ≥ 0）。
脚本验证以 |r|、|t| 为首要判据（对传播符号约定不敏感），相位作辅助。
"""
from __future__ import annotations

import numpy as np


def drude_eps_r(omega: float | np.ndarray, wp2: float, gamma: float,
                eps_inf: float = 1.0) -> complex | np.ndarray:
    """Drude 相对介电常数 ε_r(ω) = eps_inf − ω_p²/(ω² + jγω)。"""
    return eps_inf - wp2 / (omega**2 + 1j * gamma * omega)


def drude_n(omega: float | np.ndarray, wp2: float, gamma: float,
            eps_inf: float = 1.0) -> complex | np.ndarray:
    """复折射率 n = sqrt(ε_r)，取 Im n ≥ 0 主支（对应损耗/倏逝衰减）。"""
    eps = drude_eps_r(omega, wp2, gamma, eps_inf)
    n = np.sqrt(np.asarray(eps, dtype=complex))
    n = np.where(n.imag < 0, -n, n)
    return n if np.ndim(omega) else complex(n)


def fresnel_r_complex(n1: complex, n2: complex) -> complex:
    """正入射复电场反射系数 r = (n1 − n2)/(n1 + n2)。"""
    return (n1 - n2) / (n1 + n2)


def fresnel_t_complex(n1: complex, n2: complex) -> complex:
    """正入射复电场透射系数 t = 2 n1/(n1 + n2)。"""
    return 2.0 * n1 / (n1 + n2)


def slab_rt(omega: float, wp2: float, gamma: float, d: float,
            eps_inf: float = 1.0, c0: float = 1.0) -> tuple[complex, complex]:
    """真空中厚度 d 的 Drude 板（两侧真空 n=1）的正入射复 r、t（Airy 公式）。

    n2 = n(ω)；r12=(1−n2)/(1+n2)，r23=−r12，t12=2/(1+n2)，t23=2 n2/(1+n2)；
    相位 φ = n2·(ω/c0)·d。
        r = (r12 + r23 e^{2jφ})/(1 + r12 r23 e^{2jφ})
        t = t12 t23 e^{jφ}/(1 + r12 r23 e^{2jφ})
    """
    n2 = complex(drude_n(omega, wp2, gamma, eps_inf))
    phi = n2 * (omega / c0) * d
    r12 = fresnel_r_complex(1.0 + 0j, n2)
    r23 = -r12
    t12 = fresnel_t_complex(1.0 + 0j, n2)
    t23 = fresnel_t_complex(n2, 1.0 + 0j)
    damp = 1.0 + r12 * r23 * np.exp(2j * phi)
    r = (r12 + r23 * np.exp(2j * phi)) / damp
    t = t12 * t23 * np.exp(1j * phi) / damp
    return r, t


def plasma_cutoff_wavelength(wp: float, c0: float = 1.0) -> float:
    """等离子体截止波长 λ_p = 2π c0/ω_p（ω<ω_p 即 λ>λ_p 倏逝）。"""
    return 2.0 * np.pi * c0 / wp


def group_index(omega: float, wp2: float, gamma: float = 0.0,
                eps_inf: float = 1.0) -> float:
    """无损 Drude 群折射率 n_g = n + ω dn/dω（γ=0，n²=eps_inf−wp2/ω²）。"""
    n2 = eps_inf - wp2 / omega**2
    if n2 <= 0:
        return float("nan")
    n = np.sqrt(n2)
    dndw = wp2 / (n * omega**3)   # d/dω sqrt(eps_inf−wp2/ω²)
    return float(n + omega * dndw)
