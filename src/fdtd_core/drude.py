"""一维 Drude 色散介质 FDTD（Ez–Hy + 辅助微分方程 ADE，v1.2）。

物理：Drude 介质的复介电常数 ε(ω)/ε0 = eps_inf − ω_p²/(ω² + jγω)。
引入极化电流 J（与 Ez 同位形），在时域满足辅助微分方程
    ∂J/∂t + γ J = ε0 ω_p² E
Maxwell 安培定律写成
    ∂E/∂t = (1/(ε0 eps_inf)) [ ∇×H − J ]

采用蛙跳 + 梯形（Crank–Nicolson）离散 J 到半步：
    J^{n+1/2} = [(1 − γΔt/2) J^{n−1/2} + ε0 ω_p² Δt E^n] / (1 + γΔt/2)
    E^{n+1}   = E^n + Δt/(ε0 eps_inf) [ (∇×H)^{n+1/2}/dx − J^{n+1/2} ]

数值约定（与 v1.0/v1.1 手写求解器一致）：差分用**未除 dx 的节点差**，
系数 dt/(ε·dx)，避免双重除 dx 发散。为量纲干净，本模块用归一化单位
dx=1、c0=1、ε0=1、μ0=1（ω_p、γ 以「每格时间」rad 计）。

验证（scripts/09）：等离子体截止、Fabry–Pérot 板 r/t vs 解析、相速度。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .boundaries import mur1_update

C0 = 1.0
EPS0 = 1.0
MU0 = 1.0


@dataclass
class FDTD1D_Drude:
    """一维 TM 平面波 FDTD，支持位置相关的 Drude 色散介质。

    参数
    ----
    nx, dx        : 网格点数、空间步长（默认 dx=1 归一化）
    courant       : S = c0·dt/dx，须 ≤ 1
    eps_inf       : 高频介电常数（标量或长度 nx 数组），默认 1
    wp2           : 等离子体频率平方 ω_p²（标量或长度 nx 数组），默认 0（真空）
    gamma         : 碰撞频率 γ（标量或长度 nx 数组），默认 0
    """

    nx: int
    dx: float = 1.0
    courant: float = 0.95
    eps_inf: float | np.ndarray = 1.0
    wp2: float | np.ndarray = 0.0
    gamma: float | np.ndarray = 0.0

    Ez: np.ndarray = field(init=False, repr=False)
    Hy: np.ndarray = field(init=False, repr=False)
    Jz: np.ndarray = field(init=False, repr=False)
    dt: float = field(init=False)
    n_steps: int = field(default=0, init=False)
    _Ez_old: np.ndarray = field(init=False, repr=False)
    _eps_inf: np.ndarray = field(init=False, repr=False)
    _wp2: np.ndarray = field(init=False, repr=False)
    _gamma: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0 < self.courant <= 1.0):
            raise ValueError(f"1D CFL 要求 0 < courant ≤ 1，收到 {self.courant}")
        self.dt = self.courant * self.dx / C0
        n = self.nx
        self._eps_inf = np.broadcast_to(np.asarray(self.eps_inf, float), (n,)).copy()
        self._wp2 = np.broadcast_to(np.asarray(self.wp2, float), (n,)).copy()
        self._gamma = np.broadcast_to(np.asarray(self.gamma, float), (n,)).copy()
        if np.any(self._eps_inf <= 0):
            raise ValueError("eps_inf 须为正")
        self.Ez = np.zeros(n)
        self.Hy = np.zeros(n - 1)
        self.Jz = np.zeros(n)
        self._Ez_old = np.zeros(n)
        self._mur_coef = (C0 * self.dt - self.dx) / (C0 * self.dt + self.dx)

    # ---------- 材料辅助 ----------
    @staticmethod
    def drude_box(nx: int, x0: int, x1: int, wp2: float, gamma: float,
                  eps_inf: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在 [x0, x1) 铺设 Drude 介质，返回 (eps_inf, wp2, gamma) 数组。"""
        ei = np.ones(nx)
        w2 = np.zeros(nx)
        ga = np.zeros(nx)
        ei[x0:x1] = eps_inf
        w2[x0:x1] = wp2
        ga[x0:x1] = gamma
        return ei, w2, ga

    # ---------- 步进 ----------
    def step_h(self) -> None:
        """Hy 半步（无耗项；介质色散只在 E 方程出现）。"""
        diff = self.Ez[1:] - self.Ez[:-1]
        self.Hy += (self.dt / (MU0 * self.dx)) * diff

    def step_e(self, source: float = 0.0, source_index: int | None = None,
               soft: bool = True, pec_mask: np.ndarray | None = None) -> None:
        """Ez 整步 + ADE 极化电流更新 + 一阶 Mur 吸收边界。"""
        self._Ez_old[:] = self.Ez
        # 1) ADE：J^{n+1/2} 由 E^n 更新（梯形格式，系数恒正稳定）
        g = self._gamma * self.dt / 2.0
        self.Jz = ((1.0 - g) * self.Jz + EPS0 * self._wp2 * self.dt * self.Ez) / (1.0 + g)
        # 2) E 更新：∇×H 内部节点，边缘 Mur
        new = self.Ez.copy()
        coef = self.dt / (EPS0 * self._eps_inf * self.dx)
        new[1:-1] += coef[1:-1] * (self.Hy[1:] - self.Hy[:-1])
        new -= (self.dt / (EPS0 * self._eps_inf)) * self.Jz
        self.Ez = new
        mur1_update(self.Ez, self._Ez_old, self._mur_coef, axis=0, side="lo")
        mur1_update(self.Ez, self._Ez_old, self._mur_coef, axis=0, side="hi")
        # 3) 源
        if source_index is not None:
            if soft:
                self.Ez[source_index] += source
            else:
                self.Ez[source_index] = source
        if pec_mask is not None:
            self.Ez[pec_mask] = 0.0
        self.n_steps += 1

    def run_record(self, n_steps: int, source_fn=None, source_index: int | None = None,
                   record_indices: tuple[int, ...] = ()) -> dict[str, np.ndarray]:
        """逐步运行，返回若干探针的时域序列（键为 'p0','p1',...）。"""
        rec = [[] for _ in record_indices]
        for n in range(n_steps):
            self.step_h()
            s = float(source_fn(n * self.dt)) if source_fn is not None else 0.0
            self.step_e(source=s, source_index=source_index)
            for k, idx in enumerate(record_indices):
                rec[k].append(float(self.Ez[idx]))
        return {f"p{k}": np.asarray(rec[k]) for k in range(len(record_indices))}


def cw_ramp(t: float, omega: float, t_ramp: float) -> float:
    """线性斜坡开启的连续正弦：sin(ωt)·min(1, t/t_ramp)。"""
    return float(np.sin(omega * t) * min(1.0, max(0.0, t / t_ramp)))
