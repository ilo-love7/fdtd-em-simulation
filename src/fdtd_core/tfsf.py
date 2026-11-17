"""一维 TFSF（Total-Field / Scattered-Field）连接边界，v1.2。

把计算域分为：
- 总场区（total field, i ≥ i_b）：E = E_inc + E_scat，含入射波与散射体；
- 散射场区（scattered field, i < i_b）：只保留 E_scat/H_scat。

入射波用解析平面波在连接边界按需求值（真空 +x 传播，归一化单位 c0=ε0=μ0=1）：
    E_inc(x,t) = A sin(ω(t − x/c0))
    H_inc(x,t) = −A sin(ω(t − x/c0))     （Ez 沿 +x 传播时 Hy = −Ez/η0，η0=1）

连接边界修正（Yee：Ez[i] 在整点、Hy[i] 在 i+½）：
- 更新最后一个散射场 H 节点（Hy[i_b−1]，位于 i_b−½）时，curl 用到总场 E[i_b]，
  须扣掉其入射分量：ΔHy += −(dt/(μ·dx))·E_inc(x_{i_b}, t=n·dt)
- 更新第一个总场 E 节点（Ez[i_b]）时，curl 用到散射场 Hy[i_b−1]，
  须补上其入射分量：ΔEz += (dt/(ε·dx))·H_inc(x_{i_b−½}, t=(n+½)·dt)

验证（scripts/10）：
- 无散射体：散射场区 |E_scat| 应达机器精度（连接边界无泄漏），总场区 = 入射；
- 介质半空间散射体：反射回散射场区的波幅 vs 解析 Fresnel |r|；
- 缺 H 修正会泄漏（负结果，证明连接边界双侧修正缺一不可）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .boundaries import mur1_update

C0 = 1.0
EPS0 = 1.0
MU0 = 1.0


@dataclass
class FDTD1D_TFSF:
    """一维 TM 平面波 FDTD + TFSF 连接边界。

    参数
    ----
    nx, dx, courant : 网格、步长、Courant 数（≤1）
    i_b             : 连接边界 E 节点下标（i < i_b 为散射场区）
    eps_r           : 长度 nx 相对介电常数（散射体可放在总场区 i ≥ i_b）
    omega, amp      : 入射 CW 角频率与幅度（归一化单位）
    """

    nx: int
    i_b: int
    dx: float = 1.0
    courant: float = 0.95
    eps_r: np.ndarray | float = 1.0
    omega: float = 0.3
    amp: float = 1.0
    correct_h: bool = True     # 关掉 H 修正可复现泄漏（负结果对照）
    correct_e: bool = True     # 关掉 E 修正的对照

    Ez: np.ndarray = field(init=False, repr=False)
    Hy: np.ndarray = field(init=False, repr=False)
    dt: float = field(init=False)
    n_steps: int = field(default=0, init=False)
    _Ez_old: np.ndarray = field(init=False, repr=False)
    _eps: np.ndarray = field(init=False, repr=False)
    _mur_coef: float = field(init=False)

    def __post_init__(self) -> None:
        if not (0 < self.courant <= 1.0):
            raise ValueError("courant 须 ≤ 1")
        if not (2 <= self.i_b <= self.nx - 3):
            raise ValueError("i_b 须在 [2, nx-3]")
        self.dt = self.courant * self.dx / C0
        self._eps = np.broadcast_to(np.asarray(self.eps_r, float), (self.nx,)).copy()
        self.Ez = np.zeros(self.nx)
        self.Hy = np.zeros(self.nx - 1)
        self._Ez_old = np.zeros(self.nx)
        self._mur_coef = (C0 * self.dt - self.dx) / (C0 * self.dt + self.dx)

    # ---------- 解析入射波 ----------
    def e_inc(self, i: int, t: float) -> float:
        return self.amp * np.sin(self.omega * (t - i * self.dx / C0))

    def h_inc(self, i_half: float, t: float) -> float:
        """Hy 在半节点 i_half（例如 i_b−0.5）：Hy_inc = −Ez_inc。"""
        return -self.amp * np.sin(self.omega * (t - i_half * self.dx / C0))

    # ---------- 步进 ----------
    def step(self) -> None:
        n = self.n_steps
        t_e = n * self.dt          # 当前 Ez 时刻
        t_h = (n + 0.5) * self.dt   # Hy 更新后时刻
        # --- H 半步 ---
        self.Hy += (self.dt / (MU0 * self.dx)) * (self.Ez[1:] - self.Ez[:-1])
        # TFSF：最后一个散射场 H 节点 Hy[i_b-1]（半节点 i_b−0.5）扣入射 curl。
        # 该 H 节点 curl 用到总场 E[i_b]，须减去其入射分量以保持纯散射场。
        if self.correct_h:
            self.Hy[self.i_b - 1] -= (self.dt / (MU0 * self.dx)) * self.e_inc(self.i_b, t_e)
        # --- E 整步 ---
        self._Ez_old[:] = self.Ez
        new = self.Ez.copy()
        coef = self.dt / (EPS0 * self._eps * self.dx)
        new[1:-1] += coef[1:-1] * (self.Hy[1:] - self.Hy[:-1])
        self.Ez = new
        mur1_update(self.Ez, self._Ez_old, self._mur_coef, axis=0, side="lo")
        mur1_update(self.Ez, self._Ez_old, self._mur_coef, axis=0, side="hi")
        # TFSF：第一个总场 E 节点 Ez[i_b] 的 curl 用到散射场 Hy[i_b-1]，
        # 须补上其缺失的入射分量（E 节点与左侧半节点 H 相减，符号为负）。
        if self.correct_e:
            self.Ez[self.i_b] -= (self.dt / (EPS0 * self._eps[self.i_b] * self.dx)) * \
                                 self.h_inc(self.i_b - 0.5, t_h)
        self.n_steps += 1

    def run(self, n_steps: int, record: tuple[int, ...] = ()) -> dict[str, np.ndarray]:
        rec = [[] for _ in record]
        for _ in range(n_steps):
            self.step()
            for k, idx in enumerate(record):
                rec[k].append(float(self.Ez[idx]))
        return {f"p{k}": np.asarray(rec[k]) for k in range(len(record))}
