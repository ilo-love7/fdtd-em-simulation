"""一维 FDTD（Ez–Hy，Yee 交错网格）。

物理模型：均匀/分层介质中的平面波，∂Ez/∂t = (1/ε)∂Hy/∂x，∂Hy/∂t = (1/μ)∂Ez/∂x。
Yee 网格：Ez[i] 在节点 x_i = i·dx；Hy[i] 在半节点 (i+0.5)·dx。
时间步进为标准蛙跳（leapfrog）：H 先走半步，E 再走一步。
CFL 稳定性：c·dt/dx ≤ 1（1D）。

支持：
- 位置相关 εr（介质界面/多层）
- PEC 墙（Ez 强制 0）
- 一阶 Mur 吸收边界 / PML 电导率渐变吸收层（pml_cells>0 时自动启用并关闭 Mur）
- 软电流源 / 硬源 / 高斯脉冲或单频连续波
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .boundaries import mur1_update
from .pml import lossy_coeffs, sigma_profile_1d

C0 = 299_792_458.0
EPS0 = 8.8541878128e-12
MU0 = 4e-7 * np.pi


@dataclass
class FDTD1D:
    """一维 TM 平面波 FDTD。

    参数
    ----
    nx: 网格点数
    dx: 空间步长 [m]
    courant: Courant 数 S = c·dt/dx（默认 0.95，须 ≤ 1）
    eps_r: 相对介电常数数组（长度 nx，默认全 1）
    """

    nx: int
    dx: float
    courant: float = 0.95
    eps_r: np.ndarray | None = None
    pml_cells: int = 0          # >0 时两侧铺设 PML 吸收层（取代 Mur）
    pml_m: int = 4              # σ 渐变多项式阶

    # 运行时状态
    Ez: np.ndarray = field(init=False, repr=False)
    Hy: np.ndarray = field(init=False, repr=False)
    _Ez_old_edge: np.ndarray = field(init=False, repr=False)
    dt: float = field(init=False)
    n_steps: int = field(default=0, init=False)
    _sigma: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0 < self.courant <= 1.0):
            raise ValueError(f"1D CFL 要求 0 < courant ≤ 1，收到 {self.courant}")
        if self.eps_r is None:
            self.eps_r = np.ones(self.nx)
        self.eps_r = np.asarray(self.eps_r, dtype=float)
        if self.eps_r.shape != (self.nx,):
            raise ValueError("eps_r 形状须为 (nx,)")
        self.dt = self.courant * self.dx / C0
        self.Ez = np.zeros(self.nx)
        self.Hy = np.zeros(self.nx - 1)  # 半节点
        self._Ez_old_edge = np.zeros(self.nx)
        self._mur_coef = (C0 * self.dt - self.dx) / (C0 * self.dt + self.dx)
        self._sigma = sigma_profile_1d(self.nx, self.pml_cells, self.dx, self.pml_m)

    # ---------- 材料辅助 ----------
    @staticmethod
    def layered_eps_r(nx: int, dx: float, layers: list[tuple[float, float, float]]) -> np.ndarray:
        """按 (x_start, x_end, eps_r) 列表铺设介质层（未覆盖区域为 1）。"""
        eps = np.ones(nx)
        for x0, x1, e in layers:
            i0 = max(0, int(round(x0 / dx)))
            i1 = min(nx, int(round(x1 / dx)))
            eps[i0:i1] = e
        return eps

    # ---------- 步进 ----------
    def step_h(self) -> None:
        """Hy 半步更新：Hy[i] 位于 i+1/2，∂Ez/∂x 用 (Ez[i+1]-Ez[i])/dx。

        PML 路径：H 同步按匹配条件衰减（σ_m/μ0=σ_e/ε0，同因子 x=σΔt/ε0），
        否则 PML 内 H 无阻尼会与静态场耦合成积分器泵浦（v1.1 教训）。
        """
        diff = self.Ez[1:] - self.Ez[:-1]
        drive = (self.dt / (MU0 * self.dx)) * diff
        if self.pml_cells > 0:
            sig_h = 0.5 * (self._sigma[:-1] + self._sigma[1:])   # 半节点
            x = sig_h * self.dt / EPS0
            decay = np.exp(-x)
            small = x < 1e-8
            phi = np.where(small, 1.0 - x / 2.0 + x**2 / 6.0,
                           (1.0 - decay) / np.where(small, 1.0, x))
            self.Hy = decay * self.Hy + phi * drive
        else:
            self.Hy += drive

    def step_e(self, source: float = 0.0, source_index: int | None = None,
               soft: bool = True, pec_mask: np.ndarray | None = None) -> None:
        """Ez 整步更新：∂Hy/∂x 用 (Hy[i]-Hy[i-1])/dx。

        source: 注入值（A/m² 语义的软源为加性项，硬源为直接赋值）
        source_index: 注入点；None 则不注入
        pec_mask: 布尔数组，True 处 Ez 强制 0（PEC）
        """
        if self.pml_cells > 0:
            # PML 路径：精确积分有耗更新（系数恒正）+ 外缘零梯度闭合
            self._Ez_old_edge[:] = self.Ez
            curl = np.zeros(self.nx)
            curl[1:-1] = self.Hy[1:] - self.Hy[:-1]   # 边缘缺邻居 → curl=0（镜像）
            eps = EPS0 * self.eps_r
            decay, drive = lossy_coeffs(self._sigma, self.dt, eps)
            self.Ez = decay * self.Ez + drive * (self.dt / (eps * self.dx)) * curl
        else:
            self._Ez_old_edge[:] = self.Ez  # 保存旧值供 Mur 使用
            new = self.Ez.copy()
            # 内部节点 1..nx-2：Hy 有效下标 0..nx-3 → Hy[i]-Hy[i-1] 需 i≥1 且 i-1 ≤ nx-3
            new[1:-1] += (self.dt / (EPS0 * self.eps_r[1:-1] * self.dx)) * (self.Hy[1:] - self.Hy[:-1])
            self.Ez = new
            # Mur 一阶吸收边界
            mur1_update(self.Ez, self._Ez_old_edge, self._mur_coef, axis=0, side="lo")
            mur1_update(self.Ez, self._Ez_old_edge, self._mur_coef, axis=0, side="hi")
        # 源
        if source_index is not None:
            if soft:
                self.Ez[source_index] += source
            else:
                self.Ez[source_index] = source
        # PEC
        if pec_mask is not None:
            self.Ez[pec_mask] = 0.0
        self.n_steps += 1

    def run(self, n_steps: int, source_fn=None, source_index: int | None = None,
            soft: bool = True, pec_mask: np.ndarray | None = None,
            record_every: int = 1, record_index: int | None = None) -> dict:
        """运行 n_steps 步。source_fn(n) -> float。返回记录的时序。"""
        rec_t, rec_e = [], []
        for n in range(n_steps):
            self.step_h()
            s = float(source_fn(n)) if source_fn is not None else 0.0
            self.step_e(source=s, source_index=source_index, soft=soft, pec_mask=pec_mask)
            if record_index is not None and n % record_every == 0:
                rec_t.append((n + 1) * self.dt)
                rec_e.append(float(self.Ez[record_index]))
        return {"t": np.array(rec_t), "Ez": np.array(rec_e)}


def gaussian_pulse(t: float, t0: float, width: float) -> float:
    """高斯脉冲 exp(-(t-t0)²/(2 width²))。"""
    return float(np.exp(-((t - t0) ** 2) / (2.0 * width**2)))


def gaussian_derivative(t: float, t0: float, width: float) -> float:
    """高斯一阶导（无直流，适合宽带激励）。"""
    u = (t - t0) / width
    return float(-u * np.exp(-(u**2) / 2.0) / width)
