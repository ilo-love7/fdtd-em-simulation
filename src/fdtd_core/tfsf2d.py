"""二维 TMz TFSF 连接边界（斜入射平面波，v1.3 模块 A）。

矩形 TFSF 盒 [i0,i1]×[j0,j1]（Ez 节点下标）：盒内 = 总场（入射+散射），
盒外 = 纯散射场。入射平面波解析给出，方向与 x 轴夹角 θ：

    E_inc(x,y,t) = A sin(ωt − k(x cosθ + y sinθ))
    Hy_inc = −A cosθ sin(ωt − k(x cosθ + y sinθ))
    Hx_inc = +A sinθ sin(ωt − k(x cosθ + y sinθ))

（归一化 c=ε=μ=1；k 可用物理 ω 或网格离散 k_num 以压低泄漏。）

Yee TMz 更新（未除 dx 的节点差，dt/dx）：
    Hy[i,j] += (dt/dx)(Ez[i+1,j] − Ez[i,j])
    Hx[i,j] −= (dt/dx)(Ez[i,j+1] − Ez[i,j])
    Ez[i,j] += (dt/dx)(Hy[i,j] − Hy[i−1,j] − Hx[i,j] + Hx[i,j−1])

连接边界修正原则：散射区的 H 节点其旋度跨到总场 E 时扣入射 E；
总场的 E 节点其旋度跨到散射区 H 时扣入射 H。四角自动叠加、无需额外项。

验证（scripts/13）：无散射体泄漏 ≈ 0；介质圆柱散射场 vs 2D Mie 级数；
关掉任一侧修正 → 泄漏暴涨（负结果）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .boundaries import mur1_update

C0 = 1.0
EPS0 = 1.0
MU0 = 1.0


@dataclass
class FDTD2D_TFSF:
    """二维 TMz + 矩形 TFSF 盒 + 斜入射平面波。

    Parameters
    ----------
    nx, ny, dx, courant : 网格、步长、Courant（≤1/√2）
    box : (i0, i1, j0, j1) 总场 Ez 区域（含端点）
    theta : 入射角（rad，相对 +x；0=正入射，π/4=45°）
    omega, amp : 入射角频率与幅度（归一化）
    k_num : 入射波数（默认 ω 物理值；可传离散 k 降泄漏）
    eps_r : (nx, ny) 相对介电常数（散射体放在总场区）
    correct_h / correct_e : 消融开关（关掉可复现泄漏）
    """

    nx: int
    ny: int
    box: tuple[int, int, int, int]
    dx: float = 1.0
    courant: float = 0.5
    theta: float = 0.0
    omega: float = 0.3
    amp: float = 1.0
    k_num: float | None = None
    eps_r: np.ndarray | float = 1.0
    correct_h: bool = True
    correct_e: bool = True

    Ez: np.ndarray = field(init=False, repr=False)
    Hx: np.ndarray = field(init=False, repr=False)
    Hy: np.ndarray = field(init=False, repr=False)
    dt: float = field(init=False)
    k: float = field(init=False)
    n_steps: int = field(default=0, init=False)
    _Ez_old: np.ndarray = field(init=False, repr=False)
    _eps: np.ndarray = field(init=False, repr=False)
    _mur_coef: float = field(init=False)

    def __post_init__(self) -> None:
        smax = 1.0 / np.sqrt(2.0)
        if not (0 < self.courant <= smax + 1e-12):
            raise ValueError(f"courant 须 ≤ 1/√2 = {smax:.4f}，收到 {self.courant}")
        i0, i1, j0, j1 = self.box
        if not (2 <= i0 < i1 <= self.nx - 3 and 2 <= j0 < j1 <= self.ny - 3):
            raise ValueError("TFSF 盒须留 ≥2 格边界余量")
        self.dt = self.courant * self.dx / C0
        self.k = self.omega if self.k_num is None else self.k_num
        self._eps = np.broadcast_to(np.asarray(self.eps_r, float), (self.nx, self.ny)).copy()
        self.Ez = np.zeros((self.nx, self.ny))
        self.Hx = np.zeros((self.nx, self.ny - 1))
        self.Hy = np.zeros((self.nx - 1, self.ny))
        self._Ez_old = np.zeros_like(self.Ez)
        self._mur_coef = (C0 * self.dt - self.dx) / (C0 * self.dt + self.dx)

    # ---------- 解析入射场 ----------
    def e_inc(self, x_idx: float, y_idx: float, t: float) -> float:
        """Ez_inc，x_idx/y_idx 为格点坐标（可半节点）。"""
        ph = self.omega * t - self.k * (x_idx * np.cos(self.theta)
                                        + y_idx * np.sin(self.theta)) * self.dx
        return self.amp * np.sin(ph)

    def hy_inc(self, x_idx: float, y_idx: float, t: float) -> float:
        return -self.amp * np.cos(self.theta) * np.sin(
            self.omega * t - self.k * (x_idx * np.cos(self.theta)
                                       + y_idx * np.sin(self.theta)) * self.dx)

    def hx_inc(self, x_idx: float, y_idx: float, t: float) -> float:
        return self.amp * np.sin(self.theta) * np.sin(
            self.omega * t - self.k * (x_idx * np.cos(self.theta)
                                       + y_idx * np.sin(self.theta)) * self.dx)

    # ---------- 步进 ----------
    def step(self) -> None:
        n = self.n_steps
        t_e = n * self.dt
        t_h = (n + 0.5) * self.dt
        i0, i1, j0, j1 = self.box
        r = self.dt / (MU0 * self.dx)

        # ---- H 半步 ----
        self.Hy += r * (self.Ez[1:, :] - self.Ez[:-1, :])
        self.Hx -= r * (self.Ez[:, 1:] - self.Ez[:, :-1])
        if self.correct_h:
            # 左边界：散射场 Hy[i0-1,j] 的旋度跨到总场 Ez[i0,j] → 扣 E_inc(i0,j)
            self.Hy[i0 - 1, j0:j1 + 1] -= r * self.e_inc(i0, np.arange(j0, j1 + 1), t_e)
            # 右边界：散射场 Hy[i1,j] 的旋度跨到总场 Ez[i1,j]（Ez 在差分中带负号）→ 加 E_inc
            self.Hy[i1, j0:j1 + 1] += r * self.e_inc(i1, np.arange(j0, j1 + 1), t_e)
            # 下边界：散射场 Hx[i,j0-1] 的旋度跨到总场 Ez[i,j0]
            #   Hx -= r(Ez[j0]-Ez[j0-1])，要扣总场侧入射 → +r E_inc(j0)
            self.Hx[i0:i1 + 1, j0 - 1] += r * self.e_inc(np.arange(i0, i1 + 1), j0, t_e)
            # 上边界：散射场 Hx[i,j1] 跨到总场 Ez[i,j1] → −r E_inc(j1)
            self.Hx[i0:i1 + 1, j1] -= r * self.e_inc(np.arange(i0, i1 + 1), j1, t_e)

        # ---- E 整步 ----
        self._Ez_old[:] = self.Ez
        new = self.Ez.copy()
        re = self.dt / (EPS0 * self._eps * self.dx)
        # Ez[i,j] 的 curl = Hy[i,j]−Hy[i−1,j]−Hx[i,j]+Hx[i,j−1]。
        # Hy 形状 (nx−1,ny)：Hy[i,j] 落在 Ez[i,j] 与 Ez[i+1,j] 之间；
        # Hx 形状 (nx,ny−1)：Hx[i,j] 落在 Ez[i,j] 与 Ez[i,j+1] 之间。
        hyp = self.Hy
        hxp = self.Hx
        curl = np.zeros_like(self.Ez)
        # Hy[i,j] (右) i=0..nx-2 → Ez[i,j]; Hy[i-1,j] (左) → Ez[i+1,j]
        curl[:-1, :] += hyp
        curl[1:, :] -= hyp
        # Hx[i,j] (下, j+½) j=0..ny-2 → Ez[i,j]; Hx[i,j-1] (上) → Ez[i,j+1]
        curl[:, :-1] -= hxp
        curl[:, 1:] += hxp
        new = self.Ez + re * curl
        self.Ez = new
        mur1_update(self.Ez, self._Ez_old, self._mur_coef, axis=0, side="lo")
        mur1_update(self.Ez, self._Ez_old, self._mur_coef, axis=0, side="hi")
        mur1_update(self.Ez, self._Ez_old, self._mur_coef, axis=1, side="lo")
        mur1_update(self.Ez, self._Ez_old, self._mur_coef, axis=1, side="hi")
        if self.correct_e:
            # 总场 Ez 的 curl 跨到散射区 H → 扣缺失的入射 H
            # 左：Ez[i0,j] 用散射 Hy[i0-1,j]（位于 i0-½）→ 扣 Hy_inc(i0-½,j)
            self.Ez[i0, j0:j1 + 1] -= re[i0, j0:j1 + 1] * self.hy_inc(
                i0 - 0.5, np.arange(j0, j1 + 1), t_h)
            # 右：Ez[i1,j] 用散射 Hy[i1,j]（位于 i1+½），curl 中带负号 → 加 Hy_inc
            self.Ez[i1, j0:j1 + 1] += re[i1, j0:j1 + 1] * self.hy_inc(
                i1 + 0.5, np.arange(j0, j1 + 1), t_h)
            # 下：Ez[i,j0] 用散射 Hx[i,j0-1]（位于 j0-½），curl=...+Hx[i,j-1] → 加 Hx_inc
            self.Ez[i0:i1 + 1, j0] += re[i0:i1 + 1, j0] * self.hx_inc(
                np.arange(i0, i1 + 1), j0 - 0.5, t_h)
            # 上：Ez[i,j1] 用散射 Hx[i,j1]（位于 j1+½），curl=...−Hx[i,j] → 扣 Hx_inc
            self.Ez[i0:i1 + 1, j1] -= re[i0:i1 + 1, j1] * self.hx_inc(
                np.arange(i0, i1 + 1), j1 + 0.5, t_h)
        self.n_steps += 1

    def run(self, n_steps: int, record: tuple[tuple[int, int], ...] = ()) -> dict[str, np.ndarray]:
        rec = [[] for _ in record]
        for _ in range(n_steps):
            self.step()
            for k, (i, j) in enumerate(record):
                rec[k].append(float(self.Ez[i, j]))
        return {f"p{k}": np.asarray(rec[k]) for k in range(len(record))}
