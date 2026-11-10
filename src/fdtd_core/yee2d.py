"""二维 TMz FDTD（Ez, Hx, Hy，Yee 交错网格）。

物理模型（真空中，TMz 极化）：
    μ ∂Hx/∂t = −∂Ez/∂y
    μ ∂Hy/∂t = +∂Ez/∂x
    ε ∂Ez/∂t = ∂Hy/∂x − ∂Hx/∂y

Yee 网格（单位胞元）：
    Ez[i,j]   位于 (i,   j  )
    Hy[i,j]   位于 (i+½, j  )
    Hx[i,j]   位于 (i,   j+½)
CFL 稳定性：c·dt/dx ≤ 1/√2（正方形网格）。

数值约定（关键）：差分一律用**未除 dx 的节点差**（ΔEz、ΔHx、ΔHy），
系数为 dt/(ε·dx) 与 dt/(μ·dx)，与 1D 求解器一致。
若先把差分除以 dx 再乘 dt/(ε·dx)，会双重除 dx 导致发散（v1.0 调试教训）。

支持：PEC 金属（Ez 强制 0）、位置相关 εr、一阶 Mur 吸收边界或 PML 吸收层
（pml_cells_x/y>0 时启用 PML 并关闭 Mur）、点/行软硬源。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .boundaries import mur1_update
from .pml import lossy_coeffs, sigma_profile_2d

C0 = 299_792_458.0
EPS0 = 8.8541878128e-12
MU0 = 4e-7 * np.pi


@dataclass
class FDTD2D:
    """二维 TMz FDTD（正方形网格）。

    参数
    ----
    nx, ny: 网格点数
    dx: 空间步长 [m]（正方形胞元，dy = dx）
    courant: Courant 数 S = c·dt/dx，须 ≤ 1/√2 ≈ 0.7071
    eps_r: 形状 (nx, ny) 的相对介电常数，默认真空
    """

    nx: int
    ny: int
    dx: float
    courant: float = 0.65
    eps_r: np.ndarray | None = None
    pml_cells_x: int = 0        # x 两侧 PML 厚度（>0 启用 PML 取代 Mur）
    pml_cells_y: int = 0        # y 两侧 PML 厚度
    pml_m: int = 4

    Ez: np.ndarray = field(init=False, repr=False)
    Hx: np.ndarray = field(init=False, repr=False)
    Hy: np.ndarray = field(init=False, repr=False)
    dt: float = field(init=False)
    n_steps: int = field(default=0, init=False)
    _Ez_old: np.ndarray = field(init=False, repr=False)
    _sigma: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        smax = 1.0 / np.sqrt(2.0)
        if not (0 < self.courant <= smax + 1e-12):
            raise ValueError(f"2D CFL 要求 0 < courant ≤ 1/√2 = {smax:.4f}，收到 {self.courant}")
        if self.eps_r is None:
            self.eps_r = np.ones((self.nx, self.ny))
        self.eps_r = np.asarray(self.eps_r, dtype=float)
        if self.eps_r.shape != (self.nx, self.ny):
            raise ValueError("eps_r 形状须为 (nx, ny)")
        self.dt = self.courant * self.dx / C0
        self.Ez = np.zeros((self.nx, self.ny))
        self.Hx = np.zeros((self.nx, self.ny - 1))
        self.Hy = np.zeros((self.nx - 1, self.ny))
        self._Ez_old = np.zeros((self.nx, self.ny))
        self._mur_coef = (C0 * self.dt - self.dx) / (C0 * self.dt + self.dx)
        self._sigma = sigma_profile_2d(self.nx, self.ny, self.pml_cells_x,
                                       self.pml_cells_y, self.dx, self.pml_m)

    def use_pml(self) -> bool:
        return self.pml_cells_x > 0 or self.pml_cells_y > 0

    # ---------- 材料 ----------
    @staticmethod
    def box_pec(nx: int, ny: int, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
        """矩形 PEC 掩码（True = 金属）。[x0:x1, y0:y1) 半开区间。"""
        m = np.zeros((nx, ny), dtype=bool)
        m[x0:x1, y0:y1] = True
        return m

    @staticmethod
    def slit_screen(nx: int, ny: int, x_slit: int, thickness: int,
                    y_slit0: int, y_slit1: int) -> np.ndarray:
        """带缝金属屏：x ∈ [x_slit, x_slit+thickness) 全金属，唯缝区间 [y_slit0, y_slit1) 留空。"""
        m = np.zeros((nx, ny), dtype=bool)
        m[x_slit:x_slit + thickness, :] = True
        m[x_slit:x_slit + thickness, y_slit0:y_slit1] = False
        return m

    # ---------- 步进 ----------
    def step_h(self) -> None:
        """H 半步：Hx 对 ∂Ez/∂y，Hy 对 ∂Ez/∂x（节点差未除 dx）。

        PML 路径：H 按匹配条件同步衰减（σ_m/μ0=σ_e/ε0），否则 PML 内
        H 无阻尼会与静态场耦合成积分器泵浦（v1.1 教训）。
        """
        # Hx[i,j] 位于 (i, j+½)：ΔEz = Ez[i,j+1] − Ez[i,j]
        drive_x = -(self.dt / (MU0 * self.dx)) * (self.Ez[:, 1:] - self.Ez[:, :-1])
        # Hy[i,j] 位于 (i+½, j)：ΔEz = Ez[i+1,j] − Ez[i,j]
        drive_y = (self.dt / (MU0 * self.dx)) * (self.Ez[1:, :] - self.Ez[:-1, :])
        if self.use_pml():
            sig_hx = 0.5 * (self._sigma[:, :-1] + self._sigma[:, 1:])
            sig_hy = 0.5 * (self._sigma[:-1, :] + self._sigma[1:, :])
            for arr, sig, drive in ((self.Hx, sig_hx, drive_x), (self.Hy, sig_hy, drive_y)):
                x = sig * self.dt / EPS0
                decay = np.exp(-x)
                small = x < 1e-8
                phi = np.where(small, 1.0 - x / 2.0 + x**2 / 6.0,
                               (1.0 - decay) / np.where(small, 1.0, x))
                arr[:] = decay * arr + phi * drive
        else:
            self.Hx += drive_x
            self.Hy += drive_y

    def _update_ez(self, y_neumann: bool = False) -> None:
        """Ez 更新：PML 路径（有耗 CN）或 Mur/Neumann 路径。

        y_neumann=True 时 y 两侧用对称条件（E[:,0]=E[:,1], E[:,-1]=E[:,-2]），
        严格保持 y 均匀场（平面波验证用）；默认 y 也用 Mur（衍射问题用）。
        """
        self._Ez_old[:] = self.Ez
        if self.use_pml():
            # PML 路径：精确积分有耗更新（系数恒正），H 差分用边缘镜像闭合
            hyp = np.pad(self.Hy, ((1, 1), (0, 0)), mode="edge")   # (nx+1, ny)
            hxp = np.pad(self.Hx, ((0, 0), (1, 1)), mode="edge")   # (nx, ny+1)
            dhy = hyp[1:, :] - hyp[:-1, :]    # ΔHy（未除 dx）
            dhx = hxp[:, 1:] - hxp[:, :-1]    # ΔHx（未除 dx）
            eps = EPS0 * self.eps_r
            decay, drive = lossy_coeffs(self._sigma, self.dt, eps)
            self.Ez = (decay * self.Ez
                       + drive * (self.dt / (eps * self.dx)) * (dhy - dhx))
        else:
            new = self.Ez.copy()
            dhy = self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1]   # ΔHy（未除 dx）
            dhx = self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1]   # ΔHx（未除 dx）
            new[1:-1, 1:-1] += (self.dt / (EPS0 * self.eps_r[1:-1, 1:-1] * self.dx)) * (dhy - dhx)
            self.Ez = new
            mur1_update(self.Ez, self._Ez_old, self._mur_coef, axis=0, side="lo")
            mur1_update(self.Ez, self._Ez_old, self._mur_coef, axis=0, side="hi")
        if y_neumann:
            self.Ez[:, 0] = self.Ez[:, 1]
            self.Ez[:, -1] = self.Ez[:, -2]
        elif not self.use_pml():
            mur1_update(self.Ez, self._Ez_old, self._mur_coef, axis=1, side="lo")
            mur1_update(self.Ez, self._Ez_old, self._mur_coef, axis=1, side="hi")

    def step_e(self, source: float = 0.0, source_index: tuple[int, int] | None = None,
               soft: bool = True, pec_mask: np.ndarray | None = None,
               y_neumann: bool = False) -> None:
        """Ez 整步（点源）。"""
        self._update_ez(y_neumann=y_neumann)
        if source_index is not None:
            i, j = source_index
            if soft:
                self.Ez[i, j] += source
            else:
                self.Ez[i, j] = source
        if pec_mask is not None:
            self.Ez[pec_mask] = 0.0
        self.n_steps += 1

    def step_e_line(self, source: float, line_source: int, soft: bool = True,
                    pec_mask: np.ndarray | None = None, y_neumann: bool = False) -> None:
        """整列（x = line_source）软源注入的 Ez 整步（模拟入射平面波）。

        只注入内部列 j=1..ny-2，避免覆盖 y 边界节点（否则激发 y 向波污染平面波）。
        """
        self._update_ez(y_neumann=y_neumann)
        if soft:
            self.Ez[line_source, 1:-1] += source
        else:
            self.Ez[line_source, 1:-1] = source
        if pec_mask is not None:
            self.Ez[pec_mask] = 0.0
        self.n_steps += 1

    def run(self, n_steps: int, source_fn=None,
            source_index: tuple[int, int] | None = None,
            line_source: int | None = None, soft: bool = True,
            pec_mask: np.ndarray | None = None,
            snapshot_every: int = 0) -> dict:
        """运行 n_steps 步。source_fn(n) -> float。返回快照列表。"""
        snaps, times = [], []
        for n in range(n_steps):
            self.step_h()
            s = float(source_fn(n)) if source_fn is not None else 0.0
            if line_source is not None:
                self.step_e_line(source=s, line_source=line_source, soft=soft, pec_mask=pec_mask)
            else:
                self.step_e(source=s, source_index=source_index, soft=soft, pec_mask=pec_mask)
            if snapshot_every and n % snapshot_every == 0:
                snaps.append(self.Ez.copy())
                times.append((n + 1) * self.dt)
        return {"snapshots": snaps, "snapshot_times": np.array(times)}
