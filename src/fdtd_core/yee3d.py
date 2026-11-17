"""三维全矢量 Yee FDTD + Drude ADE + 匹配有耗 PML（v1.3 模块 C）。

归一化单位 dx=dy=dz=1、c0=ε0=μ0=1、η0=1；Courant S=c·dt/dx ≤ 1/√3。

Yee 交错：
    Ex[i,j,k] @ (i+½, j,   k  )   Ey[i,j,k] @ (i,   j+½, k  )
    Ez[i,j,k] @ (i,   j,   k+½)   Hx[i,j,k] @ (i,   j+½, k+½)
    Hy[i,j,k] @ (i+½, j,   k+½)   Hz[i,j,k] @ (i+½, j+½, k  )

Faraday: μ ∂H/∂t = −∇×E；Ampere（含 Drude 极化电流 J）:
    ε_inf ∂E/∂t + J = ∇×H,   ∂J/∂t + γ J = ε0 ω_p² E
蛙跳 + J 梯形（系数恒正稳定）。六分量全开，支持任意方向传播。

PML：κ=1 匹配有耗层，σ 按深度幂律渐变（x/y/z 取最大覆盖角落）；
E/H 用同一指数衰减 x=σ dt/ε0（匹配条件 σ_m/μ0=σ_e/ε0），
E 的驱动用精确积分系数 φ=(1−e^{−x})/x（恒正，避免 CN 负系数泵浦）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C0 = 1.0
EPS0 = 1.0
MU0 = 1.0


def sigma_profile_1d(n: int, cells: int, dx: float = 1.0, m: int = 4,
                     sigma_max: float = 0.8) -> np.ndarray:
    if cells <= 0:
        return np.zeros(n)
    sig = np.zeros(n)
    smax = sigma_max * (m + 1) / dx
    if cells == 1:
        sig[0] = sig[-1] = smax
        return sig
    prof = (np.arange(cells) / (cells - 1)) ** m
    sig[:cells] = smax * prof[::-1]
    sig[n - cells:] = smax * prof
    return sig


def sigma_profile_3d(nx: int, ny: int, nz: int, cells: int,
                     dx: float = 1.0, m: int = 4) -> np.ndarray:
    """三维 σ 图（各轴剖面取最大，角落覆盖）。"""
    sig = np.zeros((nx, ny, nz))
    if cells <= 0:
        return sig
    sx = sigma_profile_1d(nx, cells, dx, m)[:, None, None]
    sy = sigma_profile_1d(ny, cells, dx, m)[None, :, None]
    sz = sigma_profile_1d(nz, cells, dx, m)[None, None, :]
    sig[:] = np.maximum(np.maximum(sx, sy), sz)
    return sig


@dataclass
class FDTD3D_Drude:
    """三维全矢量 Yee + Drude ADE + PML。

    nx,ny,nz    : 网格点数
    courant     : S ≤ 1/√3
    pml_cells   : 各向 PML 厚度（>0 启用）
    eps_inf     : 高频介电常数（标量或 (nx,ny,nz) 数组）
    wp2, gamma  : 等离子体频率平方 / 碰撞频率（归一化，标量或数组）
    """
    nx: int
    ny: int
    nz: int
    courant: float = 0.4
    dx: float = 1.0
    pml_cells: int = 0
    pml_m: int = 4
    eps_inf: float | np.ndarray = 1.0
    wp2: float | np.ndarray = 0.0
    gamma: float | np.ndarray = 0.0

    Ex: np.ndarray = field(init=False, repr=False)
    Ey: np.ndarray = field(init=False, repr=False)
    Ez: np.ndarray = field(init=False, repr=False)
    Hx: np.ndarray = field(init=False, repr=False)
    Hy: np.ndarray = field(init=False, repr=False)
    Hz: np.ndarray = field(init=False, repr=False)
    Jx: np.ndarray = field(init=False, repr=False)
    Jy: np.ndarray = field(init=False, repr=False)
    Jz: np.ndarray = field(init=False, repr=False)
    dt: float = field(init=False)
    n_steps: int = field(default=0, init=False)
    _eps_inf: np.ndarray = field(init=False, repr=False)
    _wp2: np.ndarray = field(init=False, repr=False)
    _gamma: np.ndarray = field(init=False, repr=False)
    _sig: np.ndarray = field(init=False, repr=False)
    _decay_e: np.ndarray = field(init=False, repr=False)
    _drive_e: np.ndarray = field(init=False, repr=False)
    _decay_h: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        smax = 1.0 / np.sqrt(3.0)
        if not (0 < self.courant <= smax + 1e-12):
            raise ValueError(f"3D CFL 要求 S ≤ 1/√3 = {smax:.4f}，收到 {self.courant}")
        self.dt = self.courant * self.dx / C0
        shape = (self.nx, self.ny, self.nz)
        self._eps_inf = np.broadcast_to(np.asarray(self.eps_inf, float), shape).copy()
        self._wp2 = np.broadcast_to(np.asarray(self.wp2, float), shape).copy()
        self._gamma = np.broadcast_to(np.asarray(self.gamma, float), shape).copy()
        if np.any(self._eps_inf <= 0):
            raise ValueError("eps_inf 须为正")
        zeros = lambda: np.zeros(shape)
        self.Ex, self.Ey, self.Ez = zeros(), zeros(), zeros()
        self.Hx, self.Hy, self.Hz = zeros(), zeros(), zeros()
        self.Jx, self.Jy, self.Jz = zeros(), zeros(), zeros()
        self._sig = sigma_profile_3d(self.nx, self.ny, self.nz,
                                     self.pml_cells, self.dx, self.pml_m)
        x = self._sig * self.dt / EPS0
        decay = np.exp(-x)
        small = x < 1e-8
        phi = np.where(small, 1.0 - x / 2.0 + x**2 / 6.0,
                       (1.0 - decay) / np.where(small, 1.0, x))
        self._decay_e = decay
        self._drive_e = phi
        self._decay_h = decay  # 匹配条件：σ_m/μ0 = σ_e/ε0（归一化相等）

    @staticmethod
    def drude_slab(nx: int, ny: int, nz: int, xs: int, xe: int,
                   wp2: float, gamma: float = 0.0, eps_inf: float = 1.0):
        ei = np.ones((nx, ny, nz)); w2 = np.zeros_like(ei); ga = np.zeros_like(ei)
        ei[xs:xe, :, :] = eps_inf
        w2[xs:xe, :, :] = wp2
        ga[xs:xe, :, :] = gamma
        return ei, w2, ga

    # ---------- 旋度（未除 dx 的节点差）----------
    def _curl_e(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Ex, Ey, Ez = self.Ex, self.Ey, self.Ez
        # (curl E)_x @ Hx(i,j+½,k+½) = (Ez[j+1]-Ez[j]) - (Ey[k+1]-Ey[k])
        cz = np.pad(Ez, ((0, 0), (0, 1), (0, 0)), mode="edge")
        cx = (cz[:, 1:, :] - cz[:, :-1, :])
        cy = np.pad(Ey, ((0, 0), (0, 0), (0, 1)), mode="edge")
        cx -= (cy[:, :, 1:] - cy[:, :, :-1])
        # (curl E)_y @ Hy(i+½,j,k+½) = (Ex[k+1]-Ex[k]) - (Ez[i+1]-Ez[i])
        cy_ = np.pad(Ex, ((0, 0), (0, 0), (0, 1)), mode="edge")
        cyd = cy_[:, :, 1:] - cy_[:, :, :-1]
        ezp = np.pad(Ez, ((0, 1), (0, 0), (0, 0)), mode="edge")
        cyd -= ezp[1:, :, :] - ezp[:-1, :, :]
        # (curl E)_z @ Hz(i+½,j+½,k) = (Ey[i+1]-Ey[i]) - (Ex[j+1]-Ex[j])
        cz_ = np.pad(Ey, ((0, 1), (0, 0), (0, 0)), mode="edge")
        czd = cz_[1:, :, :] - cz_[:-1, :, :]
        exp_ = np.pad(Ex, ((0, 0), (0, 1), (0, 0)), mode="edge")
        czd -= exp_[:, 1:, :] - exp_[:, :-1, :]
        return cx, cyd, czd

    def _curl_h(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Hx, Hy, Hz = self.Hx, self.Hy, self.Hz
        # (curl H)_x @ Ex(i+½,j,k) = (Hz[j]-Hz[j-1]) - (Hy[k]-Hy[k-1])
        hz = np.pad(Hz, ((0, 0), (1, 0), (0, 0)), mode="edge")
        cx = hz[:, 1:, :] - hz[:, :-1, :]
        hy = np.pad(Hy, ((0, 0), (0, 0), (1, 0)), mode="edge")
        cx -= hy[:, :, 1:] - hy[:, :, :-1]
        # (curl H)_y @ Ey(i,j+½,k) = (Hx[k]-Hx[k-1]) - (Hz[i]-Hz[i-1])
        hx = np.pad(Hx, ((0, 0), (0, 0), (1, 0)), mode="edge")
        cy = hx[:, :, 1:] - hx[:, :, :-1]
        hz2 = np.pad(Hz, ((1, 0), (0, 0), (0, 0)), mode="edge")
        cy -= hz2[1:, :, :] - hz2[:-1, :, :]
        # (curl H)_z @ Ez(i,j,k+½) = (Hy[i]-Hy[i-1]) - (Hx[j]-Hx[j-1])
        hy2 = np.pad(Hy, ((1, 0), (0, 0), (0, 0)), mode="edge")
        cz = hy2[1:, :, :] - hy2[:-1, :, :]
        hx2 = np.pad(Hx, ((0, 0), (1, 0), (0, 0)), mode="edge")
        cz -= hx2[:, 1:, :] - hx2[:, :-1, :]
        return cx, cy, cz

    # ---------- 步进 ----------
    def step(self, source: tuple[str, int, int, int, float] | None = None) -> None:
        """推进一步。source=(component, i, j, k, value) 软源（E 整步注入）。"""
        r = self.dt / (MU0 * self.dx)
        ce = self._curl_e()
        for comp, c in ((self.Hx, ce[0]), (self.Hy, ce[1]), (self.Hz, ce[2])):
            comp *= self._decay_h if self.pml_cells else 1.0
            comp -= r * c
        # ADE 极化电流（梯形，系数恒正）
        g = self._gamma * self.dt / 2.0
        for J, E in ((self.Jx, self.Ex), (self.Jy, self.Ey), (self.Jz, self.Ez)):
            J[:] = ((1.0 - g) * J + EPS0 * self._wp2 * self.dt * E) / (1.0 + g)
        ch = self._curl_h()
        re = self.dt / (EPS0 * self.dx)
        for comp, c, J in ((self.Ex, ch[0], self.Jx),
                           (self.Ey, ch[1], self.Jy),
                           (self.Ez, ch[2], self.Jz)):
            drive = re * c / self._eps_inf - self.dt * J / (EPS0 * self._eps_inf)
            if self.pml_cells:
                comp *= self._decay_e
                comp += self._drive_e * drive
            else:
                comp += drive
        if source is not None:
            comp, i, j, k, val = source
            getattr(self, comp)[i, j, k] += val

    def inject_plane(self, comp: str, i: int, val: float,
                     js: slice | None = None, ks: slice | None = None) -> None:
        """在 x=i 平面注入软源（整片 y/z，避开 PML 时为平面波）。"""
        if js is None:
            js = slice(self.pml_cells, self.ny - self.pml_cells if self.pml_cells else self.ny)
        if ks is None:
            ks = slice(self.pml_cells, self.nz - self.pml_cells if self.pml_cells else self.nz)
        getattr(self, comp)[i, js, ks] += val
        self.n_steps += 1

    def run(self, n_steps: int, source_fn=None,
            source_comp: str = "Ey", source_index: tuple[int, int, int] = (5, 5, 5),
            record: tuple[tuple[str, int, int, int], ...] = ()) -> dict[str, np.ndarray]:
        rec = [[] for _ in record]
        for n in range(n_steps):
            val = float(source_fn(n * self.dt)) if source_fn else 0.0
            src = (source_comp, *source_index, val) if source_fn else None
            self.step(source=src)
            for q, (comp, i, j, k) in enumerate(record):
                rec[q].append(float(getattr(self, comp)[i, j, k]))
        return {f"p{q}": np.asarray(rec[q]) for q in range(len(record))}
