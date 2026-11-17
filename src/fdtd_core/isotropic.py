"""二维波方程 FDTD + 各向同性 9 点 Laplacian（v1.3 模块 B）。

用二阶波动方程的 FDTD 格式替代一阶 Yee 旋度格式，以便把 5 点 Laplacian
替换为**各向同性 9 点（Mehrstellen）Laplacian**做色散补偿：

    E^{n+1} = 2E^n - E^{n-1} + (c dt)² L E^n

5 点 Laplacian Fourier 符号（s_x = sin(k_x dx/2)）：
    L5 = -(4/dx²)[s_x² + s_y²]
9 点各向同性 Laplacian（轴向权 4、对角权 1、中心 -20，除以 6dx²）：
    L9 = -(4/dx²)[s_x² + s_y² - (2/3) s_x² s_y²]

小 K=k dx 展开：
    L5: s_x²+s_y² = K²/4 - K⁴(cos⁴θ+sin⁴θ)/48  → O(K⁴) 各向异性
    L9: 上式 -(2/3)s_x²s_y²，θ 相关项精确抵消 → 余项 O(K⁶)

色散关系：sin²(ω dt/2) = S²[s_x²+s_y² - (2/3)s_x²s_y²]（S=c dt/dx）
稳定上限：S ≤ √3/2 ≈ 0.866（5 点为 1/√2 ≈ 0.707）。

本模块保留 5 点路径作对照（isotropic=False），两路径共用同一 E^{n+1}
更新代码以保证对照公平。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C0 = 299_792_458.0


@dataclass
class Wave2D:
    """二维标量波方程 FDTD（可切 5 点 / 9 点各向同性 Laplacian）。

    Parameters
    ----------
    nx, ny    : 网格点数
    dx        : 空间步长 [m]（正方网格）
    courant   : S = c dt/dx，5 点 ≤ 1/√2，9 点 ≤ √3/2
    isotropic : True → 各向同性 9 点 Laplacian；False → 标准 5 点
    omega     : 驱动角频率 [rad/s]（用于相量累积）
    """
    nx: int
    ny: int
    dx: float = 1.0
    courant: float = 0.6
    isotropic: bool = False
    omega: float = 0.0

    E: np.ndarray = field(init=False, repr=False)
    E_prev: np.ndarray = field(init=False, repr=False)
    E_next: np.ndarray = field(init=False, repr=False)
    dt: float = field(init=False)
    n_steps: int = field(default=0, init=False)
    _mur_coef: float = field(init=False)
    _amp: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        smax = np.sqrt(3.0) / 2.0 if self.isotropic else 1.0 / np.sqrt(2.0)
        if not (0 < self.courant <= smax + 1e-12):
            raise ValueError(f"{'9pt' if self.isotropic else '5pt'} 稳定上限 S≤{smax:.4f}，"
                             f"收到 {self.courant}")
        self.dt = self.courant * self.dx / C0
        self.E = np.zeros((self.nx, self.ny))
        self.E_prev = np.zeros_like(self.E)
        self.E_next = np.zeros_like(self.E)
        self._amp = np.zeros((self.nx, self.ny), dtype=complex)
        # 一阶 Mur：E^{n+1}_0 = E^n_1 + (S-1)/(S+1)(E^{n+1}_1 - E^n_0)
        self._mur_coef = (self.courant - 1.0) / (self.courant + 1.0)

    # ---------- Laplacian ----------
    def _laplacian(self, e: np.ndarray) -> np.ndarray:
        """返回 dx²·L E 的内部点（即 L 乘 dx²）。"""
        c = e[1:-1, 1:-1]
        ax = e[2:, 1:-1] + e[:-2, 1:-1] + e[1:-1, 2:] + e[1:-1, :-2]
        if self.isotropic:
            dg = e[2:, 2:] + e[2:, :-2] + e[:-2, 2:] + e[:-2, :-2]
            return (4.0 * ax + dg - 20.0 * c) / 6.0
        return ax - 4.0 * c

    # ---------- 步进 ----------
    def step(self, source: float = 0.0, source_i: int | None = None,
             ky: float = 0.0, accumulate: bool = False) -> None:
        """推进一步。source_i 为 x 方向注入列，ky 为该列的 y 相位梯度（rad/格）。

        accumulate=True 时把当前 E 乘 exp(-i ω t) 累加到相量缓冲，用于测 k。
        """
        coef = self.courant ** 2
        self.E_next[1:-1, 1:-1] = (2.0 * self.E[1:-1, 1:-1]
                                   - self.E_prev[1:-1, 1:-1]
                                   + coef * self._laplacian(self.E))
        # Mur ABC（四边）
        m = self._mur_coef
        en, e, ep = self.E_next, self.E, self.E_prev
        en[0, :] = e[1, :] + m * (en[1, :] - e[0, :])
        en[-1, :] = e[-2, :] + m * (en[-2, :] - e[-1, :])
        en[:, 0] = e[:, 1] + m * (en[:, 1] - e[:, 0])
        en[:, -1] = e[:, -2] + m * (en[:, -2] - e[:, -1])
        # 相位列软源（x=source_i，沿 y 施加 ky 相位，朝 +x 发射平面波）
        # source 可为标量（配 ky 用 cos 对称注入）或长度 ny 的数组（自带 y 相位）。
        if source_i is not None:
            if np.isscalar(source):
                y = np.arange(self.ny)
                inj = source * np.cos(ky * (y - self.ny // 2))
            else:
                inj = np.asarray(source, dtype=float)
            self.E_next[source_i, :] += inj
        # 轮换
        self.E_prev, self.E, self.E_next = self.E, self.E_next, self.E_prev
        if accumulate and self.omega > 0:
            self._amp += self.E * np.exp(-1j * self.omega * self.n_steps * self.dt)
        self.n_steps += 1

    def phasor_wavenumbers(self, xs: tuple[int, int], ys: tuple[int, int]) -> tuple[float, float]:
        """从累积相量 arg 的线性拟合求 (kx, ky)（格点单位 rad/格）。

        A ∝ exp(-i(kx x + ky y))，故 arg A 对 x/y 的斜率分别为 -kx/-ky。
        """
        amp = self._amp[xs[0]:xs[1], ys[0]:ys[1]]
        ph = np.unwrap(np.unwrap(np.angle(amp), axis=0), axis=1)
        n_x, n_y = ph.shape
        xx = np.arange(n_x)[:, None] * np.ones((1, n_y))
        yy = np.ones((n_x, 1)) * np.arange(n_y)[None, :]
        kx = -np.sum((xx - xx.mean()) * (ph - ph.mean())) / np.sum((xx - xx.mean())**2)
        ky = -np.sum((yy - yy.mean()) * (ph - ph.mean())) / np.sum((yy - yy.mean())**2)
        return float(kx), float(ky)


def laplacian_symbol(kx_dx: float, ky_dx: float, isotropic: bool) -> float:
    """返回 dx²·L 的 Fourier 符号（应为负）：-4[sx²+sy²-(2/3)sx²sy²] 或 -4[sx²+sy²]。"""
    sx2 = np.sin(kx_dx / 2.0) ** 2
    sy2 = np.sin(ky_dx / 2.0) ** 2
    if isotropic:
        return -4.0 * (sx2 + sy2 - (2.0 / 3.0) * sx2 * sy2)
    return -4.0 * (sx2 + sy2)
