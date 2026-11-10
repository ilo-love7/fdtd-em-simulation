"""一阶 Mur 吸收边界（1D/2D）。

1D（Ez 分量，左端 i=0）：
    E0^{n+1} = E1^n + (c dt - dx) / (c dt + dx) * (E1^{n+1} - E0^n)
2D TMz 的左右边界对 Ez 同式；上下边界同式（交换空间下标）。
"""
from __future__ import annotations

import numpy as np


def mur1_update(E_new: np.ndarray, E_old: np.ndarray, coef: float, axis: int, side: str) -> None:
    """就地写入 E_new 边界值。

    coef = (c·dt − h) / (c·dt + h)；axis=0 为 x（左右），axis=1 为 y（上下）。
    side = 'lo'/'hi'。E 数组形状 (nx,) 或 (nx, ny)。
    """
    if axis == 0:
        if side == "lo":
            E_new[0] = E_old[1] + coef * (E_new[1] - E_old[0])
        else:
            E_new[-1] = E_old[-2] + coef * (E_new[-2] - E_old[-1])
    else:
        if side == "lo":
            E_new[:, 0] = E_old[:, 1] + coef * (E_new[:, 1] - E_old[:, 0])
        else:
            E_new[:, -1] = E_old[:, -2] + coef * (E_new[:, -2] - E_old[:, -1])
