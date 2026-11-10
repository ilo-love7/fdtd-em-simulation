"""解析理论与频谱分析：FFT、Fresnel 反射、单缝衍射、Bragg 条件。"""
from __future__ import annotations

import numpy as np


# ---------- 频谱 ----------
def rfft_spectrum(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (freq_Hz, complex_Y) 单边谱（去直流）。"""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 2:
        raise ValueError("时序过短")
    y = y - y.mean()
    dt = float(t[1] - t[0])
    win = np.hanning(t.size)  # 减少谱泄漏
    Y = np.fft.rfft(y * win)
    f = np.fft.rfftfreq(t.size, dt)
    return f, Y


def interp_complex(f_grid: np.ndarray, f: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """在 f_grid 上线性插值复谱（实虚分别插值）。"""
    re = np.interp(f_grid, f, Y.real)
    im = np.interp(f_grid, f, Y.imag)
    return re + 1j * im


# ---------- 一维 Fresnel（正入射） ----------
def fresnel_r(n1: float, n2: float) -> float:
    """正入射电场反射系数 r = (n1 − n2)/(n1 + n2)。"""
    return (n1 - n2) / (n1 + n2)


def fresnel_t(n1: float, n2: float) -> float:
    """正入射电场透射系数 t = 2 n1/(n1 + n2)。"""
    return 2.0 * n1 / (n1 + n2)


def fresnel_R(n1: float, n2: float) -> float:
    """功率反射率 R = r²。"""
    return fresnel_r(n1, n2) ** 2


# ---------- 单缝 Fraunhofer 衍射 ----------
def single_slit_intensity(sin_theta: np.ndarray, a_over_lambda: float) -> np.ndarray:
    """I(θ)/I0 = sinc²(π a sinθ/λ)，sinc(x)=sin(x)/x（numpy 归一化 sinc 用 np.sinc(x/π)）。"""
    x = np.pi * a_over_lambda * np.asarray(sin_theta, dtype=float)
    return np.sinc(x / np.pi) ** 2


def single_slit_minima(a_over_lambda: float, m_max: int = 5) -> np.ndarray:
    """极小值 sinθ = mλ/a（m = ±1, ±2, …），返回正侧 m=1..m_max 的 sinθ（超出 ±1 的丢弃）。"""
    out = []
    for m in range(1, m_max + 1):
        s = m / a_over_lambda
        if 0 < s <= 1.0:
            out.append(s)
    return np.array(out)


# ---------- 1D Bragg 条件 ----------
def bragg_wavelength(n1: float, d1: float, n2: float, d2: float) -> float:
    """一维四分之一波长堆栈的一阶带隙中心：λ0 = 2 (n1 d1 + n2 d2)。"""
    return 2.0 * (n1 * d1 + n2 * d2)


def quarter_wave_thickness(n: float, lam0: float) -> float:
    """四分之一波长厚度 d = λ0/(4n)。"""
    return lam0 / (4.0 * n)


# ---------- 误差指标 ----------
def band_edges_from_transmission(f: np.ndarray, T: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
    """从透射谱 T(f) 找低于 threshold 的连续带（带隙），返回 (f_lo, f_hi)；找不到返回 (nan, nan)。"""
    f = np.asarray(f, dtype=float)
    T = np.asarray(T, dtype=float)
    below = T < threshold
    if not below.any():
        return float("nan"), float("nan")
    # 取最宽的连续低于阈值区间
    best_i = best_j = -1
    i = None
    for k, b in enumerate(below):
        if b and i is None:
            i = k
        elif not b and i is not None:
            if best_i < 0 or (k - i) > (best_j - best_i):
                best_i, best_j = i, k - 1
            i = None
    if i is not None and (best_i < 0 or (len(below) - i) > (best_j - best_i)):
        best_i, best_j = i, len(below) - 1
    if best_i < 0:
        return float("nan"), float("nan")
    return float(f[best_i]), float(f[best_j])


def rel_err(sim: float, ref: float) -> float:
    """相对误差（百分比）。ref=0 时返回绝对差。"""
    if ref == 0:
        return abs(sim - ref)
    return abs(sim / ref - 1.0) * 100.0


# ---------- 时窗 + FFT 反射系数 ----------
def windowed_fft_ratio(
    t: np.ndarray,
    y_inc: np.ndarray,
    y_ref: np.ndarray,
    n_center_inc: int,
    n_center_ref: int,
    half_width: int,
    f_eval: float | None = None,
) -> dict:
    """在两个时窗内分别 FFT，求反射系数 r = Y_ref(f)/Y_inc(f)。

    y_inc/y_ref 是同一探针的完整时序（入射段与反射段重叠存放），
    n_center_* 是各自峰值步数，half_width 为半窗宽（步）。
    f_eval 为评估频率；None 时取入射窗谱峰频率。
    """
    t = np.asarray(t, dtype=float)
    y_inc = np.asarray(y_inc, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    i0 = max(0, n_center_inc - half_width)
    i1 = min(t.size, n_center_inc + half_width)
    j0 = max(0, n_center_ref - half_width)
    j1 = min(t.size, n_center_ref + half_width)
    wi, wj = y_inc[i0:i1].copy(), y_ref[j0:j1].copy()
    ti, tj = t[i0:i1], t[j0:j1]
    # 用同一长度与窗函数（长度不等时截齐）
    n = min(wi.size, wj.size)
    wi, wj = wi[:n], wj[:n]
    win = np.hanning(n)
    dt = float(ti[1] - ti[0]) if ti.size > 1 else 1.0
    Yi = np.fft.rfft((wi - wi.mean()) * win)
    Yr = np.fft.rfft((wj - wj.mean()) * win)
    f = np.fft.rfftfreq(n, dt)
    if f_eval is None:
        k = int(np.argmax(np.abs(Yi)))
    else:
        k = int(np.argmin(np.abs(f - f_eval)))
    r = Yr[k] / Yi[k] if abs(Yi[k]) > 0 else complex(np.nan, np.nan)
    return {
        "f_eval": float(f[k]),
        "r_complex": complex(r),
        "r_abs": float(abs(r)),
        "r_phase_deg": float(np.degrees(np.angle(r))),
        "window_n": n,
    }
