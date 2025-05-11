import numpy as np
from scipy.fft import fft, ifft

from .type import SequenceType


def fit_fourier(y: SequenceType, num_freqs=None):
    """
    对数据进行傅立叶变换，并根据主要频率成分进行拟合。

    参数:
    x (np.ndarray): 自变量数组
    y (np.ndarray): 因变量数组
    num_freqs (int, optional): 保留的主要频率成分的数量。如果为None，则保留所有频率成分。

    返回:
    np.ndarray: 拟合的数据
    """
    if np.nan in y:
        raise ValueError(" 'y'  contain nan,", y)

    # 进行傅立叶变换
    y_fft: np.ndarray = fft(y)  # type: ignore

    # 计算频率
    # n = len(x)
    # timestep = x[1] - x[0]
    # n = len(x)
    # timestep = x[1] - x[0]
    # freq = np.fft.fftfreq(n, d=timestep)

    # 筛选主要频率成分
    if num_freqs is not None:

        indices = np.argsort(np.abs(y_fft))[-num_freqs:]
        y_fft_filtered = np.zeros_like(y_fft)
        y_fft_filtered[indices] = y_fft[indices]
    else:
        y_fft_filtered = y_fft

    # 进行逆傅立叶变换
    y_fit = ifft(y_fft_filtered)

    return y_fit.real  # type: ignore
