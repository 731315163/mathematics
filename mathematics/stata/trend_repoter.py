import numpy as np

from mathematics.metric import MAE, MAPE, MSE, R2
from mathematics.type import SequenceType
from .trend import (
    trend_hl,
    trend_hlatr,
    trend_momentum_hlatr,
    trend_num,
    trend_segments,
)


def polyfit_segment(x: SequenceType, y: SequenceType, epsilon=1e-4):
    segments = trend_segments(ary=y, epsilon=epsilon)
    results = []

    for start, end in segments:
        x_seg = x[start:end]
        y_seg = y[start:end]

        if len(x_seg) < 2:
            continue

        coeffs = np.polyfit(x_seg, y_seg, 1)  # 线性拟合
        slope, intercept = coeffs
        y_pred = np.polyval(coeffs, x_seg)
        # 计算误差

        results.append(
            {
                "segment": (start, end),
                "slope": slope,
                "intercept": intercept,
                "pred": y_pred,
            }
        )

    return results


def polyfit_segment_pred(
    y: SequenceType, segments: SequenceType, x: SequenceType | None = None
):

    if x is None:
        x = np.arange(len(y))
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    results = np.empty(len(y))
    for start, end in segments:
        x_seg = x[start:end]
        y_seg = y[start:end]

        if len(x_seg) < 2:
            continue

        coeffs = np.polyfit(x_seg, y_seg, 1)  # 线性拟合
        y_pred = np.polyval(coeffs, x_seg)
        # 计算误差
        results[start:end] = y_pred

    return results


def polyfit_MAPE(
    y: SequenceType,
    mask_segment: SequenceType,
    x: SequenceType | None = None,
    epsilon=1e-4,
):
    trend_nums = np.sum(trend_num(mask_segment))
    segments = trend_segments(ary=mask_segment, epsilon=epsilon)
    y_pred = polyfit_segment_pred(y=y, x=x, segments=segments)
    mape = MAPE(y_true=y, y_pred=y_pred)
    return -(mape * trend_nums)


def report_trend_hl(
    high: SequenceType, low: SequenceType, omega: float, start_from: int = 1
):
    mask_segments = trend_hl(high, low, omega, start_from)
    h = polyfit_MAPE(y=high, mask_segment=mask_segments, epsilon=1e-4)
    l = polyfit_MAPE(y=low, mask_segment=mask_segments, epsilon=1e-4)
    return (h + l) / 2


# def normal(data):

#     # 统计检验
#     shapiro_test = stats.shapiro(data)  # Shapiro-Wilk（n≤5000）
#     ks_test = stats.kstest(data, "norm")  # K-S检验（需指定参数）
#     jb_test = stats.jarque_bera(data)  # Jarque-Bera检验
#     return {"shapiro_test": shapiro_test, "ks_test": ks_test, "jb_test": jb_test}
