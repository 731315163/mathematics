from collections.abc import Sequence
from numbers import Real
from typing import overload

import numpy as np
import pandas as pd


@overload
def clamp(x: int | float, min_v: int | float, max_v: int | float) -> int | float: ...
@overload
def clamp(x: Real, min_v: Real, max_v: Real) -> Real: ...
@overload
def clamp(
    x: np.number,
    min_v: np.number,
    max_v: np.number,
) -> np.number: ...
def clamp(
    x: Real | int | float | np.number,
    min_v: Real | int | float | np.number,
    max_v: Real | int | float | np.number,
):
    return max(min_v, min(x, max_v))


def divide(
    x: np.number | Sequence | np.ndarray | pd.Series | Real | int | float,
    y: np.number | Sequence | np.ndarray | pd.Series | Real | int | float,
):
    """
    安全数组除法
    """
    if isinstance(x, Sequence):
        x = np.array(x)
    if isinstance(y, Sequence):
        y = np.array(y)
    xscale = np.isscalar(x)
    yscale = np.isscalar(y)

    if xscale and yscale:
        if y == 0 or not np.isfinite(y):
            return np.nan
        else:
            return x / y  # type: ignore

    if yscale:
        num = len(x)  # type: ignore
    elif xscale:
        num = len(y)  # type: ignore
    else:
        num = max(len(x), len(y))  # type: ignore
    zeorary = np.full(num, np.nan)
    return np.divide(x, y, out=zeorary, where=(y != 0 & np.isfinite(y)))  # type: ignore
