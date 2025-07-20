"""Interpolation algorithms using piecewise cubic polynomials."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

# import arch.covariance
# import arch.covariance.kernel
# import arch.data
# import arch.unitroot
import numpy as np
from scipy.interpolate import CubicHermiteSpline, interp1d


def prepare_input(x, y, axis, dydx=None):
    """Prepare input for cubic spline interpolators.

    All data are converted to numpy arrays and checked for correctness.
    Axes equal to `axis` of arrays `y` and `dydx` are moved to be the 0th
    axis. The value of `axis` is converted to lie in
    [0, number of dimensions of `y`).
    """

    x, y = map(np.asarray, (x, y))
    if np.issubdtype(x.dtype, np.complexfloating):
        raise ValueError("`x` must contain real values.")
    x = x.astype(float)

    if np.issubdtype(y.dtype, np.complexfloating):
        dtype = complex
    else:
        dtype = float

    if dydx is not None:
        dydx = np.asarray(dydx)
        if y.shape != dydx.shape:
            raise ValueError("The shapes of `y` and `dydx` must be identical.")
        if np.issubdtype(dydx.dtype, np.complexfloating):
            dtype = complex
        dydx = dydx.astype(dtype, copy=False)

    y = y.astype(dtype, copy=False)
    axis = axis % y.ndim
    if x.ndim != 1:
        raise ValueError("`x` must be 1-dimensional.")
    if x.shape[0] < 2:
        raise ValueError("`x` must contain at least 2 elements.")
    if x.shape[0] != y.shape[axis]:
        raise ValueError(
            f"The length of `y` along `axis`={axis} doesn't " "match the length of `x`"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError("`x` must contain only finite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("`y` must contain only finite values.")

    if dydx is not None and not np.all(np.isfinite(dydx)):
        raise ValueError("`dydx` must contain only finite values.")

    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("`x` must be strictly increasing sequence.")

    y = np.moveaxis(y, axis, 0)
    if dydx is not None:
        dydx = np.moveaxis(dydx, axis, 0)

    return x, dx, y, axis, dydx


class Akima1DInterpolator(CubicHermiteSpline):
    """
    Akima interpolator

    Fit piecewise cubic polynomials, given vectors x and y. The interpolation
    method by Akima uses a continuously differentiable sub-spline built from
    piecewise cubic polynomials. The resultant curve passes through the given
    data points and will appear smooth and natural.

    Parameters
    ----------
    x : ndarray, shape (npoints, )
        1-D array of monotonically increasing real values.
    y : ndarray, shape (..., npoints, ...)
        N-D array of real values. The length of ``y`` along the interpolation axis
        must be equal to the length of ``x``. Use the ``axis`` parameter to
        select the interpolation axis.

        .. deprecated:: 1.13.0
            Complex data is deprecated and will raise an error in SciPy 1.15.0.
            If you are trying to use the real components of the passed array,
            use ``np.real`` on ``y``.

    axis : int, optional
        Axis in the ``y`` array corresponding to the x-coordinate values. Defaults
        to ``axis=0``.
    method : {'akima', 'makima'}, optional
        If ``"makima"``, use the modified Akima interpolation [2]_.
        Defaults to ``"akima"``, use the Akima interpolation [1]_.

        .. versionadded:: 1.13.0

    Methods
    -------
    __call__
    derivative
    antiderivative
    roots

    See Also
    --------
    PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.
    CubicSpline : Cubic spline data interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints

    Notes
    -----
    .. versionadded:: 0.14

    Use only for precise data, as the fitted curve passes through the given
    points exactly. This routine is useful for plotting a pleasingly smooth
    curve through a few given points for purposes of plotting.

    Let :math:`delta_i = (y_{i+1} - y_i) / (x_{i+1} - x_i)` be the slopes of
    the interval :math:`left[x_i, x_{i+1}right)`. Akima's derivative at
    :math:`x_i` is defined as:

    .. math::

        d_i = frac{w_1}{w_1 + w_2}delta_{i-1} + frac{w_2}{w_1 + w_2}delta_i

    In the Akima interpolation [1]_ (``method="akima"``), the weights are:

    .. math::

        begin{aligned}
        w_1 &= |delta_{i+1} - delta_i| 
        w_2 &= |delta_{i-1} - delta_{i-2}|
        end{aligned}

    In the modified Akima interpolation [2]_ (``method="makima"``),
    to eliminate overshoot and avoid edge cases of both numerator and
    denominator being equal to 0, the weights are modified as follows:

    .. math::

        begin{align*}
        w_1 &= |delta_{i+1} - delta_i| + |delta_{i+1} + delta_i| / 2 
        w_2 &= |delta_{i-1} - delta_{i-2}| + |delta_{i-1} + delta_{i-2}| / 2
        end{align*}

    Examples
    --------
    Comparison of ``method="akima"`` and ``method="makima"``:

    >>> import numpy as np
    >>> from scipy.interpolate import Akima1DInterpolator
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(1, 7, 7)
    >>> y = np.array([-1, -1, -1, 0, 1, 1, 1])
    >>> xs = np.linspace(min(x), max(x), num=100)
    >>> y_akima = Akima1DInterpolator(x, y, method="akima")(xs)
    >>> y_makima = Akima1DInterpolator(x, y, method="makima")(xs)

    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y, "o", label="data")
    >>> ax.plot(xs, y_akima, label="akima")
    >>> ax.plot(xs, y_makima, label="makima")
    >>> ax.legend()
    >>> fig.show()

    The overshoot that occured in ``"akima"`` has been avoided in ``"makima"``.

    References
    ----------
    .. [1] A new method of interpolation and smooth curve fitting based
           on local procedures. Hiroshi Akima, J. ACM, October 1970, 17(4),
           589-602. :doi:`10.1145/321607.321609`
    .. [2] Makima Piecewise Cubic Interpolation. Cleve Moler and Cosmin Ionita, 2019.
           https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/

    """

    def __init__(self, x, y, axis=0, *, method: Literal["akima", "makima"] = "akima"):
        if method not in {"akima", "makima"}:
            raise NotImplementedError(f"`method`={method} is unsupported.")
        # Original implementation in MATLAB by N. Shamsundar (BSD licensed), see
        # https://www.mathworks.com/matlabcentral/fileexchange/1814-akima-interpolation
        x, dx, y, axis, _ = prepare_input(x, y, axis)

        if np.iscomplexobj(y):
            msg = (
                "`Akima1DInterpolator` only works with real values for `y`. "
                "Passing an array with a complex dtype for `y` is deprecated "
                "and will raise an error in SciPy 1.15.0. If you are trying to "
                "use the real components of the passed array, use `np.real` on "
                "the array before passing to `Akima1DInterpolator`."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        # determine slopes between breakpoints
        m = np.empty((x.size + 3,) + y.shape[1:])
        dx = dx[(slice(None),) + (None,) * (y.ndim - 1)]
        m[2:-2] = np.diff(y, axis=0) / dx

        # add two additional points on the left ...
        m[1] = 2.0 * m[2] - m[3]
        m[0] = 2.0 * m[1] - m[2]
        # ... and on the right
        m[-2] = 2.0 * m[-3] - m[-4]
        m[-1] = 2.0 * m[-2] - m[-3]

        # if m1 == m2 != m3 == m4, the slope at the breakpoint is not
        # defined. This is the fill value:
        t = 0.5 * (m[3:] + m[:-3])
        # get the denominator of the slope t
        dm = np.abs(np.diff(m, axis=0))
        if method == "makima":
            pm = np.abs(m[1:] + m[:-1])
            f1 = dm[2:] + 0.5 * pm[2:]
            f2 = dm[:-2] + 0.5 * pm[:-2]
        else:
            f1 = dm[2:]
            f2 = dm[:-2]
        f12 = f1 + f2
        # These are the mask of where the slope at breakpoint is defined:
        ind = np.nonzero(f12 > 1e-9 * np.max(f12, initial=-np.inf))
        x_ind, y_ind = ind[0], ind[1:]
        # Set the slope at breakpoint
        t[ind] = (
            f1[ind] * m[(x_ind + 1,) + y_ind] + f2[ind] * m[(x_ind + 2,) + y_ind]
        ) / f12[ind]

        super().__init__(x, y, t, axis=0, extrapolate=False)
        self.axis = axis

    def extend(self, c, x, right=True):
        raise NotImplementedError(
            "Extending a 1-D Akima interpolator is not " "yet implemented"
        )

    # These are inherited from PPoly, but they do not produce an Akima
    # interpolator. Hence stub them out.
    @classmethod
    def from_spline(cls, tck, extrapolate=None):
        raise NotImplementedError(
            "This method does not make sense for " "an Akima interpolator."
        )

    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=None):
        raise NotImplementedError(
            "This method does not make sense for " "an Akima interpolator."
        )


numeric_types = [
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Float32",
    "Float64",
]


def linear_interpolator(x):

    # 假设你有一个包含None值的数组
    data = np.array([1, 2, None, 4, 5, None, 7])

    # 将None替换为np.nan
    data_nan = np.nan_to_num(data, nan=np.nan)

    # 获取非NaN值的索引
    non_nan_idx = ~np.isnan(data_nan)

    # 获取非NaN的x和y值（在这个例子中，我们假设x是数据的索引）
    x = np.arange(len(data_nan))[non_nan_idx]
    y = data_nan[non_nan_idx]

    # 创建一个插值函数
    f = interp1d(x, y, kind="linear")

    # 创建一个新的x数组，用于插值（即原始数据的索引）
    x_new = np.arange(len(data_nan))

    # 使用插值函数获取新的y值
    data_interp = f(x_new)
    return data_interp


# def akima_impute_df(
#     df: pd.DataFrame,
#     columns: Sequence[str] | None = None,
#     method: Literal["akima", "makima"] = "makima",
# ):
#     df.replace({None: np.nan}, inplace=True)
#     if columns is None or len(columns) <= 0:
#         columns = list(df.columns)
#     for col in columns:
#         if not df[col].isna().any() or df[col].dtype not in numeric_types:
#             continue
#         # 获取非缺失值的索引和值
#         copy_values = pd.Series(df[col])
#         copy_values.reset_index(drop=True, inplace=True)
#         vaild = copy_values.dropna()
#         valid_indices = vaild.index
#         valid_values = vaild.values

#         # 获取这些值的x坐标（在这种情况下，我们只是使用它们的索引）
#         x = valid_indices.values

#         # 创建Akima插值器
#         interpolator = Akima1DInterpolator(x, valid_values, method=method)

#         # 创建一个新的x坐标数组，用于插值（例如，所有的索引）
#         x_new = np.arange(df.shape[0])

#         # 使用插值器计算新的y值

#         y_new = interpolator(x_new)

#         # 用新的y值替换原始DataFrame中的缺失值
#         df.loc[df[col].isna(), col] = y_new[df[col].isna()]
#     return df
