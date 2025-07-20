import inspect

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def pdffun(x, x1, const, a, dis, loc, scal):
    return const + a * stats.norm.pdf(x + dis, loc, scal) + x1 * x


def pdf(x, r, loc, sigma):
    return r * np.exp(-np.power(x - loc, 2) / np.power(sigma, 2))


def softplut(x, rlog, xr, b):
    return rlog * np.log(1 + np.exp((x + b) * xr))


def sin(x, sr, xr, b):
    return sr * np.sin(xr * np.pi * (x + b))


def log(base, x):
    return np.log(x) / np.log(base)


def e1log(x, sr, xr, b):
    return sr * log((1 / np.e), xr * (x + b))


def pdfsoftplus(x, const, r, loc, sigma, rlog, xr, b):
    return const + pdf(x, r, loc, sigma) + softplut(x=x, rlog=rlog, xr=xr, b=b)


def pdf2softplus(x, const, r1, loc1, scal1, r2, loc2, scal2, sr, xr, b):
    return (
        const
        + pdf(x, r1, loc1, scal1)
        + pdf(x, r2, loc2, scal2)
        + softplut(x, sr, xr, b)
    )


def pdf2(x, const, r1, mu1, sigma1, r2, mu2, sigma2):
    return const + pdf(x, r1, mu1, sigma1) + pdf(x, r2, mu2, sigma2)


# def get_pdf2() -> tuple[Parameters, Callable[..., Any]]:
#     p = Parameters()
#     p.add(name="const", value=7)
#     p.add(name="r1", value=33)
#     p.add(name="mu1", value=8)
#     p.add(name="sigma1", value=-60)
#     p.add(name="r2", value=-33)
#     p.add(name="mu2", value=3)
#     p.add(name="sigma2", value=-29)

#     return p, pdf2


def pdf2sin(x, const, r1, loc1, scal1, r2, loc2, scal2, sr, xr, b):
    return const + pdf(x, r1, loc1, scal1) + pdf(x, r2, loc2, scal2) + sin(x, sr, xr, b)


def pdf2e1log(x, const, r1, loc1, scal1, r2, loc2, scal2, sr, xr, b):
    return (
        const + pdf(x, r1, loc1, scal1) + pdf(x, r2, loc2, scal2) + e1log(x, sr, xr, b)
    )


def pdf3(x, const, r1, loc1, scal1, r2, loc2, scal2, r3, loc3, scal3):
    return (
        const
        + pdf(x, r1, loc1, scal1)
        + pdf(x, r2, loc2, scal2)
        + pdf(x, r3, loc3, scal3)
    )


def pdf3x1(x, const, r1, loc1, scal1, r2, loc2, scal2, r3, loc3, scal3, x1, a1, b1):
    return (
        const
        + pdf(x, r1, loc1, scal1)
        + pdf(x, r2, loc2, scal2)
        + pdf(x, r3, loc3, scal3)
        + x1 * np.power(a1, x - b1)
    )


def polym(x, const, pdf, dis, loc, scal, x1, b1, x2, b2, x3, b3, x4, b4, x5, b5):
    return (
        const
        + x1 * np.power(x - b1, 1)
        + x2 * np.power(x - b2, 2)
        + x3 * np.power(x - b3, 3)
        + x4 * np.power(x - b4, 4)
        + x5 * np.power(x - b5, 5)
    )


def create_loss_fun(fun, x, real_y):
    def loss_fun(params):
        predict = fun(x, *params)
        result = predict - real_y
        return result

    return loss_fun


def create_fcn(fun, x, real_y):
    # 使用params进行计算
    # 可以返回残差数组，但更常见的是返回一个标量值（如成本、损失等）
    def loss_fun(params, *args, **kwargs):

        predict = fun(x, **params)
        result = predict - real_y
        return result

    return loss_fun


def get_random_params(fun, bounds=(-1000, 1000)):
    num = inspect.getfullargspec(fun).args
    p0 = np.random.uniform(bounds[0], bounds[1], len(num) - 1)
    return p0


# def fit_minimize(fun, x, real_y, params: Parameters | None = None):
#     # if params is None:
#     #     params = get_random_params(fun)
#     # res = minimize(
#     #     fun=create_loss_fun(fun, x, real_y),
#     #     x0=params,
#     #     method="powell",
#     #     options={"maxiter": 12400},
#     # )  # nelder-mead powell
#     # print(res)
#     # return res.x
#     minimizing = minimize(create_fcn(fun, x, real_y), params)

#     return minimizing.params.valuesdict()  # type: ignore


def fit(x, y, fun):
    """
    brent()：单变量无约束优化问题，混合使用牛顿法/二分法。
    fmin()：多变量无约束优化问题，使用单纯性法，只需要利用函数值，不需要函数的导数或二阶导数。
    leatsq()：非线性最小二乘问题，用于求解非线性最小二乘拟合问题。
    minimize()：约束优化问题，使用拉格朗日乘子法将约束优化转化为无约束优化问题。
    """

    ratio, b = curve_fit(f=fun, xdata=x, ydata=y, method="lm")
    return ratio


def tryfit(x, y, fun, maxnum=1000, num=0):
    try:
        return fit(x, y, fun)
    except Exception as e:
        print(f"{num} : {e}")
        if num < maxnum:
            return tryfit(x=x, y=y, fun=fun, maxnum=maxnum, num=num + 1)


def getfun(x, y, fun):
    f = curve_fit(f=fun, xdata=x, ydata=y, method="trf")
    return f
