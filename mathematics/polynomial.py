import numpy as np


def fit_polynomial(x, y, num_freqs=3):

    # 使用fit方法进行拟合
    # p = Polynomial.fit(x, y, 2)
    z1 = np.polyfit(x, y, num_freqs)  # 用3次多项式拟合
    p1 = np.poly1d(z1)
    deriv = p1.deriv()
    deriv(x)
    yvals = p1(x)  # 也可以使用yvals=np.polyval(z1,x)
    return yvals


def fit_deriv(x, y, num_freqs=3):
    # 使用fit方法进行拟合
    # p = Polynomial.fit(x, y, 2)
    z1 = np.polyfit(x, y, num_freqs)  # 用3次多项式拟合
    p1 = np.poly1d(z1)
    deriv = p1.deriv()

    print(p1)  # 在屏幕上打印拟合多项式
    yvals = deriv(x)  # 也可以使用yvals=np.polyval(z1,x)
    return yvals