# 导入所需的模块
import numpy as np
import pytest

import mathematics as ma  # 从mathefun模块导入divide函数
from mathematics import divide
@pytest.mark.parametrize("x, a_min,a_max, expected", [
    (5,2, 20, 5),
    (1.5,3, 2, 2),
    (-1,-1, 1, -1),
])
def test_clip(x,a_min,a_max, expected):
    cv = np.clip(x, a_min, a_max)
    assert cv == expected
# 测试标量除法
def test_divide_scalars():
    assert ma.divide(x=10, y=2) == 5
    assert ma.divide(x=3, y=2) == 1.5
    assert ma.divide(x=-1, y=1) == -1
    assert np.isnan(ma.divide(x=1, y=0))  # 0作为分母应该返回NaN
    assert np.isnan(ma.divide(x=1, y=np.inf))  # 无穷大作为分母应该返回NaN
    assert np.isnan(ma.divide(x=1, y=np.nan))


# 测试数组除法
def test_divide_arrays():
    x = np.array([1, 2, 3])
    y = np.array([2, 2, 2])
    expected_result = np.array([0.5, 1, 1.5])
    np.testing.assert_array_almost_equal(divide(x, y), expected_result)

    x = np.array([1, 2, 3])
    y = np.array([0, np.nan, 0])
    assert np.isnan(divide(x, y)).all()  # 分母全为0应该返回全NaN数组


# 测试混合型除法（标量和数组）
def test_divide_scalar_array():
    scalar = 10
    array = np.array([1, 2, 3])
    expected_result = np.array([10, 5, 10 / 3])
    np.testing.assert_array_almost_equal(divide(scalar, array), expected_result)

    scalar = 0
    array = np.array([1, 2, 3])
    expected_result = np.array([0, 0, 0])
    np.testing.assert_array_almost_equal(divide(scalar, array), expected_result)


# # 测试无效输入
# def test_divide_invalid_input():
#     with pytest.raises(TypeError):
#         divide("string", 2)  # 预期类型错误将引发异常
#     with pytest.raises(TypeError):
#         divide(2, "string")


# 测试特殊值
def test_divide_special_values():
    x = np.nan
    y = np.inf
    assert np.isnan(divide(x, y))  # 输入为NaN应该返回NaN
    assert np.isnan(divide(y, x))  # 输入为无穷除以NaN应该返回NaN
