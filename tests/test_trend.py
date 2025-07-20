from mathematics.stata import slopeR
import pytest
import numpy as np
def test_p_value_below_threshold():

        assert slopeR([1,2,3], [2,4,5], 1.0) == 0

def test_mae_exceeds_error():
        assert slopeR([1,2,3], [10,20,30], 5.0) == 10

def test_normal_return_value():
        """测试正常返回计算结果（用例03）"""
      
        expected = 2.0 * abs(0.9)
        assert slopeR([1,2,3], [2.5,4.5,6.5], 0.5) == pytest.approx(expected)

def test_numpy_array_input():
        """测试numpy数组输入兼容性（用例04）"""
        x = np.array([1,2,3])
        y = np.array([2.1,4.0,6.2])
        assert isinstance(slopeR(x, y, 0.3), float)

# def test_pandas_series_input():
#         """测试pandas序列输入兼容性（用例05）"""
#         x = pd.Series([1,2,3])
#         y = pd.Series([2.0,3.9,6.1])
#         assert isinstance(slopeR(x, y, 0.2), float)





# 通用测试数据
@pytest.fixture
def valid_data():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]  # 完美线性关系（p_value=0）
    return x, y

# --------------------------
# 测试异常场景
# --------------------------
def test_length_mismatch():
    x = [1, 2, 3]
    y = [4, 5]
    with pytest.raises(ValueError, match="x and y must have the same length"):
        slopeR(x, y, 0.1)

# --------------------------
# 参数化测试核心逻辑
# --------------------------
@pytest.mark.parametrize("x, y, err, expected", [
    # p_value > 0.05 时返回0（随机噪声数据）
    (
        [1, 2, 3, 4, 5],
        [0.1, 0.3, -0.2, 0.4, 0.5],
        0.5,
        0
    ),
    # MAE > err 时返回0（预测误差过大）
    (
        [1, 2, 3],
        [10, 20, 30],
        1.0,
        0
    ),
    # 正常情况（完美线性关系）
    (
        [1, 2, 3, 4, 5],
        [2, 4, 6, 8, 10],
        0.0,
        2.0  # slope=2，r_value=1 → K * |r| = 2 * 1 = 2
    ),
    # 非线性但满足条件（r_value=0.8）
    (
        [1, 2, 3, 4, 5],
        [2.1, 3.9, 6.0, 8.2, 9.9],
        0.5,
        pytest.approx(1.6, abs=0.1)  # slope≈2, r≈0.8 → 2 * 0.8=1.6
    )
])
def test_slopeR_logic(x, y, err, expected):
    result = slopeR(x, y, err)
    assert result == expected

# --------------------------
# 边界条件测试
# --------------------------
def test_single_point(valid_data):
    x, y = valid_data
    # 单点输入会导致 x[-1]-x[0]=0 → 除零错误
    with pytest.raises(ZeroDivisionError):
        slopeR([5], [5], 0.1)