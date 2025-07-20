
import pytest
from mathematics.arithmetic import generate_arithmetic_sequence,generate_fibonacci_sequence,generate_geometric_sequence
import numpy as np
import math

# 等差数列测试
def test_arithmetic_basic_cases():
    # 基本功能测试
    arr = generate_arithmetic_sequence(n=5, total=100)
    assert np.array_equal(arr,[  0,  25,  50,  75, 100])
    # 小数值测试
    arr = generate_arithmetic_sequence(n=3, total=6)
    assert np.array_equal(arr,[  0,  3,6])
    
    # 负数公差测试
    arr = generate_arithmetic_sequence(total=10, d=-1)
    assert np.array_equal(arr,[ 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,0])

def test_arithmetic_none_parameters():
    # n为None的情况
    arr = generate_arithmetic_sequence(total=15, d=2)
    assert math.isclose(arr[-1]-15,0) 
    assert len(arr) == 9 # 0，1+2+3+4+5=15
    
    # d为None的情况
    arr = generate_arithmetic_sequence(n=4, total=20)
    assert math.isclose(arr[-1] - 20,0)
    assert len(arr) == 4

def test_arithmetic_edge_cases():
    # n=1的情况
    arr = generate_arithmetic_sequence(n=1, total=5, d=10)
    assert np.array_equal(arr, [5])
    
    # n=2的情况
    arr = generate_arithmetic_sequence(n=2, total=10, d=2)
    assert np.allclose(arr, [4, 6])  # 4+6=10，差为2
    
    # 总和为0的情况
    arr = generate_arithmetic_sequence(n=3, total=0, d=2)
    assert np.allclose(arr.sum(), 0)
    assert np.allclose(np.diff(arr), 2)

def test_arithmetic_error_conditions():
    with pytest.raises(ValueError, match="total不能为None"):
        generate_arithmetic_sequence(n=5, total=None, d=2)
    
    with pytest.raises(ValueError):
        generate_arithmetic_sequence(n=None, total=100, d=None)
    
    with pytest.raises(ZeroDivisionError):
        generate_arithmetic_sequence(n=None, total=100, d=0)

# 等比数列测试
def test_geometric_basic_cases():
    # 基本功能测试
    arr = generate_geometric_sequence(n=4, total=25)
    #[ 0.          1.          3.27491722 10.72508278]
    print(arr)
    assert np.allclose(arr[-1], 15)
    assert len(arr) == 4
    assert np.allclose(arr, [1, 2, 4, 8])  # 1+2+4+8=15
    
    # 公比小于1的情况
    arr = generate_geometric_sequence( total=7/4, r=1/2)
    assert np.allclose(arr[-1], 7/4)
    assert np.allclose(arr, [1, 0.5, 0.25])

def test_geometric_none_parameters():
    # n为None的情况
    arr = generate_geometric_sequence(total=15, r=2)
    assert abs(arr.sum() - 15) < 1e-6
    assert len(arr) == 4  # 1+2+4+8=15
    
    # r为None的情况
    arr = generate_geometric_sequence(n=4, total=15)
    assert abs(arr.sum() - 15) < 1e-6
    assert len(arr) == 4
    ratios = arr[1:]/arr[:-1]
    assert np.allclose(ratios, ratios[0], atol=1e-6)  # 所有公比应相等

def test_geometric_edge_cases():
    # n=1的情况
    arr = generate_geometric_sequence(n=1, total=10, r=3)
    assert np.array_equal(arr, [10])
    
    # n=2的情况
    arr = generate_geometric_sequence(n=2, total=6, r=2)
    assert np.allclose(arr, [2, 4])  # 2+4=6，比为2
    
    # 总和为0的情况（只有n=1可能）
    arr = generate_geometric_sequence(n=1, total=0, r=2)
    assert np.array_equal(arr, [0])

def test_geometric_error_conditions():
    with pytest.raises(ValueError, match="total不能为None"):
        generate_geometric_sequence(n=5, total=None, r=2)
    
    with pytest.raises(ValueError, match="n和r不能同时为None"):
        generate_geometric_sequence(n=None, total=100, r=None)
    
    with pytest.raises(ValueError, match="公比r不能为1"):
        generate_geometric_sequence(n=5, total=100, r=1)

# 斐波那契数列测试
def test_fibonacci_basic_cases():
    # 基本功能测试
    arr = generate_fibonacci_sequence(n=5, total=10)
    assert abs(arr.sum() - 10) < 1e-6
    assert len(arr) == 5
    # 验证斐波那契比例 (每个数是前两个数之和)
    assert np.allclose(arr[2], arr[0] + arr[1], atol=1e-6)
    assert np.allclose(arr[3], arr[1] + arr[2], atol=1e-6)
    
    # 标准斐波那契数列缩放测试
    std_fib = np.array([0, 1, 1, 2, 3, 5])
    scaled = generate_fibonacci_sequence(n=6, total=std_fib.sum() * 2)
    assert np.allclose(scaled, std_fib * 2)

def test_fibonacci_none_parameters():
    # n为None的情况
    arr = generate_fibonacci_sequence(n = 6,total=12)
    # 标准斐波那契前6项和为0+1+1+2+3+5=12，所以应该返回这6项
    assert abs(arr.sum() - 12) < 1e-6
    assert len(arr) == 6

def test_fibonacci_edge_cases():
    # n=1的情况
    arr = generate_fibonacci_sequence(n=1, total=5)
    assert np.array_equal(arr, [5])
    
    # n=2的情况
    arr = generate_fibonacci_sequence(n=2, total=10)
    assert abs(arr.sum() - 10) < 1e-6
    assert len(arr) == 2
    
    # n=3的情况
    arr = generate_fibonacci_sequence(n=3, total=20)
    assert abs(arr.sum() - 20) < 1e-6
    assert np.allclose(arr[2], arr[0] + arr[1], atol=1e-6)

def test_fibonacci_error_conditions():
    with pytest.raises(TypeError):
        generate_fibonacci_sequence(n=5, total=None)

# 参数化测试 - 覆盖更多组合
@pytest.mark.parametrize("n, total, d, expected_diff", [
    (5, 50, 2, 2),
    (3, 30, -1, -1),
    (4, 24, None, 2),  # d由函数计算
    (None, 15, 1, 1),   # n由函数计算
])
def test_arithmetic_parametrized(n, total, d, expected_diff):
    arr = generate_arithmetic_sequence(n=n, total=total, d=d)
    assert abs(arr.sum() - total) < 1e-6
    if n:
        assert len(arr) == n
    if expected_diff:
        assert np.allclose(np.diff(arr), expected_diff, atol=1e-6)

@pytest.mark.parametrize("n, total, r, expected_ratio", [
    (4, 15, 2, 2),
    (3, 7/4, 1/2, 1/2),
    (5, 31, None, 2),   # r由函数计算
    (None, 31, 2, 2),    # n由函数计算
])
def test_geometric_parametrized(n, total, r, expected_ratio):
    arr = generate_geometric_sequence(n=n, total=total, r=r)
    assert abs(arr.sum() - total) < 1e-6
    if n:
        assert len(arr) == n
    if expected_ratio and len(arr) > 1:
        assert np.allclose(arr[1]/arr[0], expected_ratio, atol=1e-6)

@pytest.mark.parametrize("n, total, expected_length", [
    (5, 100, 5),
    (10, 50, 10),
    (None, 144, 12),  # 前12项斐波那契和为144
])
def test_fibonacci_parametrized(n, total, expected_length):
    arr = generate_fibonacci_sequence(n=n, total=total)
    assert abs(arr.sum() - total) < 1e-6
    assert len(arr) == expected_length




def test_arithmetic_normal_case():
    """测试正常情况：n=3, total=6, d=2 生成 [0, 2, 4]"""
    np.testing.assert_array_equal(
        generate_arithmetic_sequence(total=6, n=3, d=2),
        np.array([0, 2, 4])
    )

def test_arithmetic_n_none():
    """测试 n 为 None：d=2, total=6 应计算 n=3"""
    np.testing.assert_array_equal(
        generate_arithmetic_sequence(total=6, d=2),
        np.array([0, 2, 4])
    )

def test_arithmetic_d_none():
    """测试 d 为 None：n=3, total=6 应计算 d=2"""
    np.testing.assert_array_equal(
        generate_arithmetic_sequence(total=6, n=3),
        np.array([0, 2, 4])
    )




def test_arithmetic_n_zero_or_negative():
    """测试 n <= 0 返回空数组"""
    assert len(generate_arithmetic_sequence(total=5, n=0, d=1)) == 0
    assert len(generate_arithmetic_sequence(total=5, n=-2, d=1)) == 0

# ---------------------
# 测试 generate_geometric_sequence
# ---------------------

def test_geometric_normal_case():
    """测试正常情况：n=3, total=14, r=2 生成 [2, 4, 8]"""
    np.testing.assert_array_equal(
        generate_geometric_sequence(total=14, n=3, r=2),
        np.array([2, 4, 8])
    )

def test_geometric_n_none():
    """测试 n 为 None：r=2, total=14 应计算 n=3"""
    np.testing.assert_array_equal(
        generate_geometric_sequence(total=14, r=2),
        np.array([2, 4, 8])
    )

def test_geometric_r_none():
    """测试 r 为 None：n=3, total=14 应计算 r=2"""
    np.testing.assert_array_equal(
        generate_geometric_sequence(total=14, n=3),
        np.array([2, 4, 8])
    )



def test_geometric_n_zero_or_negative():
    """测试 n <= 0 返回空数组"""
    assert len(generate_geometric_sequence(total=5, n=0, r=2)) == 0
    assert len(generate_geometric_sequence(total=5, n=-2, r=2)) == 0

def test_geometric_ratio_less_than_1():
    """测试 r < 1 的情况：n=2, total=3, r=0.5 生成 [2.0, 1.0]"""
    np.testing.assert_array_equal(
        generate_geometric_sequence(total=3, n=2, r=0.5),
        np.array([2.0, 1.0])
    )

def test_geometric_negative_ratio():
    """测试负数 r：n=3, total=3, r=-2 生成 [1, -2, 4]"""
    np.testing.assert_array_equal(
        generate_geometric_sequence(total=3, n=3, r=-2),
        np.array([1, -2, 4])
    )

def test_geometric_ratio_zero():
    """测试 r=0 的情况：n=2, total=5 生成 [5, 0]"""
    np.testing.assert_array_equal(
        generate_geometric_sequence(total=5, n=2, r=0),
        np.array([5, 0])
    )

# ---------------------
# 测试 generate_fibonacci_sequence
# ---------------------

def test_fibonacci_normal_case():
    """测试正常情况：n=5, total=12 生成 [0, 1, 1, 2, 8]"""
    np.testing.assert_array_equal(
        generate_fibonacci_sequence(total=12, n=5),
        np.array([0, 1, 1, 2, 8])
    )

def test_fibonacci_n_none():
    """测试 n 为 None：total=12 应找到 n=5"""
    np.testing.assert_array_equal(
        generate_fibonacci_sequence(total=12, n=None),
        np.array([0, 1, 1, 2, 8])
    )

def test_fibonacci_n_one():
    """测试 n=1 返回 [total]"""
    np.testing.assert_array_equal(
        generate_fibonacci_sequence(total=7, n=1),
        np.array([7])
    )

def test_fibonacci_n_zero_or_negative():
    """测试 n <= 0 返回空数组"""
    assert len(generate_fibonacci_sequence(total=5, n=0)) == 0
    assert len(generate_fibonacci_sequence(total=5, n=-2)) == 0

def test_fibonacci_invalid_total():
    """测试 total 为 None 时抛出异常"""
    with pytest.raises(ValueError):
        generate_fibonacci_sequence(total=None, n=3)

# ---------------------
# 边界和异常测试
# ---------------------

def test_edge_cases_for_all_functions():
    """测试所有函数的边界情况"""
    # 等差数列边界情况
    np.testing.assert_array_equal(generate_arithmetic_sequence(total=5, n=1, d=0), np.array([5]))
    
    # 等比数列边界情况
    np.testing.assert_array_equal(generate_geometric_sequence(total=5, n=1, r=2), np.array([5]))
    
    # 斐波那契边界情况
    np.testing.assert_array_equal(generate_fibonacci_sequence(total=5, n=1), np.array([5]))




# ---------------------
# 参数化测试 generate_arithmetic_sequence
# ---------------------
@pytest.mark.parametrize(
    "total,n,d,expected",
    [
        (6, 3, 2, [0, 2, 4]),
        (6, None, 2, [0, 2, 4]),
        (9, 3, None, [0, 3, 6]),
    ]
)
def test_arithmetic_sequence(total, n, d, expected):
    """测试等差数列生成器的多种参数组合"""
    result = generate_arithmetic_sequence(total=total, n=n, d=d)
    np.testing.assert_array_equal(result, np.array(expected))

# ---------------------
# 参数化测试 generate_geometric_sequence
# ---------------------
@pytest.mark.parametrize(
    "total,n,r,expected",
    [
        (14, 4, 2, [0, 2, 4, 8]),
        (14, None, 2, [0, 2, 4, 8]),
        (14, 4, None, [0, 2, 4, 8]),
        (3, 3, 0.5, [0, 1.5, 0.75]),
        (3, 3, -2, [0, 1, -2]),
    ]
)
def test_geometric_sequence(total, n, r, expected):
    """测试等比数列生成器的多种参数组合"""
    if n is None and r is not None:
        result = generate_geometric_sequence(total=total, r=r)
    elif r is None and n is not None:
        result = generate_geometric_sequence(total=total, n=n)
    else:
        result = generate_geometric_sequence(total=total, n=n, r=r)
    np.testing.assert_allclose(result, np.array(expected), atol=1e-6)

# ---------------------
# 参数化测试 generate_fibonacci_sequence
# ---------------------
@pytest.mark.parametrize(
    "total,n,expected",
    [
        (12, 5, [0, 1, 1, 2, 8]),
        (12, None, [0, 1, 1, 2, 8]),
        (7, 1, [7]),
    ]
)
def test_fibonacci_sequence(total, n, expected):
    """测试斐波那契数列生成器的多种参数组合"""
    result = generate_fibonacci_sequence(total=total, n=n)
    np.testing.assert_allclose(result, np.array(expected), atol=1e-6)

# ---------------------
# 独立测试用例（异常/边界）
# ---------------------
def test_arithmetic_invalid_total():
    """测试 total 为 None 时抛出异常"""
    with pytest.raises(ValueError):
        generate_arithmetic_sequence(total=None, n=3, d=2)

def test_arithmetic_both_n_d_none():
    """测试 n 和 d 同时为 None 时抛出异常"""
    with pytest.raises(ValueError):
        generate_arithmetic_sequence(total=6)

def test_geometric_both_n_r_provided():
    """测试同时提供 n 和 r 时抛出异常"""
    with pytest.raises(ValueError):
        generate_geometric_sequence(total=5, n=2, r=2)

def test_geometric_r_equals_1():
    """测试 r=1 时抛出异常"""
    with pytest.raises(ValueError):
        generate_geometric_sequence(total=5, n=2, r=1)

# ---------------------
# 边界测试
# ---------------------
@pytest.mark.parametrize(
    "func,args,expected",
    [
        (generate_arithmetic_sequence, {"total": 5, "n": 1, "d": 0}, [5]),
        (generate_geometric_sequence, {"total": 5, "n": 1, "r": 2}, [5]),
        (generate_fibonacci_sequence, {"total": 5, "n": 1}, [5]),
    ]
)
def test_boundary_cases(func, args, expected):
    """测试所有函数的边界情况"""
    result = func(**args)
    np.testing.assert_array_equal(result, np.array(expected))