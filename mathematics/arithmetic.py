from numbers import Real
from typing import overload
from mathematics.type import SequenceType
import numpy as np


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
    x: np.number |SequenceType | Real | int | float,
    y: np.number | SequenceType  | Real | int | float,
):
    """
    安全数组除法
    """
    if isinstance(x, SequenceType):
        x = np.array(x)
    if isinstance(y, SequenceType):
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



def generate_arithmetic_sequence( total,n=None, d=None):
    """
    生成等差数组
    n: 元素个数，与d不能同时为None
    total: 数组总和，必须提供，不能为None
    d: 公差，与n不能同时为None
    """
    # 验证参数合法性
    if total is None:
        raise ValueError("total不能为None")
    elif n is None and d is not None:
        if d > 0 :
            start ,stop =0,total+d
        else:
            start,stop = total,d
        arr= np.arange(start=start,stop=stop,step=d)
        arr[-1]=total
        arr[0] = 0
        return arr
    # 处理d为None的情况（从0开始，计算公差）
    elif d is None and n is not None:
        return np.linspace(0, total, abs(n))
    else:
        raise ValueError("undefined error")
    
   
def generate_geometric_sequence(total,n=None, r=None ):
    """
    生成等比数组，从0开始末尾为total
    
    参数:
        n: 元素个数（与r二选一，不能同时提供或同时为None）
        r: 公比 * 暂不不支持
        total: 数组元素总和（必须提供）
    
    返回:
        np.array: 等比数组
    """
    # 参数验证
    if total is None:
        raise ValueError("total必须提供，不能为None")
    if n is not None and n <= 0:
        return np.array([])
    
    # 处理n为None的情况（已知公比r，计算元素个数）
    if n is None and r is not None:
        raise NotImplementedError("暂不支持未知公比r的情况")
    # 处理r为None的情况（已知元素个数n，计算公比）
    elif n is not None and r is None:  # r is None
        direction = 1 if total > 1 else -1
        if abs(total) >= 1:
            start,stop = 1*direction,total
            arr= np.geomspace(start=start,stop= stop, num=n,endpoint=True)
        else:
            start,stop = 1- abs(total),1
            arr= np.geomspace(start=start,stop= stop, num=n,endpoint=True)

            arr =  np.flip(arr)
            if direction < 0:
                arr = arr*direction

        arr[0] = 0
        return arr
    else:
        raise ValueError("n和r只能提供其中一个，不能同时提供,且不能同时为None")
 
    
    

def generate_fibonacci_sequence( total,n:int):
    """
    生成斐波那契数组（按比例缩放以满足总和要求）
    n: 元素个数
    total: 数组元素总和
    """
    if n <= 0:
        return np.array([])
    elif n == 1:
        return np.array([total])
    
    # 生成标准斐波那契数列
    fib = np.zeros(n, dtype=np.float64)
    fib[0], fib[1] = 0, 1
    for i in range(2, n):
        fib[i] = fib[i-1] + fib[i-2]
    
    # 计算缩放因子
    fib_sum = fib.sum()
    scale_factor = total / fib_sum if fib_sum != 0 else 0
    # 应用缩放因子
    return fib * scale_factor

