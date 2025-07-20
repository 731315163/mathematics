import math
from scipy import stats
import numpy as np
from mathematics.metric import MAE
from mathematics.type import SequenceGenericType

SequenceType = SequenceGenericType[int | float | np.number]

def min_max(seq:SequenceType):
    norma = np.asarray(seq)
    min_val, max_val = np.min(norma), np.max(norma)
    normalized = (norma - min_val) / (max_val - min_val)
    return normalized, min_val, max_val
def inverse_min_max(normalized_seq, min_val, max_val):
    return normalized_seq * (max_val - min_val) + min_val
def slopeR(x:SequenceType,y:SequenceType,err=None):
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
 
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    
    if err :
        y_pred = [slope*xi+intercept for xi in x]
        mae = MAE(y,y_pred)
        if mae > err:
            return 0
    # 将角度归一化到 [-1, 1] 区间
    K = math.atan(slope) / (math.pi / 2)
    # 斜率 [-1, 1]，一元一次方程的参数与r值
    return K*abs(r_value)

def linregress(x:SequenceType|tuple,y:SequenceType,err=None):
    
    if isinstance(x,tuple):
        x_normal,min_x,max_x = x
    else:
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        x_normal,min_x,max_x = min_max(x)
    y_normal,min_y,max_y = min_max(y)
    
    slope_norm, intercept_norm, r_value, p_value, _ = stats.linregress(x_normal, y_normal)
    scale_y = max_y - min_y
    scale_x = max_x - min_x
    slope_original = slope_norm * (scale_y / scale_x)  # 修正：考虑x轴缩放
    intercept_original = min_y + (intercept_norm - slope_norm * min_x / scale_x) * scale_y 
    # 还原截距到原始尺度
    
    
    if err :
        y_pred = [slope_original*xi+intercept_original for xi in x]
        mae = MAE(y,y_pred)
        if mae > err:
            return 0
    # 将角度归一化到 [-1, 1] 区间
    K = math.atan(slope_norm) / (math.pi / 2)
    # 斜率 [-1, 1]，一元一次方程的参数与r值
    # if r_value < 0:
    #     raise ValueError(f"Invalid parameters: r_value {r_value} must be positive.")
    return  slope_original, intercept_original,r_value,K*abs(r_value)

def corrcoef(x:SequenceType,y:SequenceType):
    r= np.corrcoef(x,y)[0,1]
    return np.sign(r) * np.sqrt(r)
    

def trend_num(arr: SequenceType, rel_tol=1e-9):
    def compare_zero(num: float | int):
        if num > rel_tol:
            return 1
        elif num < -rel_tol:
            return -1
        return 0

    # 处理数组为空的情况
    if len(arr) <= 0:
        return 0, 0, 0

    # 初始化计数器
    state_count = {0: 0, 1: 0, -1: 0}

    # 初始化当前状态
    current_state = arr[0]

    # 遍历数组（从第二个元素开始，因为第一个元素已经用于初始化当前状态）
    for num in arr[1:]:
        # 如果当前数字与上一个状态不同，说明遇到了新的状态
        if num != current_state:
            # 根据上一个状态更新计数器
            state_count[compare_zero(current_state)] += 1
            # 更新当前状态为新的数字
            current_state = num

    # 处理最后一个状态
    state_count[compare_zero(current_state)] += 1

    return state_count[-1], state_count[0], state_count[1]


def trend_segments(ary: SequenceType, epsilon=1e-9):
    segments = []
    start = 0
    if len(ary) < 1:
        return segments

    for i in range(1, len(ary)):
        if (ary[i] > epsilon and ary[i - 1] <= epsilon) or (
            ary[i] < -epsilon and ary[i - 1] >= -epsilon
        ):
            segments.append((start, i))
            start = i

    segments.append((start, len(ary)))
    return segments


def trend_hl(high: SequenceType, low: SequenceType, omega: float, start_from: int = 1):
    # very good
    N = len(high)
    if N != len(low):
        raise ValueError("high and low must have the same length")
    if start_from > N:
        raise ValueError("start_from must be less than N")
    HT = int(np.argmax(high[0:start_from]))

    LT = int(np.argmin(low[0:start_from]))
    xH = high[HT]  # Highest price
    xL = low[LT]  # Lowest price
    Cid = 0  # Current direction of labeling (1 for up, -1 for down)
    Directions = np.zeros(N, dtype=int)  # Label vector initialized with zeros

    for i in range(start_from, N):
        xhi = high[i]
        xli = low[i]
        ti = i

        if Cid > 0:  # Current trend is up
            if xhi > xH:
                xH, HT = xhi, ti
            elif xhi < xH * (1 - omega) and LT < HT:
                # Label the range from LT+1 to HT as up (inclusive of LT but exclusive of HT)
                Directions[LT:HT] = 1
                # Update lowest price, time, and change trend direction to down
                xL, LT, Cid = xli, ti, -1

        elif Cid < 0:  # Current trend is down
            if xli < xL:
                xL, LT = xli, ti
            elif xli > xL + xL * omega and HT < LT:
                # Label the range from HT+1 to LT as down (inclusive of HT but exclusive of LT)
                Directions[HT:LT] = -1
                # Update highest price, time, and change trend direction to up
                xH, HT, Cid = xhi, ti, 1
        else:
            if xhi > xH * (1 + omega):
                xH, HT, Cid = xhi, ti, 1
            elif xli < xL * (1 - omega):
                xL, LT, Cid = xli, ti, -1
    if HT > LT:
        Directions[LT:N] = 1
    elif HT < LT:
        Directions[HT:N] = -1
    return Directions


def trend_hlatr(high: SequenceType, low: SequenceType, atr: SequenceType):
    # very good
    N = len(high)
    if N != len(low) or N != len(atr):
        raise ValueError("high and low must have the same length")
    xH = high[0]  # Highest price
    xL = low[0]  # Lowest price
    HT = 0  # Time when highest price occurs
    LT = 0  # Time when lowest price occurs
    Cid = 0  # Current direction of labeling (1 for up, -1 for down)
    Directions = np.zeros(N, dtype=int)  # Label vector initialized with zeros

    for i in range(0, N):
        xhi = high[i]
        xli = low[i]
        ti = i
        delta = atr[i]

        if Cid > 0:  # Current trend is up
            if xhi > xH:
                xH, HT = xhi, ti
            elif xhi < xH - delta and LT < HT:
                # Label the range from LT+1 to HT as up (inclusive of LT but exclusive of HT)
                Directions[LT:HT] = 1
                # Update lowest price, time, and change trend direction to down
                xL, LT, Cid = xli, ti, -1

        elif Cid < 0:  # Current trend is down
            if xli < xL:
                xL, LT = xli, ti
            elif xli > xL + delta and HT < LT:
                # Label the range from HT+1 to LT as down (inclusive of HT but exclusive of LT)
                Directions[HT:LT] = -1
                # Update highest price, time, and change trend direction to up
                xH, HT, Cid = xhi, ti, 1
        else:
            if xhi > xH + delta:
                xH, HT, Cid = xhi, ti, 1
            elif xli < xL - delta:
                xL, LT, Cid = xli, ti, -1
    if HT > LT:
        Directions[LT:N] = 1
    elif HT < LT:
        Directions[HT:N] = -1
    return Directions


def trend(dataX: SequenceType, omega: float):
    """
    Function to label trends in a time series data.

    Parameters:
    dataX (list or numpy array): The original time series data.
    omega (float): The proportion threshold parameter for trend definition.

    Returns:
    numpy array: The label vector indicating trend directions.
    """

    if omega <= 0:
        raise ValueError("omega must be a positive number")

    # Initialization of variables
    N = len(dataX)
    FP = dataX[0]  # First price
    xH = dataX[0]  # Highest price

    xL = dataX[0]  # Lowest price
    HT = 0  # Time when highest price occurs
    LT = 0  # Time when lowest price occurs
    Cid = 0  # Current direction of labeling (1 for up, -1 for down)
    FP_N = 0  # Index of the highest or lowest point obtained initially
    Y = np.zeros(N, dtype=int)  # Label vector initialized with zeros

    # Calculate thresholds outside the loop
    FP_up_threshold = FP * (1 + omega)
    FP_down_threshold = FP * (1 - omega)

    # First loop to find initial trend direction and set FP_N
    for i in range(N):
        xi = dataX[i]
        ti = i

        if xi > FP_up_threshold:
            xH, HT, FP_N, Cid = xi, ti, i, 1
            break
        elif xi < FP_down_threshold:
            xL, LT, FP_N, Cid = xi, ti, i, -1
            break

    # Second loop to label the trends
    for i in range(FP_N + 1, N):
        xi = dataX[i]
        ti = i

        if Cid > 0:  # Current trend is up
            if xi > xH:
                xH, HT = xi, ti
            elif xi < xH - xH * omega and LT < HT:
                # Label the range from LT+1 to HT as up (inclusive of LT but exclusive of HT)
                Y[LT:HT] = 1
                # Update lowest price, time, and change trend direction to down
                xL, LT, Cid = xi, ti, -1

        elif Cid < 0:  # Current trend is down
            if xi < xL:
                xL, LT = xi, ti
            elif xi > xL + xL * omega and HT < LT:
                # Label the range from HT+1 to LT as down (inclusive of HT but exclusive of LT)
                Y[HT:LT] = -1
                # Update highest price, time, and change trend direction to up
                xH, HT, Cid = xi, ti, 1

    # Special case: If the last trend was not labeled due to no reversal, handle it
    # (This part might need adjustment based on specific requirements)
    if HT > LT:
        Y[LT:N] = 1
    elif HT < LT:
        Y[HT:N] = -1
    return Y


def trend_momentum_hlatr(
    high: SequenceType, low: SequenceType, atr: SequenceType, start_from: int = 1
):
    # very good
    N = len(high)
    if N != len(low) or N != len(atr):
        raise ValueError("high and low must have the same length")
    HT = int(np.argmax(high[0:start_from]))

    LT = int(np.argmin(low[0:start_from]))
    xH = high[HT]  # Highest price
    xL = low[LT]  # Lowest price
    # Time when highest price occurs
    # Time when lowest price occurs
    Cid = 0  # Current direction of labeling (1 for up, -1 for down)
    Directions = np.zeros(N, dtype=float)  # Label vector initialized with zeros

    for i in range(start_from, N):
        xhi = high[i]
        xli = low[i]
        ti = i
        delta = atr[i]

        if Cid > 0.01:  # Current trend is up
            if xhi > xH:
                xH, HT = xhi, ti
            elif xhi < xH - delta and LT < HT:
                # Label the range from LT+1 to HT as up (inclusive of LT but exclusive of HT)
                Directions[LT:HT] = Cid
                # Update lowest price, time, and change trend direction to down
                xL, LT, Cid = xli, ti, (xhi - xH) / delta

        elif Cid < -0.01:  # Current trend is down
            if xli < xL:
                xL, LT = xli, ti
            elif xli > xL + delta and HT < LT:
                # Label the range from HT+1 to LT as down (inclusive of HT but exclusive of LT)
                Directions[HT:LT] = Cid
                # Update highest price, time, and change trend direction to up
                xH, HT, Cid = xhi, ti, (xli - xL) / delta
        else:
            if xhi > xH + delta:
                xH, HT, Cid = xhi, ti, (xhi - xH) / delta
            elif xli < xL - delta:
                xL, LT, Cid = xli, ti, (xli - xL) / delta
    if HT > LT:
        Directions[LT:N] = Cid
    elif HT < LT:
        Directions[HT:N] = Cid
    return Directions


# def trend_df(df: pd.DataFrame, atr:str="atr" , high: str = "high", low: str = "low"):

#     return trend_hlatr(high=df[high].to_numpy(), low=df[low].to_numpy(), atr=atr)
