import numpy as np
from .type import SequenceType
def R2(y_true:SequenceType, y_pred:SequenceType):
    """
    计算R²（确定系数）
 
    参数:
    y_true: numpy数组，包含真实值
    y_pred: numpy数组，包含预测值
 
    返回:
    R²值，一个浮点数,越接近1越好
    """
    # 确保y_true和y_pred是numpy数组
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 计算观测值的平均值
    mean_y_true = np.mean(y_true)
    
    # 计算总平方和（TSS）
    ss_tot = np.sum((y_true - mean_y_true) ** 2)
    
    # 计算残差平方和（RSS）
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # 计算R²
    r2 = 1 - (ss_res / ss_tot)
    
    return r2



def MSE(y_seg:SequenceType,y_pred:SequenceType):
     return np.mean((np.array(y_seg) -np.array( y_pred)) ** 2)


def MAE(y_seg:SequenceType,y_pred:SequenceType):
     return  np.mean(np.abs(np.array( y_seg) -np.array( y_pred)))




def MAPE(y_true:SequenceType, y_pred:SequenceType):
    """
    计算平均绝对百分比误差（MAPE）
    
    参数：
        y_true: 真实值数组或列表
        y_pred: 预测值数组或列表
    
    返回：
        MAPE值，单位为百分比
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 检查是否存在真实值为0的情况，以避免除以0的错误
    if np.any(y_true == 0):
        raise ValueError("y_true 中存在0值，无法计算 MAPE。")
    
    # 计算绝对百分比误差
    percentage_errors = np.abs((y_true - y_pred) / y_true)
    
    # 返回平均值并转换为百分比
    mape = np.mean(percentage_errors) * 100
    return mape