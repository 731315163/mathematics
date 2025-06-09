import math
import time
from typing import cast

import numpy as np
import pandas as pd
import pytest

from tradeutils.volumeprofile import (VolumeDistribution,
                                             analyze_volumeprofile,
                                             get_analyze_volumeprofiles,
                                             get_volumedistributions)
from utils.testutil import equal


@pytest.fixture
def VD():
    """创建基础测试实例（范围5-10，初始值np.nan）"""
    return VolumeDistribution(range(5, 10), init_value=np.nan)

# 参数化数据
index_in_range = [(5, 10), (7, 20), (9, "test")]
index_out_range = [(4, -5), (10, 3.14), (1000, object())]

def test_initialization(VD: VolumeDistribution):
    # 验证数组初始化
    #assert len(VD.array) == 5
  
    assert VD.sparse_array == {}
    assert VD.max_index == -1
    assert VD.default_value is np.nan




@pytest.mark.parametrize("index, default", [
    (5, np.nan),    # 范围内未赋值
    (8, np.nan),    # 范围内未赋值
    (100, np.nan) ,  # 范围外未赋值
    (range(0,10),[np.nan]*10)
])
def test_getitem_default(VD, index, default):
    if default is np.nan:
        assert np.isnan(VD[index])
    else:
        assert VD[index] == default

@pytest.mark.parametrize("index, value", [
    (0,100),(100,20),(-1,-2),(11,'f'),(slice(0,None),np.arange(10)),(slice(None,None),np.arange(10)),(slice(None,-1),np.arange(10)),(range(0,10),np.arange(0,10))

])
def test_getitem_after_set(VD, index, value):
    VD[index] = value

    assert equal( VD[index] , value ) # 验证赋值后读取




       

def test_max_index_update(VD):
    VD[15] = "new"  # 范围外赋值
    assert VD.max_index == 15
    VD[8] = 20      # 范围内赋值
    assert VD.max_index == 15  # 不应更新



def test_length_calculation(VD: VolumeDistribution):
    # 初始状态（全nan）
    assert len(VD) == 0
    
    # 主数组有效值
    VD[5] = 10
    assert len(VD) == 1
    
    # 稀疏字典值
    VD[20] = 30
    assert len(VD) == 2
    
    # 混合状态
    VD[6] = 6
    VD[100] = 200
    assert len(VD) == 4



def test_iteration_sequence(VD: VolumeDistribution):
    # 设置测试数据
    test_data = {
        3: 3,     # 稀疏
        5: 10,      # 主数组
        8: 20,      # 主数组
        15: 30      # 稀疏
    }
    for k, v in test_data.items():
        VD[k] = v
  
    # 验证迭代顺序和值
    expected = [
        (0, np.nan), (1, np.nan), (2, np.nan),
        (3, 3), (4, np.nan),  # 前5个索引
        (5, 10), (6, np.nan), (7, np.nan), 
        (8, 20), (9, np.nan),   # 主数组范围
        (10, np.nan), (11, np.nan), 
        (12, np.nan), (13, np.nan), 
        (14, np.nan), (15, 30)  # 稀疏部分
    ]
    
    for i in range(0,VD.max_index+1 ):
        idx, volume = expected[i]
        assert idx  == i
        if np.isnan(volume):
            assert np.isnan(VD[i])
        else:
            assert volume == VD[i]
    
 




@pytest.fixture
def sparse_vd():
    """稀疏数据测试用例"""
    vd = VolumeDistribution(range(100, 105), init_value=np.nan)
    # 稀疏索引设置（主数组外的数据）
    vd[99] = 20  # 稀疏存储
    vd[105] = 25 # 稀疏存储
    return vd
# 在现有测试文件中追加以下测试内容

# --------------- 测试 get_volumedistributions ---------------
@pytest.fixture
def sample_price_data():
    """生成测试用价格/成交量数据"""
    return {
        'high': np.array([105, 110, 108, 115, 120], dtype=float),
        'low': np.array([95, 105, 103, 110, 115], dtype=float),
        'vol': np.array([1000, 1500, 800, 2000, 2500], dtype=float)
    } 



def test_sliding_sumvolume(sample_price_data):
    """测试窗口滑动时的加减逻辑"""
    timeperiod = 2
    result = get_volumedistributions(
        sample_price_data['high'],
        sample_price_data['low'],
        sample_price_data['vol'],
        timeperiod=timeperiod
    )
    for k in range(timeperiod-1,len(result)):
        print("---------------------")
        filtered_data = [x for x in result[k] if not pd.isna(x)]
        value = np.sum(filtered_data)
       
        print(f"{k} : {value}")
        expected = np.sum([ y for y in sample_price_data['vol'][k+1-timeperiod:k+1] if not pd.isna(y)])
        assert pytest.approx(value) == expected
   

# --------------- 测试 get_analyze_volumeprofiles ---------------
def test_full_workflow(sample_price_data):
    """测试完整工作流"""
    high, low = get_analyze_volumeprofiles(
        sample_price_data['high'],
        sample_price_data['low'],
        sample_price_data['vol'],
        timeperiod=2,
        PV_percent=(0.5, 0.6)
    )
    
    # 验证结果维度
    assert len(high) == len(sample_price_data['high'])
    assert len(low) == len(sample_price_data['low'])
    
    # 验证首个结果应为NaN（时间周期之前）
    assert np.isnan(high[0])
    assert np.isnan(low[0])
    print(high)
    print(low)
    # 验证有效结果计算
    assert 110 < high[2] < 115
    assert 105 < low[2] < 110

# --------------- 边界条件测试 ---------------
def test_empty_input():
    """测试空输入处理"""
    with pytest.raises(ValueError):
        get_volumedistributions([], [], [], 1)
        
    

@pytest.mark.parametrize("timeperiod", [1, 5, 10])
def test_various_timeperiods(sample_price_data, timeperiod):
    """不同时间周期参数测试"""
    result = get_volumedistributions(
        sample_price_data['high'],
        sample_price_data['low'],
        sample_price_data['vol'],
        timeperiod
    )
    
    # 验证前timeperiod-1个元素未处理
    for i in range(timeperiod-1):
        assert np.all(np.isnan(result[i].array))




@pytest.fixture
def initialized_vd():
    """初始化带锚点的VolumeDistribution实例"""
    vd = VolumeDistribution(range(100), 0)
    vd.set_anchor(100, 1000.0)  # 设置标准索引对应1000元
    return vd

def test_invalid_pv_percent():
    with pytest.raises(ValueError):
        vd = VolumeDistribution(range(100), 0)
        analyze_volumeprofile(vd, (1.5, 0.5))  # price_percent超出范围

def test_unset_anchor():
    vd = VolumeDistribution(range(100), 0)
    high, low = vd.analyze_volumeprofile((0.8, 0.8))
    assert pd.isna(high) and pd.isna(low)  # 未设置锚点应返回NaN[8](@ref)

@pytest.mark.parametrize("price_percent, expected_range", [
    (0.3, (800, 820)),  
    (0.3, (900, 950)),
])
def test_normal_case(initialized_vd: VolumeDistribution, price_percent, expected_range):
    # 填充测试数据
    low_p,high_p = expected_range
    length = 10
    prince = np.linspace(low_p-10,high_p+10,10)
    volume = [2,1,3,1,100,100,2,1,3,1]
    for i in range(0,length):
        initialized_vd.add_volume(i, volume,prince  )  # 生成递增价格
    
   
    high, low = initialized_vd.analyze_volumeprofile((price_percent, 0.3))
    print(low,high)
    # 验证价格区间
    assert expected_range[0] <= low <= high <= expected_range[1]

def test_boundary_condition(initialized_vd: VolumeDistribution):
    # 当price_percent=1时应覆盖全部价格
    high, low = initialized_vd.analyze_volumeprofile((1.0, 0.1))
    assert low == 0.0 and high == 1000.0  # 根据锚点计算[6](@ref)

def test_insufficient_volume():
    vd = VolumeDistribution(range(10), 0)
    vd.set_anchor(10, 100)
    high, low = vd.analyze_volumeprofile((0.5, 0.9))  # 成交量不足90%
    assert pd.isna(high) and pd.isna(low)