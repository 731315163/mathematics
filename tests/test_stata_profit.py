import pytest
from pytest import approx

from mathematics.stata import (
    PriceVolume,
    calculate_avgprice,
    calculate_avgprice_appendprofit,
    calculate_avgprice_profit,
    calculate_profit,
    calculate_profit_with_short,
)

# 假设 PriceVolume 类型定义在同一个文件中


@pytest.mark.parametrize(
    "orders,exp_real,exp_vol",
    [
        ([], 0, 0),
        ([(10, 0), (20, 0)], 0, 0),  # "所有数量为零的订单列表应返回 (0, 0)"
        ([(10, 2), (20, 3)], 80 / 5, 5),
        ([(10, -2), (20, -3)], 80 / 5, 5),
        ([(10, 2), (20, -3)], 80 / 5, 5),
        ([(100.1, 0.0), (200.2, 0.0)], 0, 0),
        ([(100.1, -10.1), (200.2, -20.2)], (100.1 * 10.1 + 200.2 * 20.2) / 30.3, 30.3),
        (
            [(100.1, 10.1), (200.2, 20.2), (300.3, 30.3)],
            (100.1 * 10.1 + 200.2 * 20.2 + 300.3 * 30.3) / 60.6,
            60.6,
        ),
    ],
)
def test_caculate_avgprice(orders, exp_real, exp_vol):

    real, vol = calculate_avgprice(orders)
    assert approx(real) == exp_real
    assert approx(vol) == exp_vol


order_cases = [
    ( 170,
        # 买入订单（平均价160，总量5）
        [(100.0, 2.0), (200.0, 3.0)],
        # 卖出订单（平均价170，总量3）
        [(150.0, 1.0), (180.0, 2.0)],
        30.0,  # (170-160)*3 = 30
        80.0,  # (200-160)*(5-3) = 80
    ),
    (
        170,
        # 互换买卖订单（原卖出变买入，原买入变卖出）
        [(150.0, 1.0), (180.0, 2.0)],
        [(100.0, 2.0), (200.0, 3.0)],
        -50.0,  # (160-170)*5 = -50
        -500.0,  # (200-170)*(-2) = -60 → 实际会抛出异常（见问题分析）
    ),
    (   150,
        [(100.0, 5.0)],
        [(150.0, 3.0)],
        150.0,  # (150-100)*3
        0.0,  # (160-100)*2 = 120 → 需要修正测试数据
    ),
]


@pytest.mark.parametrize(
    "current_rate, buy_orders, sell_orders,exp_real,exp_float", order_cases
)
def test_profit_calculations(
    current_rate, buy_orders, sell_orders, exp_real, exp_float
):
    """参数化测试核心逻辑"""
    realized, floating = calculate_profit(current_rate, buy_orders, sell_orders)

    assert realized == approx(exp_real)
    assert floating == approx(exp_float)


def test_oversell_protection():
    """卖单总量超过买单的异常验证"""
    with pytest.raises(ValueError) as excinfo:
        calculate_profit(
            current_rate=200.0,
            buy_orders=[(100.0, 5.0)],
            sell_orders=[(150.0, 6.0)],  # 卖出量6 > 买入量5
        )

    error_msg = str(excinfo.value)
    assert error_msg


def test_empty_orders():
    """空订单的特殊情况处理"""
    # 完全空订单
    realized, floating = calculate_profit(100.0, [], [])
    assert realized == 0.0
    assert floating == 0.0

    # 仅卖出订单为空
    realized, floating = calculate_profit(
        current_rate=200.0, buy_orders=[(150.0, 3.0)], sell_orders=[]
    )
    assert realized == 0.0
    assert floating == (200 - 150) * 3  # 200当前价，150买入均价，3持仓量


@pytest.mark.parametrize(
    "current_rate,buy_orders,sell_orders,expected_realized,expected_unrealized",
    [
        # 正常多头场景
        (
            150,
            [(100, 2), (200, 3)],  # 买入均价 = (100 * 2 + 200 * 3)/5 = 160
            [(180, 4)],  # 卖出均价 = 180
            (180 - 160) * 4,
            (150 - 160) * (5 - 4),
        ),  # 已实现 80，浮动 -10
        # 空订单边界
        (200, [], [], 0, 0),
        # 做空场景（通过with_short函数）
        (
            80,
            [(100, 3)],  # 平仓买入订单
            [(120, 5)],  # 开仓卖出订单
            (100 - 120) * 3,
            (80 - 120) * (5 - 3),
        ),  # 原始计算后取反
    ],
)
def test_profit_calculation(
    current_rate, buy_orders, sell_orders, expected_realized, expected_unrealized
):
    realized, unrealized = calculate_profit(current_rate, buy_orders, sell_orders)
    assert realized == pytest.approx(expected_realized)
    assert unrealized == pytest.approx(expected_unrealized)


def test_short_position():
    # 做空测试（开仓价120，平仓价100）
    realized, unrealized = calculate_profit_with_short(
        current_rate=80,
        is_short=True,
        buy_orders=[(100, 3)],  # 平仓订单
        sell_orders=[(120, 5)],  # 开仓订单
    )
    assert realized == pytest.approx((120 - 100) * 3)  # 正确方向
    assert unrealized == pytest.approx((120 - 80) * (5 - 3))  # (120-80)*2=80


@pytest.mark.parametrize(
    "sell_vol,buy_vol", [(5, 3), (0, 0)]  # 卖出量>买入量（合法场景）  # 零交易
)
def test_negative_balance(sell_vol, buy_vol):
    # 验证不再抛出异常
    buy = [(100, buy_vol)]
    sell = [(100, sell_vol)]
    _, unrealized = calculate_profit(100, buy, sell)
    assert unrealized == (100 - 100) * (buy_vol - sell_vol)
