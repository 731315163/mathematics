import math
from typing import TypeAlias

import numpy as np

from mathematics.type import SequenceGenericType

PriceVolume: TypeAlias = (
    tuple[float, float] | tuple[int, int] | tuple[np.number, np.number]
)


def calculate_avgprice(
    orders: SequenceGenericType[PriceVolume],
) -> PriceVolume:
    """

    计算订单的平均价格。

    参数:
    orders (List[PriceVolume]): 一个订单列表，其中每个订单由价格和数量组成。

    返回:
    tuple: 返回一个元组，第一个元素是平均价格，第二个元素是总数量。
           如果总数量为0，则平均价格也为0。

    """
    sum_volume = 0
    sum_pv = 0
    for price, volume in orders:
        if np.isnan(price) or np.isnan(volume):
            continue
        volume = abs(volume)
        sum_volume += volume
        sum_pv += price * volume
    if sum_volume == 0:
        return 0, sum_volume
    else:
        return sum_pv / sum_volume, sum_volume


def calculate_avgprice_profit(
    buy_orders: SequenceGenericType[PriceVolume],
    sell_orders: SequenceGenericType[PriceVolume],
):
    """
    计算给定买卖订单列表的平均买入价格和卖出利润。

    参数：
    买入订单 (List[价量结构]) : 由价格和数量构成的买入订单列表
    卖出订单 (List[价量结构]) : 由价格和数量构成的卖出订单列表

    返回：
    Tuple[float, float] : 包含平均买入价格和卖出利润的元组

    本函数首先计算卖出订单的平均价格和总数量。接着反向遍历买入订单计算总买入成本和数量。
    若卖出数量超过买入数量，将抛出ValueError异常。最终计算平均买入价格和卖出利润并返回。
    """
    sell_avgprice, sell_vol = calculate_avgprice(sell_orders)

    buy_price, buy_vol, buy_cost = 0, -sell_vol, 0
    for price, volume in reversed(buy_orders):
        buy_vol += volume
        if buy_vol <= 0:
            continue
        elif buy_vol >= volume:
            buy_cost += volume * price
        else:
            buy_cost += buy_vol * price

    balance = buy_vol - sell_vol
    if balance < 0:
        raise ValueError(
            f"The sell volume : {sell_vol} must be less than or equal to the buy volume:{buy_vol}."
        )

    avg_price = 0 if math.isclose(buy_vol, 0.0) else buy_cost / buy_vol
    sell_profit = (sell_avgprice - buy_price) * sell_vol
    return avg_price, sell_profit


def calculate_avgprice_appendprofit(
    buy_orders: list[PriceVolume], sell_orders: list[PriceVolume]
):
    """
    根据买卖订单计算平均买入价格和卖出利润

    参数:
    buy_orders (List[PriceVolume]): 包含价格和数量的买入订单列表
    sell_orders (List[PriceVolume]): 包含价格和数量的卖出订单列表

    返回:
    Tuple[float, float]: 包含平均买入价格和卖出利润的元组

    本函数首先计算卖出订单的平均价格和总成交量以确定卖出所得，
    然后计算买入订单的平均价格和总成交量以确定买入成本。
    接着计算成交量差异以确保卖出量不超过买入量。
    最后结合卖出所得计算考虑后的平均买入价格，以及卖出产生的利润，
    返回这两个计算结果。
    """
    sell_avgprice, sell_vol = calculate_avgprice(sell_orders)
    sell_gain = sell_avgprice * sell_vol
    buy_price, buy_vol = calculate_avgprice(buy_orders)
    buy_cost = buy_price * buy_vol
    balance = buy_vol - sell_vol
    if balance < 0:
        raise ValueError(
            f"The sell volume : {sell_vol} must be less than or equal to the buy volume:{buy_vol}."
        )
    buy_cost -= sell_gain
    avg_price = buy_price if math.isclose(balance, 0.0) else buy_cost / balance

    sell_profit = (sell_avgprice - buy_price) * sell_vol
    return avg_price, sell_profit


def calculate_profit(
    current_rate: float,
    buy_orders: SequenceGenericType[PriceVolume],
    sell_orders: SequenceGenericType[PriceVolume],
):
    """
    计算已实现利润和未实现利润

    参数：
    current_rate -- 当前市场价格/汇率，用于计算未实现利润
    buy_orders -- 买入订单列表，每个元素应包含价格和数量
    sell_orders -- 卖出订单列表，每个元素应包含价格和数量

    返回值：
    realized_profit -- 已实现的利润（通过实际卖出获得的利润）
    unrealized_profit -- 未实现的利润（持仓部分的浮动盈亏）
    balance
    异常：
    ValueError -- 当卖出量超过买入量时抛出
    """
    sell_avg, sell_vol = calculate_avgprice(sell_orders)
    buy_avg, buy_vol = calculate_avgprice(buy_orders)

    balance = buy_vol - sell_vol
    if balance < 0:
        raise ValueError(
            f"The sell volume : {sell_vol} must be less than or equal to the buy volume:{buy_vol}."
        )
    realized_profit = (sell_avg - buy_avg) * sell_vol
    unrealized_profit = (current_rate - buy_avg) * balance  # 允许balance为负

    return realized_profit, unrealized_profit


def calculate_profit_with_short(
    current_rate: float,
    is_short: bool,
    buy_orders: SequenceGenericType[PriceVolume],
    sell_orders: SequenceGenericType[PriceVolume],
):
    """计算的已实现和未实现利润

    Args:
        current_rate: 当前市场汇率/价格
        is_short: 是否做空，True表示做空仓位
        buy_orders: 买单列表，元素为(价格, 成交量)结构体
        sell_orders: 卖单列表，元素为(价格, 成交量)结构体

    Returns:
        tuple: 包含两个浮点数的元组，分别表示已实现利润和未实现利润

    实现逻辑：
    - 做空时，开仓操作是卖出，平仓操作是买入。利润计算时需要反转符号。
    - 非做空时直接调用基础利润计算方法。
    """
    if is_short:
        # 做空时：开仓是卖出，平仓是买入
        realized, unrealized = calculate_profit(
            current_rate,
            sell_orders,  # 原卖出订单视为开仓（做空）
            buy_orders,  # 原买入订单视为平仓
        )
        return -realized, -unrealized  # 利润方向反转
    else:
        return calculate_profit(current_rate, buy_orders, sell_orders)

class Cache_Profit:
    def __init__(
        self,
        pair: str,
        buy_orders: list[PriceVolume] = [],
        sell_orders: list[PriceVolume] = [],
    ):
        self.pair = pair
        self.buy_orders = buy_orders
        self.sell_orders = sell_orders
        self.balance = 0
        self.buy_price = 0
        self.realized_profit = 0

    def length(self):
        return len(self.buy_orders) + len(self.sell_orders)


    def calculate_cache_profit(self, current_rate: float, is_short: bool):
        if is_short:
            float_profit = (self.buy_price - current_rate) * self.balance
        float_profit = (current_rate - self.buy_price) * self.balance
        return self.realized_profit + float_profit

    def calculate_profit(
        self,
        current_rate: float,
        buy_orders: list[PriceVolume],
        sell_orders: list[PriceVolume],
    ):

        sell_avg, sell_vol = calculate_avgprice(sell_orders)
        buy_avg, buy_vol = calculate_avgprice(buy_orders)
        self.buy_price = buy_avg
        balance = buy_vol - sell_vol
        self.balance = balance
        if balance < 0:
            raise ValueError(
                f"The sell volume : {sell_vol} must be less than or equal to the buy volume:{buy_vol}."
            )
        realized_profit = (sell_avg - buy_avg) * sell_vol
        unrealized_profit = (current_rate - buy_avg) * balance  # 允许balance为负

        return realized_profit, unrealized_profit, balance

    def calculate_profit_with_short(
        self,
        current_rate: float,
        is_short: bool,
        buy_orders: list[PriceVolume],
        sell_orders: list[PriceVolume],
    ):
        self.buy_orders = buy_orders
        self.sell_orders = sell_orders
        if is_short:
            # 做空时：开仓是卖出，平仓是买入
            realized, unrealized, balance = self.calculate_profit(
                current_rate,
                sell_orders,  # 原卖出订单视为开仓（做空）
                buy_orders,  # 原买入订单视为平仓
            )

            realized = -realized
            unrealized = -unrealized  # 利润方向反转

        else:
            realized, unrealized, balance = self.calculate_profit(
                current_rate, buy_orders, sell_orders
            )
        self.realized_profit = realized
        self.balance = balance
        return realized + unrealized
