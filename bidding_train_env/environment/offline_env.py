import numpy as np


class OfflineEnv:
    """
    Simulate an advertising bidding environment.
    """

    def __init__(self, min_remaining_budget: float = 0.1):
        """
        Initialize the simulation environment.
        :param min_remaining_budget: The minimum remaining budget allowed for bidding advertiser.
        """
        #用于设置允许竞价广告商保留的最低剩余预算。
        self.min_remaining_budget = min_remaining_budget

    def simulate_ad_bidding(self, pValues: np.ndarray,pValueSigmas: np.ndarray, bids: np.ndarray, leastWinningCosts: np.ndarray, bids_other: list):
        """
        Simulate the advertising bidding process.

        :param pValues: Values of each pv .
        :param pValueSigmas: uncertainty of each pv .
        :param bids: Bids from the bidding advertiser.
        :param leastWinningCosts: Market prices for each pv.
        :return: Win values, costs spent, and winning status for each bid.

        """
        
        # ----------------------------------- stimulate agents -------------------------------------
        
        bids_other['player'] = bids
        bids_matrix = np.array(list(bids_other.values()))
        bids_sorted = np.sort(bids_matrix, axis=0)
        leastWinningCosts = bids_sorted[-4, :]
        
        bids_num = len(bids_other)
        
        tick_status = {}
        tick_cost = {}
        tick_value = {}
        tick_conversion = {}
        
        values = np.random.normal(loc=pValues, scale=pValueSigmas)
        
        for k, v in bids_other:
            tick_status[k] = v >= leastWinningCosts
            tick_cost[k] = leastWinningCosts * tick_status[k]
            _values = values*tick_status[k]
            tick_value[k] = np.clip(_values,0,1)
            tick_conversion[k] = np.random.binomial(n=1, p=tick_value[k])
        
        return tick_value, tick_cost, tick_status,tick_conversion, leastWinningCosts
        # ----------------------------------- stimulate agents -------------------------------------
        
        #接受四个变量：表示广告曝光给用户时的转化概率、无效变量、表示出价、表示赢得当前展现机会的最低费用，即当前的竞价队列中的第4高的出价
        # tick_status = bids >= leastWinningCosts #确定是否赢得展现机会
        # tick_cost = leastWinningCosts * tick_status #赢得的最低成本
        #  #以pValues为均值、以0为标准差的正态分布中生成一个随机值
        # values = values*tick_status
        # tick_value = np.clip(values,0,1) #将values限制在0-1之间,将数组中小于 0 的元素设为 0，大于 1 的元素设为 1
        # tick_conversion = np.random.binomial(n=1, p=tick_value) #生成一个符合二项分布的随机数、n表示实验次数、p表示成功概率

        # return tick_value, tick_cost, tick_status,tick_conversion
        

        
        



def test():
    pv_values = np.array([10, 20, 30, 40, 50])
    pv_values_sigma = np.array([1, 2, 3, 4, 5])
    bids = np.array([15, 20, 35, 45, 55])
    market_prices = np.array([12, 22, 32, 42, 52])

    env = OfflineEnv()
    tick_value, tick_cost, tick_status,tick_conversion = env.simulate_ad_bidding(pv_values, bids, market_prices)

    print(f"Tick Value: {tick_value}")
    print(f"Tick Cost: {tick_cost}")
    print(f"Tick Status: {tick_status}")


if __name__ == '__main__':
    test()
