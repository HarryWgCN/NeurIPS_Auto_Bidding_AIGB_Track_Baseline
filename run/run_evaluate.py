import numpy as np
import math
import logging
from bidding_train_env.strategy import PlayerBiddingStrategy
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def run_test(i):
    """
    offline evaluation
    """

    data_loader = TestDataLoader(file_path='./data/traffic/period-7.csv')
    env = OfflineEnv()
    agent = PlayerBiddingStrategy(i=i)
    print(agent.name,agent.budget)

    keys, test_dict = data_loader.keys, data_loader.test_dict
    key = keys[0]
    num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(key)
    rewards = np.zeros(num_timeStepIndex)
    history = {
        'historyBids': [],
        'historyAuctionResult': [],
        'historyImpressionResult': [],
        'historyLeastWinningCost': [],
        'historyPValueInfo': []
    }

    for timeStep_index in range(num_timeStepIndex): #循环每个决策步
        # logger.info(f'Timestep Index: {timeStep_index + 1} Begin')

        pValue = pValues[timeStep_index]#表示广告曝光给用户时的转化概率
        pValueSigma = pValueSigmas[timeStep_index]
        leastWinningCost = leastWinningCosts[timeStep_index]

        if agent.remaining_budget < env.min_remaining_budget:
            bid = np.zeros(pValue.shape[0])#此时出价为0
        else:
            #出价
            bid = agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                history["historyBids"],
                                history["historyAuctionResult"], history["historyImpressionResult"],
                                history["historyLeastWinningCost"])

        tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                      leastWinningCost)
        print("bid",bid.shape)
        # Handling over-cost (a timestep costs more than the remaining budget of the bidding advertiser)
        over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
        #循环保证不会超预算
        while over_cost_ratio > 0:#超过预算
            print('Exceeding Budget Constraint')
            pv_index = np.where(tick_status == 1)[0]  #找到赢得展现机会的索引
            #选取一部分索引
            dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                replace=False)
            #对应索引的出价置为0
            bid[dropped_pv_index] = 0
            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                          leastWinningCost)
            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

        agent.remaining_budget -= np.sum(tick_cost)
        rewards[timeStep_index] = np.sum(tick_conversion)
        temHistoryPValueInfo = [(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]
        history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
        history["historyBids"].append(bid)
        history["historyLeastWinningCost"].append(leastWinningCost)
        temAuctionResult = np.array(
            [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
        history["historyAuctionResult"].append(temAuctionResult)
        temImpressionResult = np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])])
        history["historyImpressionResult"].append(temImpressionResult)
        logger.info(f'Timestep Index: {timeStep_index + 1} End')
    all_reward = np.sum(rewards)
    all_cost = agent.budget - agent.remaining_budget
    cpa_real = all_cost / (all_reward + 1e-10)
    cpa_constraint = agent.cpa
    print(agent.budget , agent.remaining_budget,cpa_constraint,all_reward)
    score = getScore_nips(all_reward, cpa_real, cpa_constraint)

    print("模型索引------:",i,"------END")
    logger.info(f'Total Reward: {all_reward}')
    logger.info(f'Total Cost: {all_cost}')
    logger.info(f'CPA-real: {cpa_real}')
    logger.info(f'CPA-constraint: {cpa_constraint}')
    logger.info(f'Score: {score}')
    return i,all_reward,all_cost,cpa_real,cpa_constraint,score


if __name__ == '__main__':
    # for i in range(0,100):
        run_test(23)
