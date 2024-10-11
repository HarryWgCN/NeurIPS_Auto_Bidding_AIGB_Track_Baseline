import numpy as np
import pandas as pd
import math
import logging
import time
from bidding_train_env.strategy import PlayerBiddingStrategy 
from bidding_train_env.strategy import DtBiddingStrategy
from bidding_train_env.strategy import CPABiddingStrategy
from bidding_train_env.strategy import base_dd_bidding_strategy
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

data_loader = TestDataLoader(file_path='/home/disk2/auto-bidding/data/traffic_new_final/period-24.csv')
env = OfflineEnv()

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
    score_all = 0
    reward_all = 0
    cost_all = 0
    dd_path = '/home/disk2/guoyuning-23/NeurIPS_Auto_Bidding_AIGB_Track_Baseline/saved_model/DDtest/diffuser.pt'
    # 传入模型编号
    agent = PlayerBiddingStrategy()
    print(agent.name)
 # ----------------------------------------- stimulate agent init ↓----------------------------------------
    # 每类模拟agent的个数    
    stimulate_agent_num = 3
    # baseline 模型以及budget samples的路径
    dt_path = '/home/disk2/guoyuning-23/NeurIPS_Auto_Bidding_AIGB_Track_Baseline/saved_model/DTtest/dt.pt'
    pkl_path = '/home/disk2/guoyuning-23/NeurIPS_Auto_Bidding_AIGB_Track_Baseline/saved_model/DTtest/normalize_dict.pkl'
    samples_path = '/home/disk2/guoyuning-23/auto-bidding/NeurIPS_Auto_Bidding_AIGB_Track_Baseline/budget_cpa_samples.csv'
    dd_agents = []
    dt_agents = []
    cpa_agents = []
    budget_cpa_sampled = []
    sample_i = 0
    
    # 采样CPA和budget
    df = pd.read_csv(samples_path)
    random_sample = df.sample(n=9)
    print(random_sample)
    for _, row in random_sample.iterrows():
        budget_cpa_sampled.append([row['budget'], int(row['CPAConstraint'])])

    start = time.time()
    # 实例化3种agent各stimulate_agent_num个，分配不同CPA和budget
    for _ in range(0, stimulate_agent_num):
        cpa_agent = CPABiddingStrategy(budget=budget_cpa_sampled[sample_i][0], cpa=budget_cpa_sampled[sample_i][1])
        sample_i += 1
        cpa_agents.append(cpa_agent)
        
        dd_agent = base_dd_bidding_strategy(budget=budget_cpa_sampled[sample_i][0], cpa=budget_cpa_sampled[sample_i][1])
        sample_i += 1
        dd_agents.append(dd_agent)
        
        dt_agent = DtBiddingStrategy(i = 1, budget=budget_cpa_sampled[sample_i][0], cpa=budget_cpa_sampled[sample_i][1], base_model_path=dt_path, base_pkl_path=pkl_path)
        sample_i += 1
        dt_agents.append(dt_agent)
    print(f'agents initialized using {time.time() - start} seconds')
#------------------------------------- stimulate agent init ↑---------------------------------------------- 
    keys, test_dict = data_loader.keys, data_loader.test_dict
    delivery_period_keys = list(set([element[0] for element in keys]))
    for key in delivery_period_keys:
        key = (key, 0)
        num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(key)

        rewards = np.zeros(num_timeStepIndex)
        history = {
            'historyBids': [],
            'historyAuctionResult': [],
            'historyImpressionResult': [],
            'historyLeastWinningCost': [],
            'historyPValueInfo': []
        }
        stimulate_history = {}

        # 初始化嵌套字典用来保存模拟agent的历史数据：
        for i in range(0, stimulate_agent_num):
            stimulate_history[f'cpa_{i}'] = {
                'historyBids': [],
                'historyAuctionResult': [],
                'historyImpressionResult': [],
            }

            stimulate_history[f'dd_{i}'] = {
                'historyBids': [],
                'historyAuctionResult': [],
                'historyImpressionResult': [],
            }

            stimulate_history[f'dt_{i}'] = {
                'historyBids': [],
                'historyAuctionResult': [],
                'historyImpressionResult': [],
            }

        for timeStep_index in range(num_timeStepIndex): #循环每个决策步
            # logger.info(f'Timestep Index: {timeStep_index + 1} Begin')

            pValue = pValues[timeStep_index]#表示广告曝光给用户时的转化概率
            pValueSigma = pValueSigmas[timeStep_index]

    # ------------------------------- stimulate agent bidding ↓-------------------------------
            bids = {}

            for i in range(0, stimulate_agent_num):
                if cpa_agents[i].remaining_budget < env.min_remaining_budget:
                    bid_cpa = np.zeros(pValue.shape[0])
                else:
                    bid_cpa = cpa_agents[i].bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                    stimulate_history[f'cpa_{i}']["historyBids"],
                                    stimulate_history[f'cpa_{i}']["historyAuctionResult"], stimulate_history[f'cpa_{i}']["historyImpressionResult"],
                                    history["historyLeastWinningCost"])
                bids[f'cpa_{i}'] = bid_cpa

                if dd_agents[i].remaining_budget < env.min_remaining_budget:
                    bid_dd = np.zeros(pValue.shape[0])
                else :
                    bid_dd = dd_agents[i].bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                    stimulate_history[f'dd_{i}']["historyBids"],
                                    stimulate_history[f'dd_{i}']["historyAuctionResult"], stimulate_history[f'dd_{i}']["historyImpressionResult"],
                                    history["historyLeastWinningCost"])
                bids[f'dd_{i}'] = bid_dd

                if dt_agents[i].remaining_budget < env.min_remaining_budget:
                    bid_dt = np.zeros(pValue.shape[0])
                else :
                    bid_dt = dt_agents[i].bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                    stimulate_history[f'dt_{i}']["historyBids"],
                                    stimulate_history[f'dt_{i}']["historyAuctionResult"], stimulate_history[f'dt_{i}']["historyImpressionResult"],
                                    history["historyLeastWinningCost"])
                bids[f'dt_{i}'] = bid_dt
            print(f'agents bid in timestemp {timeStep_index} using {time.time() - start} seconds')
    # ----------------------------------------- stimulate agent bidding ↑----------------------------------------

            if agent.remaining_budget < env.min_remaining_budget:
                bid = np.zeros(pValue.shape[0])#此时出价为0
            else:
                #出价
                bid = agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                    history["historyBids"],
                                    history["historyAuctionResult"], history["historyImpressionResult"],
                                    history["historyLeastWinningCost"])

            # 构成一共stimulate_agent_num + 1个元素的出价字典
            bids['player'] = bid

            # 返回字典，包括player和stimulate agents
            tick_value, tick_cost, tick_status, tick_conversion, leastWinningCost = env.simulate_ad_bidding(pValue, pValueSigma,
                                                                                           bids)
    # -------------------------------------------- all agents to be caculated ↓-------------------------------
            for agent_name, _ in bids.items():
                # 若非player agent，则为stimulate agent，通过取出key的最后一位进行agent赋值
                if agent_name == 'player':
                    tmp_agent = agent
                elif 'cpa' in agent_name:
                    tmp_agent = cpa_agents[int(agent_name[-1])]
                elif 'dd' in agent_name:
                    tmp_agent = dd_agents[int(agent_name[-1])]
                elif 'dt' in agent_name:
                    tmp_agent = dt_agents[int(agent_name[-1])]
                else:
                    print('no such agent!')

                # Handling over-cost (a timestep costs more than the remaining budget of the bidding advertiser)
                over_cost_ratio = max((np.sum(tick_cost[agent_name]) - tmp_agent.remaining_budget) / (np.sum(tick_cost[agent_name]) + 1e-4), 0)

                #循环保证不会超预算
                while over_cost_ratio > 0:#超过预算
                    print('Exceeding Budget Constraint on '+agent_name)
                    pv_index = np.where(tick_status[agent_name] == 1)[0]  #找到赢得展现机会的索引
                    #选取一部分索引
                    dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                        replace=False)
                    #对应索引的出价置为0
                    bids[agent_name][dropped_pv_index] = 0
                    tick_value, tick_cost, tick_status, tick_conversion, leastWinningCost = env.simulate_ad_bidding(pValue, pValueSigma,
                                                                                                bids)
                    over_cost_ratio = max((np.sum(tick_cost[agent_name]) - tmp_agent.remaining_budget) / (np.sum(tick_cost[agent_name]) + 1e-4), 0)

                tmp_agent.remaining_budget -= np.sum(tick_cost[agent_name])
            # -------------------------------------------- all agents to be caculated ↑-------------------------------

            # ---------------------------------------------- original code ----------------------------------------
            # over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
            # while over_cost_ratio > 0:
            #     pv_index = np.where(tick_status == 1)[0]
            #     dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
            #                                         replace=False)
            #     bid[dropped_pv_index] = 0
            #     tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
            #                                                                                   leastWinningCost)
            #     over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

            # agent.remaining_budget -= np.sum(tick_cost)
            # ----------------------------------- deal with history of stimulate agents ----------------------------------
            for agent_name, bid in bids.items():
                if agent_name != 'player':
                    stimulate_history[agent_name]['historyBids'].append(bid)
                    temAuctionResult = np.array(
                [(tick_status[agent_name][i], tick_status[agent_name][i], tick_cost[agent_name][i]) for i in range(tick_status[agent_name].shape[0])])
                    stimulate_history[agent_name]['historyAuctionResult'].append(temAuctionResult)
                    temImpressionResult = np.array([(tick_conversion[agent_name][i], tick_conversion[agent_name][i]) for i in range(pValue.shape[0])])
                    stimulate_history[agent_name]['historyImpressionResult'].append(temImpressionResult)
            # ----------------------------------- deal with history of stimulate agents ----------------------------------
            # print(f"-------------------------least winning cost :{leastWinningCost}-----------------------------------")
            # 以下代码针对player agent进行，因为字典的关系，所有的stimulate输出都加入key:"player"”
            rewards[timeStep_index] = np.sum(tick_conversion['player'])
            temHistoryPValueInfo = [(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]
            history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
            history["historyBids"].append(bid)
            history["historyLeastWinningCost"].append(leastWinningCost)
            temAuctionResult = np.array(
                [(tick_status['player'][i], tick_status['player'][i], tick_cost['player'][i]) for i in range(tick_status['player'].shape[0])])
            history["historyAuctionResult"].append(temAuctionResult)
            temImpressionResult = np.array([(tick_conversion['player'][i], tick_conversion['player'][i]) for i in range(pValue.shape[0])])
            history["historyImpressionResult"].append(temImpressionResult)
        logger.info(f'Delivery Period Index: {key} End')
        all_reward = np.sum(rewards)

        all_cost = agent.budget - agent.remaining_budget
        cpa_real = all_cost / (all_reward + 1e-10)
        cpa_constraint = agent.cpa
        score = getScore_nips(all_reward, cpa_real, cpa_constraint)
        score_all += score
        reward_all += all_reward
        cost_all += all_cost


    print("模型索引------:",i,"------END")
    logger.info(f'Score: {score_all / len(keys)}')
    logger.info(f'Reward: {reward_all / len(keys)}')
    logger.info(f'Cost: {cost_all / len(keys)}')
    # return i,all_reward,all_cost,cpa_real,cpa_constraint,score