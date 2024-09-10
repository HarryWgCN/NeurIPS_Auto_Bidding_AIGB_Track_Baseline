import numpy as np
import pandas as pd
import math
import logging
from bidding_train_env.strategy import PlayerBiddingStrategy 
from bidding_train_env.strategy import DtBiddingStrategy
from bidding_train_env.strategy import CPABiddingStrategy


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

    data_loader = TestDataLoader(file_path='/home/disk2/auto-bidding/data/traffic/period-7.csv')
    env = OfflineEnv()
    agent = PlayerBiddingStrategy(i)
    print(agent.name)
 # ----------------------------------------- fake agent init ----------------------------------------
    fake_agent_num = 3
    dd_path = '/home/disk2/guoyuning-23/NeurIPS_Auto_Bidding_AIGB_Track_Baseline/saved_model/DDtest/diffuser.pt'
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
    for index, row in random_sample.iterrows():
        budget_cpa_sampled.append([row['budget'], int(row['CPAConstraint'])])
    # 实例化3种agent各fake_agent_num个，分配不同CPA和budget
    for i in range(0, fake_agent_num):
        cpa_agent = CPABiddingStrategy(budget=budget_cpa_sampled[sample_i][0], cpa=budget_cpa_sampled[sample_i][1])
        sample_i += 1
        cpa_agents.append(cpa_agent)
        
        dd_agent = PlayerBiddingStrategy(budget=budget_cpa_sampled[sample_i][0], cpa=budget_cpa_sampled[sample_i][1], base_model_path=dd_path)
        sample_i += 1
        dd_agents.append(dd_agent)
        
        dt_agent = DtBiddingStrategy(budget=budget_cpa_sampled[sample_i][0], cpa=budget_cpa_sampled[sample_i][1], base_model_path=dt_path, base_pkl_path=pkl_path)
        sample_i += 1
        dt_agents.append(dt_agent)
#------------------------------------- fake agent init ---------------------------------------------- 
    
    keys, test_dict = data_loader.keys, data_loader.test_dict
    key = keys[0]
    num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(key)
    print(leastWinningCosts)
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
# ------------------------------- fake agent bidding -------------------------------

        bids_cpa = []
        bids_dt = []
        bids_dd = []
        bids = {}
        # bids = []
        for i in range(0, fake_agent_num):
            if cpa_agents[i].remaining_budget < env.min_remaining_budget:
                bid_cpa = np.zeros(pValue.shape[0])
            else :
                bid_cpa = cpa_agents[i].bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                history["historyBids"],
                                history["historyAuctionResult"], history["historyImpressionResult"],
                                history["historyLeastWinningCost"])
            bids[f'cpa_{i}'] = bid_cpa
            # bids.append(bid_cpa)
            
            if dd_agents[i].remaining_budget < env.min_remaining_budget:
                bid_dd = np.zeros(pValue.shape[0])
            else :
                bid_dd = dd_agents[i].bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                history["historyBids"],
                                history["historyAuctionResult"], history["historyImpressionResult"],
                                history["historyLeastWinningCost"])
            bids[f'dd_{i}'] = bid_dd
            # bids.append(bid_dd)
            
            if dt_agents[i].remaining_budget < env.min_remaining_budget:
                bid_dt = np.zeros(pValue.shape[0])
            else :
                bid_dt = dt_agents[i].bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                history["historyBids"],
                                history["historyAuctionResult"], history["historyImpressionResult"],
                                history["historyLeastWinningCost"])
            bids[f'dt_{i}'] = bid_dt
            # bids.append(bid_dt)
            

# ----------------------------------------- fake agent bidding ----------------------------------------

        if agent.remaining_budget < env.min_remaining_budget:
            bid = np.zeros(pValue.shape[0])#此时出价为0
        else:
            #出价
            bid = agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                history["historyBids"],
                                history["historyAuctionResult"], history["historyImpressionResult"],
                                history["historyLeastWinningCost"])
        # 返回字典，包括player和fake agents
        tick_value, tick_cost, tick_status, tick_conversion, leastWinningCost = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                      leastWinningCost)
        # -------------------------------------------- all agents to be caculated -------------------------------
        for agent_name, _ in bids:
            # 若非player agent，则为fake agent，通过取出key的最后一位进行agent赋值
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
            over_cost_ratio[agent_name] = max((np.sum(tick_cost[agent_name]) - tmp_agent.remaining_budget) / (np.sum(tick_cost[agent_name]) + 1e-4), 0)
            
            #循环保证不会超预算
            while over_cost_ratio > 0:#超过预算
                print('Exceeding Budget Constraint on '+agent_name)
                pv_index = np.where(tick_status[agent_name] == 1)[0]  #找到赢得展现机会的索引
                #选取一部分索引
                dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                    replace=False)
                #对应索引的出价置为0
                bids[agent_name][dropped_pv_index] = 0
                tick_value, tick_cost, tick_status, tick_conversion, leastWinningCost = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                            leastWinningCost, bids)
                over_cost_ratio = max((np.sum(tick_cost[agent_name]) - tmp_agent.remaining_budget) / (np.sum(tick_cost[agent_name]) + 1e-4), 0)

            tmp_agent.remaining_budget -= np.sum(tick_cost[agent_name])
        # -------------------------------------------- all agents to be caculated -------------------------------

        # ---------------------------------------------- origin code --------------------------------------
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
            
        # 以下代码针对player agent进行，因为字典的关系，所有的stimulate输出都加入key:"player"
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
        logger.info(f'Timestep Index: {timeStep_index + 1} End')
    all_reward = np.sum(rewards)
    all_cost = agent.budget - agent.remaining_budget
    cpa_real = all_cost / (all_reward + 1e-10)
    cpa_constraint = agent.cpa
    score = getScore_nips(all_reward, cpa_real, cpa_constraint)

    print("模型索引------:",i,"------END")
    logger.info(f'Total Reward: {all_reward}')
    logger.info(f'Total Cost: {all_cost}')
    logger.info(f'CPA-real: {cpa_real}')
    logger.info(f'CPA-constraint: {cpa_constraint}')
    logger.info(f'Score: {score}')
    return i,all_reward,all_cost,cpa_real,cpa_constraint,score


if __name__ == '__main__':
    for i in range(0,100):
        run_test(i)
