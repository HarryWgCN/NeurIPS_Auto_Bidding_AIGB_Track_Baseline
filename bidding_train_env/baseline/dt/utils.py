import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
import numpy as np
import pickle
import random


class EpisodeReplayBuffer(Dataset):
    def __init__(self, state_dim, act_dim, data_path, max_ep_len=24, scale=2000, K=20):
        self.device = "cpu"
        super(EpisodeReplayBuffer, self).__init__()
        self.max_ep_len = max_ep_len
        self.scale = scale

        self.state_dim = state_dim
        self.act_dim = act_dim
        training_data = pd.read_csv(data_path)

        def safe_literal_eval(val):
            if pd.isna(val): # 检查val是否为缺失值（NaN），如果是，则直接返回val
                return val
            try:
                # ast.literal_eval是一个安全的方法，用于评估包含Python字面值的字符串，如字典、列表、元组、数字等。
                return ast.literal_eval(val) # 尝试将val转换为Python字面值
            except (ValueError, SyntaxError):
                print(ValueError)
                return val

        training_data["state"] = training_data["state"].apply(safe_literal_eval) # 当前决策步状态
        training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval) # 下一个决策步状态
        self.trajectories = training_data

        #记录所有广告商的对应数据：状态、奖励、动作、总奖励、决策步数、完成状态
        self.states, self.rewards, self.actions, self.returns, self.traj_lens, self.dones = [], [], [], [], [], []
        state = []
        reward = []
        action = []
        dones = []
        for index, row in self.trajectories.iterrows(): # 对行进行迭代，index表示行索引，row表示每一行的数据
            state.append(row["state"]) # 状态 [(),(),()]
            reward.append(row['reward']) # 奖励 [ , , ]
            action.append(row["action"]) # 动作 [ , , ]
            # 表示广告投放周期的完成状态，其中1表示是投放周期的最后一步或者广告主的剩余预算低于系统设定的最低阈值。
            dones.append(row["done"]) # [ , , ]
            if row["done"]: #投放结束
                if len(state) != 1:
                    self.states.append(np.array(state)) # [[ [][][] ],[ [][][] ],[ [][][] ]]
                    self.rewards.append(np.expand_dims(np.array(reward), axis=1)) # [ [], [], [] ]
                    self.actions.append(np.expand_dims(np.array(action), axis=1)) # [ [], [], [] ]
                    self.returns.append(sum(reward)) # 总奖励 [ , , ]
                    self.traj_lens.append(len(state)) # 最多48个决策步 [ , , ]
                    self.dones.append(np.array(dones)) # [ [], [], [] ]
                state = []
                reward = []
                action = []
                dones = []
        self.traj_lens, self.returns = np.array(self.traj_lens), np.array(self.returns)

        # 将self.states列表中的所有NumPy数组沿着axis=0轴（沿着行）进行连接
        tmp_states = np.concatenate(self.states, axis=0) # [ [] [] [] [] [] [] ]
        # 计算了tmp_states数组每列的均值  计算了tmp_states数组每列的标准差
        self.state_mean, self.state_std = np.mean(tmp_states, axis=0), np.std(tmp_states, axis=0) + 1e-6

        self.trajectories = []
        for i in range(len(self.states)):
            self.trajectories.append(
                {"observations": self.states[i], "actions": self.actions[i], "rewards": self.rewards[i],
                 "dones": self.dones[i]})

        self.K = K
        #采样概率的计算
        self.pct_traj = 1. #表示选择所有轨迹
        num_timesteps = sum(self.traj_lens) #总决策步数
        num_timesteps = max(int(self.pct_traj * num_timesteps), 1)
        sorted_inds = np.argsort(self.returns)  # lowest to highest 排序 返回一个按照 self.returns 数组中元素值的大小排序后的索引数组
        num_trajectories = 1 #初始决策步数
        timesteps = self.traj_lens[sorted_inds[-1]] #获取最后一个索引对应的 traj_lens 值作为起始值
        ind = len(self.trajectories) - 2 # 46
        # 从后往前计算：累积的决策步数超过了设定的 num_timesteps 或者遍历完所有轨迹
        while ind >= 0 and timesteps + self.traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += self.traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        self.sorted_inds = sorted_inds[-num_trajectories:]#选择最后的num_trajectories个轨迹，将其索引存储

        self.p_sample = self.traj_lens[self.sorted_inds] / sum(self.traj_lens[self.sorted_inds])

    def __getitem__(self, index):
        traj = self.trajectories[int(self.sorted_inds[index])] # 确定一个广告商
        start_t = random.randint(0, traj['rewards'].shape[0] - 1) # 选择一部分决策步
        #从0到traj['rewards'].shape[0] - 1之间随机选择一个整数作为start_t的取值

        s = traj['observations'][start_t: start_t + self.K] #提取从start_t到start_t + self.K（不包括start_t + self.K）的子序列
        a = traj['actions'][start_t: start_t + self.K]
        r = traj['rewards'][start_t: start_t + self.K].reshape(-1, 1)
        if 'terminals' in traj:
            d = traj['terminals'][start_t: start_t + self.K]
        else:
            d = traj['dones'][start_t: start_t + self.K]
        timesteps = np.arange(start_t, start_t + s.shape[0]) #生成一个从start_t开始，长度为s.shape[0]的整数序列
        # 将数组中大于或等于self.max_ep_len的值替换为self.max_ep_len - 1：self.max_ep_len=24
        timesteps[timesteps >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff 限制范围
        rtg = self.discount_cumsum(traj['rewards'][start_t:], gamma=1.)[:s.shape[0] + 1].reshape(-1, 1) # 折扣累计奖励
        if rtg.shape[0] <= s.shape[0]:
            rtg = np.concatenate([rtg, np.zeros((1, 1))], axis=0)

        tlen = s.shape[0]
        s = np.concatenate([np.zeros((self.K - tlen, self.state_dim)), s], axis=0)
        s = (s - self.state_mean) / self.state_std
        a = np.concatenate([np.ones((self.K - tlen, self.act_dim)) * -10., a], axis=0)
        r = np.concatenate([np.zeros((self.K - tlen, 1)), r], axis=0)
        r = r / self.scale
        d = np.concatenate([np.ones((self.K - tlen)) * 2, d], axis=0)
        rtg = np.concatenate([np.zeros((self.K - tlen, 1)), rtg], axis=0) / self.scale
        timesteps = np.concatenate([np.zeros((self.K - tlen)), timesteps], axis=0)
        mask = np.concatenate([np.zeros((self.K - tlen)), np.ones((tlen))], axis=0)

        s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(a).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(r).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(d).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(mask).to(device=self.device)
        return s, a, r, d, rtg, timesteps, mask # 状态、动作、奖励、完成状态、折扣累计奖励
    # 计算数组x的折扣累加和：gamma是折扣
    def discount_cumsum(self, x, gamma=1.):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

# def main():
#     # replay_buffer = EpisodeReplayBuffer(16, 1, "/home/disk2/auto-bidding/data/trajectory/trajectory_data.csv")
#     replay_buffer = EpisodeReplayBuffer(16, 1, "/home/zhangyuxuan-23/test_100.csv")
#     s, a, r, d, rtg, timesteps, mask=replay_buffer.__getitem__(1)
#     print('s:', s, 'a:', a, 'r:',r, 'd:',d, 'rtg:',rtg, 'timesteps:',timesteps, 'mask:',mask)
#     #20*16 20*1 20*1 20 20 20 20

# if __name__ == "__main__":
#     main()