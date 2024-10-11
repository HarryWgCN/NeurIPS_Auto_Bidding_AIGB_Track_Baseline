import os
import pandas as pd
import warnings
import glob

warnings.filterwarnings('ignore')


# code功能：将阶段数据生成轨迹数据，并合并所有的轨迹数据
class RlDataGenerator:
    """
    RL Data Generator for RL models.
    Reads raw data and constructs training data suitable for reinforcement learning.
    """

    def __init__(self, file_folder_path="/home/disk2/auto-bidding/data/traffic"):

        self.file_folder_path = file_folder_path  # /home/disk2/auto-bidding/data/traffic
        # /home/disk2/auto-bidding/data/traffic/training_data_rlData_folder
        self.training_data_path = self.file_folder_path + "/" + "training_data_rlData_folder"

    def batch_generate_rl_data(self):
        os.makedirs(self.training_data_path, exist_ok=True)  # 用于创建目录
        # 查找指定目录下所有以 .csv 结尾的文件，并将它们的文件路径存储在一个列表中
        csv_files = glob.glob(os.path.join(self.file_folder_path, '*.csv'))
        print(csv_files)
        training_data_list = []
        # 处理所有csv文件
        for csv_path in csv_files:
            print("开始处理文件：", csv_path)
            df = pd.read_csv(csv_path)
            df_processed = self._generate_rl_data(df)  # 将一个周期的数据生成轨迹数据
            csv_filename = os.path.basename(csv_path)
            trainData_filename = csv_filename.replace('.csv', '-rlData.csv')
            trainData_path = os.path.join(self.training_data_path, trainData_filename)
            df_processed.to_csv(trainData_path, index=False)  # 写入
            training_data_list.append(df_processed)
            del df, df_processed
            print("处理文件成功：", csv_path)
        combined_dataframe = pd.concat(training_data_list, axis=0, ignore_index=True)
        combined_dataframe_path = "/home/disk2/auto-bidding/data/training-data/training_data_all-rlData.csv"
        combined_dataframe.to_csv(combined_dataframe_path, index=False)
        print("整合多天训练数据成功；保存至:", combined_dataframe_path)

    def _generate_rl_data(self, df):  # csv文件
        """
        Construct a DataFrame in reinforcement learning format based on the raw data.

        Args:
            df (pd.DataFrame): The raw data DataFrame.

        Returns:
            pd.DataFrame: The constructed training data in reinforcement learning format.
        """

        training_data_rows = []
        # 根据以下内容进行分组：一个周期内的一个广告主在所有决策步和展现机会下的信息
        # 补充：一个周期内有50,000个展现机会，针对这些展现机会分为48个决策步，由48个agent进行出价
        # deliveryPeriodIndex: 表示当前投放周期的索引
        # advertiserNumber: 表示广告主的唯一标识符
        # advertiserCategoryIndex: 表示广告主的行业类别索引
        # budget: 表示广告主在一个投放周期内的预算
        # CPAConstraint: 表示广告主的CPA约束
        for (
                    deliveryPeriodIndex, advertiserNumber, advertiserCategoryIndex, budget,
                    CPAConstraint), group in df.groupby(
            ['deliveryPeriodIndex', 'advertiserNumber', 'advertiserCategoryIndex', 'budget', 'CPAConstraint']):

            # 对分组内按照'timeStepIndex'（表示当前决策步的索引）进行排序：1-48
            group = group.sort_values('timeStepIndex')

            # 对每个分组中的 'timeStepIndex' 列进行分组，
            # 并为每个行添加一个新的列 'timeStepIndex_volume'，该列包含了每个 'timeStepIndex' 在该分组中出现的次数（一个决策步的展现机会数量）
            # 'timeStepIndex'：1；'timeStepIndex_volume'：5
            group['timeStepIndex_volume'] = group.groupby('timeStepIndex')['timeStepIndex'].transform('size')
            # 对 timeStepIndex_volume 列进行分组，并取每个组的第一个值，存储在 timeStepIndex_volume_sum 变量中
            timeStepIndex_volume_sum = group.groupby('timeStepIndex')['timeStepIndex_volume'].first()
            # 一个周期内一个广告主的每个决策步的展现机会的统计

            # 计算 timeStepIndex_volume_sum 的累积和，并向后移动一位，然后填充缺失值，最后转换为整数型
            historical_volume = timeStepIndex_volume_sum.cumsum().shift(1).fillna(0).astype(int)
            # 通过 map 函数，将 historical_volume 中的值与 group['timeStepIndex'] 列的值进行匹配，
            # 然后将匹配到的历史累积值赋值给 'historical_volume' 列，为每个行添加了这个新的历史累积值的列。
            group['historical_volume'] = group['timeStepIndex'].map(historical_volume)  # 展现机会的累积统计（历史数据）

            # slot_1_win: 表示广告主在slot_1历史获胜次数。
            group['slot_1_win'] = group.apply(lambda row: 1 if row['adSlot'] == 1 else 1e-10, axis=1)
            group['slot_2_win'] = group.apply(lambda row: 1 if row['adSlot'] == 2 else 1e-10, axis=1)
            group['slot_3_win'] = group.apply(lambda row: 1 if row['adSlot'] == 3 else 1e-10, axis=1)
            # slot_1_exposed: 表示广告主在slot_1历史展现次数。
            group['slot_1_exposed'] = group.apply(lambda row: 1 if row['isExposed'] == 1 and row['adSlot'] == 1 else 0,
                                                  axis=1)
            group['slot_2_exposed'] = group.apply(lambda row: 1 if row['isExposed'] == 1 and row['adSlot'] == 2 else 0,
                                                  axis=1)
            group['slot_3_exposed'] = group.apply(lambda row: 1 if row['isExposed'] == 1 and row['adSlot'] == 3 else 0,
                                                  axis=1)
            # slot_1_win_least_alpha: 表示广告主赢得slot1 的最小alpha action
            group['slot_1_win_least_alpha'] = group.apply(
                lambda row: row['bid'] / row['pValue'] if row['adSlot'] == 1 else 2, axis=1)
            group['slot_2_win_least_alpha'] = group.apply(
                lambda row: row['bid'] / row['pValue'] if row['adSlot'] == 2 else 2, axis=1)
            group['slot_3_win_least_alpha'] = group.apply(
                lambda row: row['bid'] / row['pValue'] if row['adSlot'] == 3 else 2, axis=1)

            # 对 timeStepIndex_volume_sum 进行滚动计算，计算最近三个值的和，然后向后移动一位，填充缺失值，并转换为整数型
            last_3_timeStepIndexs_volume = timeStepIndex_volume_sum.rolling(window=3, min_periods=1).sum().shift(
                1).fillna(0).astype(int)
            group['last_3_timeStepIndexs_volume'] = group['timeStepIndex'].map(last_3_timeStepIndexs_volume)

            # pValue: 表示广告曝光给用户时的转化概率。
            # conversionAction: 表示是否发生转化，其中1表示发生转化，0表示未发生。
            # leastWinningCost: 表示赢得当前展现机会的最低费用，即当前的竞价队列中的第4高的出价。
            # xi: 表示广告主在展现机会中的获胜状态，其中1表示获胜，0表示未获胜。
            # bid: 表示出价agent对当前展现机会的出价。
            # timeStepIndex_volume：表示当前决策步的索引。
            # slot_1_win: 表示赢得slot1
            # slot_1_exposed: 表示slot1被曝光
            group_agg = group.groupby('timeStepIndex').agg({
                'bid': 'mean',
                'leastWinningCost': 'mean',
                'conversionAction': 'mean',
                'xi': 'mean',
                'pValue': 'mean',
                'timeStepIndex_volume': 'first',
                'slot_1_win': 'sum',
                'slot_1_exposed': 'sum',
                'slot_2_win': 'sum',
                'slot_2_exposed': 'sum',
                'slot_3_win': 'sum',
                'slot_3_exposed': 'sum',
                'slot_1_win_least_alpha': 'min',
                'slot_2_win_least_alpha': 'min',
                'slot_3_win_least_alpha': 'min'
            }).reset_index()  # 将结果中的索引列重置为默认的整数索引，将之前的分组键列 'timeStepIndex' 变成一个普通的列

            # 计算了这些列的扩展均值和最近三个值的滚动平均值
            for col in ['bid', 'leastWinningCost', 'conversionAction', 'xi', 'pValue']:
                group_agg[f'avg_{col}_all'] = group_agg[col].expanding().mean().shift(1)
                group_agg[f'avg_{col}_last_3'] = group_agg[col].rolling(window=3, min_periods=1).mean().shift(1)
            for col in ['slot_1_win', 'slot_1_exposed', 'slot_2_win', 'slot_2_exposed', 'slot_3_win', 'slot_3_exposed']:
                group_agg[f'sum_{col}_all'] = group_agg[col].expanding().sum().shift(1)
                group_agg[f'sum_{col}_last_3'] = group_agg[col].rolling(window=3, min_periods=1).sum().shift(1)
            for col in ['slot_1_win_least_alpha', 'slot_2_win_least_alpha', 'slot_3_win_least_alpha']:
                group_agg[f'min_{col}_all'] = group_agg[col].expanding().min().shift(1)
                group_agg[f'min_{col}_last_3'] = group_agg[col].rolling(window=3, min_periods=1).min().shift(1)

            # 将 group 和 group_agg 按照 'timeStepIndex' 列进行合并
            # suffixes=('', '_agg') 参数指定了如果存在重叠的列名时如何添加后缀以区分这些列
            group = group.merge(group_agg, on='timeStepIndex', suffixes=('', '_agg'))
            # 计算 realCost 和 realConversion
            # cost: 表示如果广告曝光给用户，广告主需要支付的费用
            # isExposed: 表示广告坑位是否被曝光，其中1表示广告被曝光，0表示未曝光
            realAllCost = (group['isExposed'] * group['cost']).sum()
            realAllConversion = group['conversionAction'].sum()

            # 一个周期一个广告主在一个决策步中的所有出价
            for timeStepIndex in group['timeStepIndex'].unique():
                current_timeStepIndex_data = group[group['timeStepIndex'] == timeStepIndex]

                timeStepIndexNum = 48
                current_timeStepIndex_data['sum_slot_1_win_all'].fillna(1e-10, inplace=True)
                current_timeStepIndex_data['sum_slot_1_exposed_all'].fillna(1e-10, inplace=True)
                current_timeStepIndex_data['sum_slot_2_win_all'].fillna(5e-11, inplace=True)
                current_timeStepIndex_data['sum_slot_2_exposed_all'].fillna(1e-10, inplace=True)
                current_timeStepIndex_data['sum_slot_3_win_all'].fillna(2e-11, inplace=True)
                current_timeStepIndex_data['sum_slot_3_exposed_all'].fillna(1e-10, inplace=True)
                current_timeStepIndex_data.fillna(0, inplace=True)  # 缺失值用 0 填充

                budget = current_timeStepIndex_data['budget'].iloc[0]  # 获取 'budget' 列的第一个值
                remainingBudget = current_timeStepIndex_data['remainingBudget'].iloc[0]

                timeleft = (timeStepIndexNum - timeStepIndex) / timeStepIndexNum
                bgtleft = remainingBudget / budget if budget > 0 else 0

                # 当前时间步的第一行数据转换为字典形式
                state_features = current_timeStepIndex_data.iloc[0].to_dict()

                state = (
                    timeleft, bgtleft,
                    state_features['avg_bid_all'],
                    state_features['avg_bid_last_3'],
                    state_features['avg_leastWinningCost_all'],
                    state_features['avg_pValue_all'],
                    state_features['avg_conversionAction_all'],
                    state_features['avg_xi_all'],
                    state_features['avg_leastWinningCost_last_3'],
                    state_features['avg_pValue_last_3'],
                    state_features['avg_conversionAction_last_3'],
                    state_features['avg_xi_last_3'],
                    state_features['pValue_agg'],  # mean
                    state_features['timeStepIndex_volume_agg'],  # 决策步索引
                    state_features['last_3_timeStepIndexs_volume'],  # 展现机会的滚动统计
                    state_features['historical_volume'],  # 展现机会的累积统计
                    state_features['sum_slot_1_exposed_all'] / state_features['sum_slot_1_win_all'],  # 坑位1的展现概率
                    state_features['sum_slot_2_exposed_all'] / state_features['sum_slot_2_win_all'],  # 坑位2的展现概率
                    state_features['sum_slot_3_exposed_all'] / state_features['sum_slot_3_win_all'],  # 坑位3的展现概率
                    state_features['min_slot_1_win_least_alpha_all'],  # 坑位1的最低获胜alpha
                    state_features['min_slot_2_win_least_alpha_all'],  # 坑位2的最低获胜alpha
                    state_features['min_slot_3_win_least_alpha_all'],  # 坑位3的最低获胜alpha
                )

                total_bid = current_timeStepIndex_data['bid'].sum()  # 一个决策步的出价总和
                total_value = current_timeStepIndex_data['pValue'].sum()  # 一个决策步的转化概率总和（曝光/未曝光）
                action = total_bid / total_value if total_value > 0 else 0  # 动作计算

                reward = current_timeStepIndex_data[current_timeStepIndex_data['isExposed'] == 1][
                    'conversionAction'].sum()  # 被曝光的转化次数的总和
                reward_continuous = current_timeStepIndex_data[current_timeStepIndex_data['isExposed'] == 1][
                    'pValue'].sum()  # 被曝光的转化概率的总和

                # isEnd: 表示广告投放周期的完成状态，其中1表示是投放周期的最后一步或者广告主的剩余预算低于系统设定的最低阈值。
                done = 1 if timeStepIndex == timeStepIndexNum - 1 or current_timeStepIndex_data['isEnd'].iloc[
                    0] == 1 else 0

                # 轨迹数据
                training_data_rows.append({
                    'deliveryPeriodIndex': deliveryPeriodIndex,  # 表示当前投放周期的索引
                    'advertiserNumber': advertiserNumber,  # 表示广告主的唯一标识符
                    'advertiserCategoryIndex': advertiserCategoryIndex,  # 表示广告主的行业类别索引
                    'budget': budget,  # 表示广告主在一个投放周期内的预算
                    'CPAConstraint': CPAConstraint,  # 表示广告主的CPA约束。
                    'realAllCost': realAllCost,  # 表示广告主在该投放周期下的消耗总数
                    'realAllConversion': realAllConversion,  # 表示广告主在该投放周期下的转化总数
                    'timeStepIndex': timeStepIndex,  # 表示当前决策步的索引
                    'state': state,  # 当前决策步状态。
                    'action': action,  # 当前决策步动作。
                    'reward': reward,  # 当前决策步稀疏奖励（当前决策步转化之和）
                    'reward_continuous': reward_continuous,  # 当前决策步连续奖励（当前决策步所有展示流量的pValue之和）
                    'done': done  # 表示广告投放周期的完成状态，其中1表示是投放周期的最后一步或者广告主的剩余预算低于系统设定的最低阈值。
                })

        training_data = pd.DataFrame(training_data_rows)
        training_data = training_data.sort_values(by=['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex'])

        training_data['next_state'] = training_data.groupby(['deliveryPeriodIndex', 'advertiserNumber'])['state'].shift(
            -1)
        training_data.loc[training_data['done'] == 1, 'next_state'] = None
        return training_data


def generate_rl_data():
    # TODO change to full set
    file_folder_path = "/home/disk2/auto-bidding/data/traffic_new_final"
    # file_folder_path = "/home/disk2/auto-bidding/data/truncated"
    data_loader = RlDataGenerator(file_folder_path=file_folder_path)
    data_loader.batch_generate_rl_data()


if __name__ == '__main__':
    generate_rl_data()
