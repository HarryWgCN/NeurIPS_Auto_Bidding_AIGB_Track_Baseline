import os
import numpy as np
import pandas as pd
import pickle
import warnings

warnings.filterwarnings('ignore')


class TestDataLoader:
    """
    Offline evaluation data loader.
    """

    def __init__(self, file_path="/home/disk2/auto-bidding/data/log.csv"):
        """
        Initialize the data loader.
        Args:
            file_path (str): The path to the training data file.

        """
        # 用全部的阶段数据进行评估
        # self.file_paths = ["/home/disk2/auto-bidding/data/traffic/period-7.csv",
        #                    "/home/disk2/auto-bidding/data/traffic/period-8.csv",
        #                    "/home/disk2/auto-bidding/data/traffic/period-9.csv",
        #                    "/home/disk2/auto-bidding/data/traffic/period-10.csv",
        #                    "/home/disk2/auto-bidding/data/traffic/period-11.csv",
        #                    "/home/disk2/auto-bidding/data/traffic/period-12.csv",
        #                    "/home/disk2/auto-bidding/data/traffic/period-13.csv"]
        # self.raw_data_paths = ["/home/zhangyuxuan-23/raw_data_1.pickle",
        #                    "/home/zhangyuxuan-23/raw_data_2.pickle",
        #                    "/home/zhangyuxuan-23/raw_data_3.pickle",
        #                    "/home/zhangyuxuan-23/raw_data_4.pickle",
        #                    "/home/zhangyuxuan-23/raw_data_5.pickle",
        #                    "/home/zhangyuxuan-23/raw_data_6.pickle",
        #                    "/home/zhangyuxuan-23/raw_data_7.pickle"]
        # self.raw_data = self._get_raw_data_com()
        # self.keys, self.test_dict = self._get_test_data_dict()

        # 用单阶段数据进行评估
        self.file_path = file_path
        self.raw_data_path = os.path.join(os.path.dirname(file_path), "raw_data.pickle")
        self.raw_data = self._get_raw_data()
        self.keys, self.test_dict = self._get_test_data_dict()
    #检查是否存在pickle文件并返回数据
    def _get_raw_data(self):
        """
        Read raw data from a pickle file.

        Returns:
            pd.DataFrame: The raw data as a DataFrame.
        """
        if os.path.exists(self.raw_data_path):
            with open(self.raw_data_path, 'rb') as file:
                return pickle.load(file)
        else:
            tem = pd.read_csv(self.file_path)
            with open(self.raw_data_path, 'wb') as file:
                pickle.dump(tem, file)
            return tem
    #对数据进行分组和排序，返回键列表和字典
    def _get_test_data_dict(self):
        """
        Group and sort the raw data by deliveryPeriodIndex and advertiserNumber.

        Returns:
            list: A list of group keys.
            dict: A dictionary with grouped data.

        """
        # 对原始数据按照 timeStepIndex 进行排序，然后根据 deliveryPeriodIndex 和 advertiserNumber 进行分组
        grouped_data = self.raw_data.sort_values('timeStepIndex').groupby(['deliveryPeriodIndex', 'advertiserNumber'])
        # key:['deliveryPeriodIndex', 'advertiserNumber']
        data_dict = {key: group for key, group in grouped_data}
        return list(data_dict.keys()), data_dict
    #返回决策步数量、表示广告曝光给用户时的转化概率、无效变量、表示赢得当前展现机会的最低费用，即当前的竞价队列中的第4高的出价
    def mock_data(self, key):
        """
        Get training data based on deliveryPeriodIndex and advertiserNumber, and construct the test data.
        """
        #key表示一个周期内一个广告主的数据
        data = self.test_dict[key]
        #对数据按照'timeStepIndex'进行分组，然后分别提取对应列的数据，每个内部列表代表一个决策步索引的数据
        pValues = data.groupby('timeStepIndex')['pValue'].apply(list).apply(np.array).tolist()
        pValueSigmas = data.groupby('timeStepIndex')['pValueSigma'].apply(list).apply(np.array).tolist()
        leastWinningCosts = data.groupby('timeStepIndex')['leastWinningCost'].apply(list).apply(np.array).tolist()
        num_timeStepIndex = len(pValues) #决策步的数量
        return num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts

    def _get_raw_data_com(self):
        combined_data = pd.DataFrame()
        for file_path, raw_data_path in zip(self.file_paths, self.raw_data_paths):
            if os.path.exists(raw_data_path):
                with open(raw_data_path, 'rb') as file:
                    data = pickle.load(file)
            else:
                data = pd.read_csv(file_path)
                with open(raw_data_path, 'wb') as file:
                    pickle.dump(data, file)
            combined_data = pd.concat([combined_data, data])
        return combined_data

if __name__ == '__main__':
    pass
