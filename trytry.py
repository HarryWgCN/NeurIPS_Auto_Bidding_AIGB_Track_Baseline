import pandas as pd

dfs = list()
dfs.append(pd.read_csv('/home/disk2/auto-bidding/data/traffic_new_final/period-21.csv'))
print('read 1')
dfs.append(pd.read_csv('/home/disk2/auto-bidding/data/traffic_new_final/period-22.csv'))
print('read 2')
dfs.append(pd.read_csv('/home/disk2/auto-bidding/data/traffic_new_final/period-23.csv'))
print('read 3')
dfs.append(pd.read_csv('/home/disk2/auto-bidding/data/traffic_new_final/period-24.csv'))
print('read 4')
dfs.append(pd.read_csv('/home/disk2/auto-bidding/data/traffic_new_final/period-25.csv'))
print('read 5')

combined_dataframe = pd.concat(dfs, axis=0, ignore_index=True)
combined_dataframe_path = "/home/disk2/auto-bidding/data/test/5-traffic-test.csv"
combined_dataframe.to_csv(combined_dataframe_path, index=False)