import pandas as pd

df = pd.read_csv('/home/disk2/auto-bidding/data/trajectory/trajectory_data.csv')

print(len(df.index))

df = df[:10000]

df.to_csv('/home/disk2/auto-bidding/data/trajectory/trajectory_data_truncated.csv')
