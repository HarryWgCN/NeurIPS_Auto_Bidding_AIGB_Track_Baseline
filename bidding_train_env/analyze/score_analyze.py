import pandas as pd

path='/home/weihanxiao-23/autobidding1/data/trajectory/trajectory_score.csv'
df = pd.read_csv(path)

df2=df.groupby('advertiserNumber')['score'].mean().reset_index()
df3=df.groupby('advertiserNumber')['budget'].mean().reset_index()
df_merged = pd.merge(df2, df3, on='advertiserNumber')
df_merged.to_csv('/home/weihanxiao-23/autobidding1/data/trajectory/trajectory_score_by_agent.csv',index=False)