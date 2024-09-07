import pandas as pd

def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


csv_path='/home/weihanxiao-23/autobidding1/data/trajectory/trajectory_data.csv'
df = pd.read_csv(csv_path)
# print(df)
df_info=pd.DataFrame(df, columns=['deliveryPeriodIndex','advertiserNumber','budget','realAllCost','CPAConstraint','reward','done'])[df.done == 1]
print(df_info[['deliveryPeriodIndex','advertiserNumber','budget','realAllCost','CPAConstraint','reward','done']])
df_reward=df[['reward','done']]

print(df_reward)
df_reward=df_reward.assign(reward=df_reward.groupby(df_reward.index//48)['reward'].transform('sum'))
print(df_reward,"!!!!",df_reward[df_reward['done']==1]['reward'],"222")
df_info = df_info.assign(reward=df_reward[df_reward['done']==1]['reward'])
# df_info.loc['reward']=df_reward['reward']
print(df_info[['deliveryPeriodIndex','advertiserNumber','budget','realAllCost','CPAConstraint','reward','done']])
print("@#2323")
df_result=df_info[['deliveryPeriodIndex','advertiserNumber','budget','realAllCost','CPAConstraint','reward','done']]
df_result=df_result.assign(score=df_result['reward'])

# all_cost = agent.budget - agent.remaining_budget
# cpa_real = all_cost / (all_reward + 1e-10)


for index, row in df_result.iterrows():
    row['score']=getScore_nips(row['reward'], row['realAllCost']/(row['reward'] + 1e-10), row['CPAConstraint'])
    print(index)

print(df_result)
df_result=df_result.sort_values(by='score', ascending=False)
print(df_result)
df_result.to_csv('/home/weihanxiao-23/autobidding1/data/trajectory/trajectory_score.csv', index=False)
