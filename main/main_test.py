import numpy as np
import torch
import os
import sys


from run.run_evaluate import run_test

torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    # for i in range(0,100):
    run_test('temp')
    
# test different epoch evaluate result

# import pandas as pd
# if __name__ == "__main__":
#     results = []
#     for i in range(0, 10):
#         result = run_test(i)
#         results.append(result)
#     df = pd.DataFrame(results, columns=['epoch', 'Total Reward', 'Total Cost', 'CPA-real','CPA-constraint','Score'])
#     df.to_excel('/home/zhangyuxuan-23/auto-bidding_1.xlsx', index=False)
