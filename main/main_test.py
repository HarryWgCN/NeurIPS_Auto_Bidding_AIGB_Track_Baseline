import numpy as np
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run.run_evaluate import run_test

torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    run_test(0)
    
# test different epoch evaluate result

# import pandas as pd
# if __name__ == "__main__":
#     results = []
#     for i in range(0, 10):
#         result = run_test(i)
#         results.append(result)
#     df = pd.DataFrame(results, columns=['epoch', 'Total Reward', 'Total Cost', 'CPA-real','CPA-constraint','Score'])
#     df.to_excel('/home/zhangyuxuan-23/auto-bidding_1.xlsx', index=False)
