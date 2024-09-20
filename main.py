import numpy as np
import torch
import os
import sys

from run.run_decision_diffuser import run_decision_diffuser

# 设置随机种子以确保在使用PyTorch（torch）和NumPy（np）时生成的随机数可重现。
torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    run_decision_diffuser(train_epoch=300)
