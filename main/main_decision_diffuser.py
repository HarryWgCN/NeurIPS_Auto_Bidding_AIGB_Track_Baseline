import numpy as np
import torch
import os
import sys
# 获取包含当前Python脚本的目录的父目录。
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from run.run_decision_diffuser import run_decision_diffuser
# os.path.abspath(__file__)：获取当前Python脚本的绝对路径。
# os.path.dirname(...)：从路径中获取目录名。
# sys.path.append(...)：将这个父目录添加到Python用于搜索导入模块时使用的目录列表中。

# 设置随机种子以确保在使用PyTorch（torch）和NumPy（np）时生成的随机数可重现。
torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    run_decision_diffuser(train_epoch=10)
