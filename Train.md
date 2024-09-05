### Train

#### Train decision-transformer

运行：python main/main_decision_transformer.py 

训练数据集：单个轨迹数据

模型保存：saved_model/DTtest

可修改 run_decision_transformer.py 中 train_model 函数中的 step_num 参数调整训练时长

评估：

```
bidding_train_env/strategy/__init__.py
from .dt_bidding_strategy import DtBiddingStrategy as PlayerBiddingStrategy
python main/main_test.py
```

#### Train decision-diffuser

运行：python main/main_decision_diffuser.py

训练数据集：单个轨迹数据/所有轨迹数据（在bidding_train_env/baseline/dd/dataset.py中修改）

模型保存：saved_model/DDtest

可修改main_decision_diffuser.py中的train_epoch参数调整训练轮数，并保存每轮的模型diffuser_{epoch}.pt

评估：

```
bidding_train_env/strategy/__init__.py
from .dd_bidding_strategy import DdBiddingStrategy as PlayerBiddingStrategy
python main/main_test.py
```

在main_test.py中注释code可获得不同epoch的评估结果