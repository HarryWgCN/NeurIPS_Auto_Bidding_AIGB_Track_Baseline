import os

import torch
from bidding_train_env.baseline.dd.DFUSER import (DFUSER)
import time
from bidding_train_env.baseline.dd.dataset import aigb_dataset
from torch.utils.data import DataLoader


def run_decision_diffuser(
        save_dir="/home/disk2/auto-bidding/models",
        train_epoch=1,
        batch_size=1000):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("train_epoch", train_epoch)
    print("batch-size", batch_size)

    algorithm = DFUSER()
    algorithm = algorithm.to(device)

    args_dict = {'data_version': 'monk_data_small'}
    dataset = aigb_dataset(algorithm.step_len, **args_dict)
    dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, num_workers=2, pin_memory=True)

    # 参数数量
    total_params = sum(p.numel() for p in algorithm.parameters())
    print(f"参数数量：{total_params}")

    # 3. 迭代训练

    for epoch in range(0, train_epoch):
        for batch_index, (states, actions, cpa, returns, masks) in enumerate(dataloader):
            states.to(device)
            actions.to(device)
            cpa.to(device)
            returns.to(device)
            masks.to(device)

            start_time = time.time()

            # 训练
            all_loss, (diffuse_loss, inv_loss) = algorithm.trainStep(states, actions, cpa, returns, masks)
            all_loss = all_loss.detach().clone()
            diffuse_loss = diffuse_loss.detach().clone()
            inv_loss = inv_loss.detach().clone()
            end_time = time.time()
            print(
                f"Epoch {epoch} 第{batch_index}个batch训练时间为: {end_time - start_time} s, all_loss: {all_loss}, diffuse_loss: {diffuse_loss}, inv_loss: {inv_loss}")
        # 保存每轮结果
        print('---------------------------------------------------')
        algorithm.save_net(os.path.join(save_dir, f'diffuser_{epoch}.pt'), epoch)
        print(f'第 {epoch} 轮 model saved to {save_dir}')

    # algorithm.save_model(save_path, epi)
    # algorithm.save_net(save_dir, train_epoch)
    # print(f'Model saved to {save_dir}')


if __name__ == '__main__':
    run_decision_diffuser()
