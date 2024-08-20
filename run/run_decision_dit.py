import os
from collections import OrderedDict
from copy import deepcopy

import torch
from bidding_train_env.baseline.dit.DiT import (DIT)
import time
from bidding_train_env.baseline.dd.dataset import aigb_dataset
from torch.utils.data import DataLoader

from bidding_train_env.baseline.dit.diffusion import create_diffusion

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def run_decision_dit(
        save_path="saved_model/DiTtest",
        train_epoch=1,
        batch_size=1000,
        model_name='DiT-XL/2'
):
    """
        Trains a new DiT model.
        """

    # 1.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("train_epoch", train_epoch)
    print("batch-size", batch_size)


    #2.  Create model:
    algorithm = DIT()
    # Note that parameter initialization is done within the DiT constructor
    algorithm = algorithm.to(device)
    model = algorithm.model

    args_dict = {'data_version': 'monk_data_small'}
    dataset = aigb_dataset(algorithm.step_len, **args_dict)
    dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, num_workers=2, pin_memory=True)

    # 参数数量
    total_params = sum(p.numel() for p in algorithm.parameters())
    print(f"参数数量：{total_params}")

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    # opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # 3. Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    # train_steps = 0
    # log_steps = 0
    # running_loss = 0
    # start_time = time()
    # 4. 迭代训练

    epi = 1
    for epoch in range(0, train_epoch):
        for batch_index, (states, actions, returns, masks) in enumerate(dataloader):
            states.to(device)
            actions.to(device)
            returns.to(device)
            masks.to(device)

            start_time = time.time()

            # 训练
            loss = algorithm.trainStep(states, actions, returns, masks,diffusion)

            end_time = time.time()
            print(
                f"第{epi}个batch训练时间为: {end_time - start_time} s, loss: {loss}")
            epi += 1

    model.eval()
    # algorithm.save_model(save_path, epi)
    algorithm.save_net(save_path, epi)


    #
    # for epoch in range(args.epochs):
    #     if accelerator.is_main_process:
    #         logger.info(f"Beginning epoch {epoch}...")
    #     for x, y in loader:
    #         x = x.to(device)
    #         y = y.to(device)
    #         x = x.squeeze(dim=1)
    #         y = y.squeeze(dim=1)
    #         t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
    #         model_kwargs = dict(y=y)
    #         loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
    #         loss = loss_dict["loss"].mean()
    #         opt.zero_grad()
    #         accelerator.backward(loss)
    #         opt.step()
    #         update_ema(ema, model)
    #
    #         # Log loss values:
    #         running_loss += loss.item()
    #         log_steps += 1
    #         train_steps += 1
    #         if train_steps % args.log_every == 0:
    #             # Measure training speed:
    #             torch.cuda.synchronize()
    #             end_time = time()
    #             steps_per_sec = log_steps / (end_time - start_time)
    #             # Reduce loss history over all processes:
    #             avg_loss = torch.tensor(running_loss / log_steps, device=device)
    #             avg_loss = avg_loss.item() / accelerator.num_processes
    #             if accelerator.is_main_process:
    #                 logger.info(
    #                     f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
    #             # Reset monitoring variables:
    #             running_loss = 0
    #             log_steps = 0
    #             start_time = time()
    #
    #         # Save DiT checkpoint:
    #         if train_steps % args.ckpt_every == 0 and train_steps > 0:
    #             if accelerator.is_main_process:
    #                 checkpoint = {
    #                     "model": model.module.state_dict(),
    #                     "ema": ema.state_dict(),
    #                     "opt": opt.state_dict(),
    #                     "args": args
    #                 }
    #                 checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
    #                 torch.save(checkpoint, checkpoint_path)
    #                 logger.info(f"Saved checkpoint to {checkpoint_path}")
    #
    # model.eval()  # important! This disables randomized embedding dropout
    # # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    #
    # if accelerator.is_main_process:
    #     logger.info("Done!")




if __name__ == '__main__':
    run_decision_dit()
