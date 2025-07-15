# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from biped_tasks.utils.cleanrl.rl_cfg import CleanRlPpoActorCriticCfg


@configclass
class H12_12dof_FlatPPORunnerCfg(CleanRlPpoActorCriticCfg):
    save_interval = 200
    learning_rate = 3.0e-4
    num_steps = 24
    num_iterations = 50000
    gamma = 0.99
    gae_lambda = 0.95
    updates_epochs = 5
    minibatch_size = 16384
    clip_coef = 0.2
    ent_coef = 0.0081
    vf_coef = 2.0
    max_grad_norm = 1.0
    norm_adv = True
    clip_vloss = True
    anneal_lr = True
    experiment_name = "h12_12dof_flat"
    logger = "tensorboard"
    wandb_project = "h12_12dof_flat"
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
