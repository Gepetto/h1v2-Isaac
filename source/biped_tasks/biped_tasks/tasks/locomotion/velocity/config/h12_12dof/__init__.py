# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from biped_tasks.utils.cat.cat_env import CaTEnv

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Rough-H12_12dof-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:H12_12dof_RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H12_12dof_RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-H12_12dof-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:H12_12dof_RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H12_12dof_RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-H12_12dof-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:H12_12dof_FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H12_12dof_FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-H12_12dof-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:H12_12dof_FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H12_12dof_FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-CaT-Flat-H12_12dof-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:H12_12dof_EnvCfg",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:H12_12dof_FlatPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-CaT-Flat-H12_12dof-Play-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:H12_12dof_EnvCfg_PLAY",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:H12_12dof_FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rsl-H12_12dof-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rsl_env_cfg:H12_12dof_FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H12_12dof_FlatPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rsl-H12_12dof-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rsl_env_cfg:H12_12dof_FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H12_12dof_FlatPPORunnerCfg",
    },
)
