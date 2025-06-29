# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_torque(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr = torch.abs(data.applied_torque[:, asset_cfg.joint_ids]) - limit
    return cstr


def joint_velocity(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return torch.abs(data.joint_vel[:, asset_cfg.joint_ids]) - limit


def joint_acceleration(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return torch.abs(data.joint_acc[:, asset_cfg.joint_ids]) - limit


def contact(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return torch.any(
        torch.max(
            torch.norm(net_contact_forces[:, :, asset_cfg.body_ids], dim=-1),
            dim=1,
        )[0]
        > 1.0,
        dim=1,
    )

def base_orientation(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return torch.norm(data.projected_gravity_b[:, :2], dim=1) - limit

def air_time(
    env: ManagerBasedRLEnv,
    limit: float,
    velocity_deadzone: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    touchdown = contact_sensor.compute_first_contact(env.step_dt)[:, asset_cfg.joint_ids]
    last_air_time = contact_sensor.data.last_air_time[:, asset_cfg.joint_ids]
    
    # Get velocity command and check ALL components against deadzone
    velocity_cmd = env.command_manager.get_command("base_velocity")[:, :3]
    cmd_active = torch.any(
        torch.abs(velocity_cmd) > velocity_deadzone,  # Check x,y,z separately
        dim=1
    ).float().unsqueeze(1)  # Shape: (num_envs, 1)
    
    # Apply constraint only when command is active (any component > deadzone)
    cstr = (limit - last_air_time) * touchdown.float() * cmd_active
    return cstr


def joint_range(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return (
        torch.abs(data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids])
        - limit
    )


def action_rate(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return (
        torch.abs(
            env.action_manager._action[:, asset_cfg.joint_ids]
            - env.action_manager._prev_action[:, asset_cfg.joint_ids]
        )
        / env.step_dt
        - limit
    )


def foot_contact_force(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return (
        torch.max(torch.norm(net_contact_forces[:, :, asset_cfg.body_ids], dim=-1), dim=1)[0]
        - limit
    )

def foot_contact(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    # Compute number of feet in contact per environment
    foot_contacts = (
        torch.max(
            torch.norm(
                net_contact_forces[:, :, asset_cfg.body_ids], dim=-1
            ),
            dim=1,
        )[0] > 1.0  # Boolean: (envs, num_feet)
    ).sum(1)  # Sum over feet → (envs,)

    # Penalize cases where number of contacts is not 1 or 2
    contact_cstr = ((foot_contacts < 1) | (foot_contacts > 2)).float()

    return contact_cstr 

def no_move(
    env: ManagerBasedRLEnv,
    velocity_deadzone: float,
    joint_vel_limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Constraint that penalizes joint movement when the robot should be stationary.
    
    Now only activates when ALL velocity command components (x,y,z) are below deadzone.
    """
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    
    # Get velocity command and check ALL components against deadzone
    velocity_cmd = env.command_manager.get_command("base_velocity")[:, :3]
    cmd_inactive = (
        torch.all(torch.abs(velocity_cmd) < velocity_deadzone, dim=1)  # All components must be below
        .float()
        .unsqueeze(1)  # Shape: (num_envs, 1)
    )
    
    # Apply constraint only when command is inactive (all components < deadzone)
    cstr_nomove = (torch.abs(data.joint_vel[:, asset_cfg.joint_ids]) - joint_vel_limit) * cmd_inactive
    return cstr_nomove