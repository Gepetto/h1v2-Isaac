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

from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_position_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    joint_pos = data.joint_pos[:, asset_cfg.joint_ids]
    lower_violation = data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0] - joint_pos
    upper_violation = joint_pos - data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]

    return torch.maximum(lower_violation, upper_violation)


def joint_velocity_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    joint_vel = data.joint_vel[:, asset_cfg.joint_ids]
    limit = data.joint_vel_limits[:, asset_cfg.joint_ids]
    violation = torch.abs(joint_vel) - limit
    return violation


def joint_torque_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    torques = data.applied_torque[:, asset_cfg.joint_ids]
    limit = data.joint_effort_limits[:, asset_cfg.joint_ids]
    violation = torch.abs(torques) - limit
    return violation


def joint_torque(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    cstr = torch.abs(data.applied_torque[:, asset_cfg.joint_ids]) - limit
    return cstr


def joint_velocity(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return torch.abs(data.joint_vel[:, asset_cfg.joint_ids]) - limit


def joint_acceleration(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
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
    touchdown = contact_sensor.compute_first_contact(env.step_dt)[:, asset_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, asset_cfg.body_ids]

    # Get velocity command and check ALL components against deadzone
    velocity_cmd = env.command_manager.get_command("base_velocity")[:, :3]
    cmd_active = torch.any(
        torch.abs(velocity_cmd) > velocity_deadzone,  # Check x,y,z separately
        dim=1,
    ).float().unsqueeze(1)  # Shape: (num_envs, 1)

    # Apply constraint only when command is active (any component > deadzone)
    cstr = (limit - last_air_time) * touchdown.float() * cmd_active
    return cstr


def joint_range(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
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
    return (
        torch.abs(
            env.action_manager._action[:, asset_cfg.joint_ids]
            - env.action_manager._prev_action[:, asset_cfg.joint_ids],
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
                net_contact_forces[:, :, asset_cfg.body_ids], dim=-1,
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

    Only applies when all components of the base velocity command are within the deadzone.
    """
    data = env.scene[asset_cfg.name].data

    # Get base velocity command
    velocity_cmd = env.command_manager.get_command("base_velocity")[:, :3]

    # Find environments where all command components are below the deadzone
    cmd_inactive_mask = torch.all(torch.abs(velocity_cmd) < velocity_deadzone, dim=1)  # (num_envs,)

    if cmd_inactive_mask.sum() == 0:
        # No env matches — return zero constraint (or any safe fallback)
        return torch.zeros((env.num_envs, sum(env.action_manager.action_term_dim)), device=env.device)

    # Filter only relevant environments
    active_joint_vel = data.joint_vel[cmd_inactive_mask][:, asset_cfg.joint_ids]

    # Compute constraint just for those
    cstr_nomove = torch.abs(active_joint_vel) - joint_vel_limit

    # Repeat to match the number of original environments
    num_repeat = env.num_envs // cstr_nomove.shape[0] + 1
    cstr_nomove = cstr_nomove.repeat((num_repeat, 1))[:env.num_envs]

    return cstr_nomove


def foot_orientation(
    env: ManagerBasedRLEnv,
    limit: float,
    desired_projected_gravity: list,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    foot_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids, :].reshape(-1, 4)
    foot_to_world = torch.transpose(matrix_from_quat(foot_quat_w), dim0=1, dim1=2)
    gravity_vec_foot = torch.matmul(foot_to_world, robot.data.GRAVITY_VEC_W[0].squeeze(0))
    desired_projected_gravity = torch.tensor(desired_projected_gravity, dtype=torch.float32, device=env.device)
    zero_mask = (desired_projected_gravity == 0)

    contact_sensor = env.scene[sensor_cfg.name]
    touchdown = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    return (torch.norm(gravity_vec_foot[:, zero_mask], dim=1).unsqueeze(-1) - limit) * touchdown.float()

def base_height(
    env: ManagerBasedRLEnv,
    height: float,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    base_height = robot.data.root_pos_w[:, 2]

    violation = torch.where(
        (base_height < height - std) | (base_height > height + std),
        torch.tensor(1.0, device=base_height.device),
        torch.tensor(0.0, device=base_height.device),
    )

    return violation

def foot_clearance(
    env: ManagerBasedRLEnv,
    min_height: float,
    velocity_deadzone: float,
    pos_asset_cfg: SceneEntityCfg,  # For foot positions (typically robot body)
    contact_asset_cfg: SceneEntityCfg,  # For contact sensors
) -> torch.Tensor:
    # Get foot positions from position asset
    pos_asset = env.scene[pos_asset_cfg.name]
    foot_heights = pos_asset.data.body_link_pos_w[:, pos_asset_cfg.body_ids, 2]  # shape: (num_envs, num_feet)
    # Get contact information
    contact_sensor = env.scene[contact_asset_cfg.name]
    touchdown = contact_sensor.compute_first_contact(env.step_dt)[:, contact_asset_cfg.body_ids]

    # Initialize swing max height tracking if needed
    if not hasattr(pos_asset.data, 'swing_max_height'):
        pos_asset.data.swing_max_height = torch.zeros_like(foot_heights)

    # violation
    violation = (min_height - pos_asset.data.swing_max_height.clone()) * touchdown.float()

    # Update max height for feet in swing phase
    pos_asset.data.swing_max_height = torch.where(
        ~touchdown.bool(),
        torch.maximum(pos_asset.data.swing_max_height, foot_heights),
        torch.zeros_like(foot_heights),  # Reset when not in swing
    )

    # Get velocity command and check ALL components against deadzone
    velocity_cmd = env.command_manager.get_command("base_velocity")[:, :3]
    cmd_active = torch.any(
        torch.abs(velocity_cmd) > velocity_deadzone,  # Check x,y,z separately
        dim=1,
    ).float().unsqueeze(1)  # Shape: (num_envs, 1)

    return violation * cmd_active
