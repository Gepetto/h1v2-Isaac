# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformVelocityCommandWithDeadzone(mdp.UniformVelocityCommand):
    """velocity command sampling class ported from isaacgym CaT"""

    cfg: "UniformVelocityCommandWithDeadzoneCfg"

    def __init__(
        self, cfg: "UniformVelocityCommandWithDeadzoneCfg", env: ManagerBasedEnv
    ):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        super().__init__(cfg, env)

        self.velocity_deadzone = cfg.velocity_deadzone
        self.dt = env.physics_dt
        self.max_episode_length_s = env.max_episode_length_s

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
            )
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        # Identify which envs are in deadzone
        in_deadzone = torch.norm(self.vel_command_b[:, :2], dim=1) < self.velocity_deadzone
        num_envs = self.vel_command_b.shape[0]
        
        # Calculate how many envs we want in deadzone (half of total)
        target_deadzone_count = num_envs // 2
        
        # Get current counts
        current_deadzone_count = in_deadzone.sum().item()
        current_active_count = num_envs - current_deadzone_count
        
        if current_deadzone_count < target_deadzone_count:
            # Need to move some active envs to deadzone
            num_to_deactivate = target_deadzone_count - current_deadzone_count
            active_envs = (~in_deadzone).nonzero(as_tuple=False).flatten()
            deactivate_envs = active_envs[torch.randperm(len(active_envs))[:num_to_deactivate]]
            self.vel_command_b[deactivate_envs, :2] = 0.0
            
        elif current_deadzone_count > target_deadzone_count:
            # Need to activate some deadzone envs
            num_to_activate = current_deadzone_count - target_deadzone_count
            deadzone_envs = in_deadzone.nonzero(as_tuple=False).flatten()
            activate_envs = deadzone_envs[torch.randperm(len(deadzone_envs))[:num_to_activate]]
            self._resample(activate_envs)
        
        # Random angular velocity inversion during the episode to avoid having the robot moving in circle
        p_ang_vel = (
            self.dt / self.max_episode_length_s
        )  # <- time step / duration of X seconds
        # There will be a probability of 0.63 of having at least one swap after X seconds have elapsed
        # (1 / p) policy steps for X seconds, and the probability of having no swap at all is (1 - p)**(1 / p) = 0.37
        # The mean number of swaps for (1 / p) steps with probability p is 1.
        self.vel_command_b[:, 2] *= (
            1
            - 2
            * torch.bernoulli(
                torch.full_like(self.vel_command_b[:, 2], p_ang_vel)
            ).float()
        )


@configclass
class UniformVelocityCommandWithDeadzoneCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the normal velocity command generator."""

    class_type: type = UniformVelocityCommandWithDeadzone
    velocity_deadzone: float = 0.1