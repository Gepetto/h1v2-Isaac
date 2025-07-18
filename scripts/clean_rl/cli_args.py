# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse


def add_clean_rl_args(parser: argparse.ArgumentParser):
    arg_group = parser.add_argument_group(
        "clean_rl", description="Arguments for CleanRL agent."
    )
    arg_group.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment folder where logs will be stored.",
    )
    arg_group.add_argument(
        "--resume", type=bool, default=None, help="Whether to resume from a checkpoint."
    )
    arg_group.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Name of the run folder to resume from.",
    )
    arg_group.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint file to resume from."
    )
    arg_group.add_argument(
        "--logger",
        type=str,
        default=None,
        choices={"wandb", "tensorboard"},
        help="Logger module to use.",
    )
    arg_group.add_argument(
        "--log_project_name",
        type=str,
        default=None,
        help="Name of the logging project when using wandb",
    )


def parse_clean_rl_cfg(task_name: str, args_cli: argparse.Namespace):
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    cleanrl_cfg = load_cfg_from_registry(task_name, "clean_rl_cfg_entry_point")
    cleanrl_cfg = update_clean_rl_cfg(cleanrl_cfg, args_cli)
    return cleanrl_cfg


def update_clean_rl_cfg(agent_cfg, args_cli: argparse.Namespace):
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb
    if agent_cfg.logger in {"wandb"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name

    return agent_cfg
