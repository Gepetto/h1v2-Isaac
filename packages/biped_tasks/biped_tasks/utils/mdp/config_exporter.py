import re


def get_joint_config(config):
    joint_config = [{"name": joint_name} for joint_name in config.actions.joint_pos.joint_names]
    for part in config.scene.robot.actuators.values():
        for joint_regex, kp in part.stiffness.items():
            for joint in joint_config:
                if re.fullmatch(joint_regex, joint["name"]):
                    joint["kp"] = kp
        for joint_regex, kd in part.damping.items():
            for joint in joint_config:
                if re.fullmatch(joint_regex, joint["name"]):
                    joint["kd"] = kd

    for joint_regex, pos in config.scene.robot.init_state.joint_pos.items():
        for joint in joint_config:
            if re.fullmatch(joint_regex, joint["name"]):
                joint["default_joint_pos"] = pos

    for joint in joint_config:
        joint["enabled"] = True

    return joint_config


def get_deploy_config(config):
    assert config.actions.joint_pos.preserve_order, "'preserve_order' is set to False in the environment configuration"

    command_ranges = {
        "lin_vel_x": list(config.commands.base_velocity.ranges.lin_vel_x),
        "lin_vel_y": list(config.commands.base_velocity.ranges.lin_vel_y),
        "ang_vel_z": list(config.commands.base_velocity.ranges.ang_vel_z),
    }

    observations = []
    for obs_item in config.observations.policy.to_dict().values():
        if not isinstance(obs_item, dict) or "func" not in obs_item:
            continue
        observations.append(
            {
                "name": obs_item["func"].rsplit(":", 1)[1],
                "scale": obs_item.get("scale") or 1,
            },
        )

    joint_config = get_joint_config(config)

    return {
        "control_dt": config.sim.dt * config.decimation,
        "history_length": config.observations.policy.history_length,
        "history_step": config.observations.policy.history_step,
        "action_scale": config.actions.joint_pos.scale,
        "velocity_deadzone": config.commands.base_velocity.velocity_deadzone,
        "command_ranges": command_ranges,
        "observations": observations,
        "joints": joint_config,
    }
