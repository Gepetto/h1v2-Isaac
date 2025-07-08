import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from logger import load_cmd_log, load_state_log


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    return parser.parse_args()


def plot_cmds(cmds):
    qpos = cmds["qpos"]

    joint_id = 5

    joint_names = [
        "left_hip_yaw",
        "left_hip_pitch",
        "left_hip_roll",
        "left_knee",
        "left_ankle_pitch",
        "left_ankle_roll",
        "right_hip_yaw",
        "right_hip_pitch",
        "right_hip_roll",
        "right_knee",
        "right_ankle_pitch",
        "right_ankle_roll",
    ]

    _, ax = plt.subplots()
    ax.plot(qpos[:, joint_id])
    ax.set_title(joint_names[joint_id])
    plt.show()


def plot_states(states):
    qpos = states["qpos"]

    _, ax = plt.subplots()
    ax.plot(qpos[:, 12])
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    if "cmd" in args.path.name:
        cmds = load_cmd_log(args.path)
        plot_cmds(cmds)

    elif "state" in args.path.name:
        states = load_state_log(args.path)
        plot_states(states)
