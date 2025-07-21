import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(log_dir: Path) -> list[dict]:
    """Load metrics data from JSON file"""
    metrics_path = log_dir / "metrics.json"
    with metrics_path.open() as f:
        return json.load(f)


def load_violations(log_dir: Path) -> list[dict]:
    """Load safety violations data from JSON file"""
    violations_path = log_dir / "safety_check.json"
    with violations_path.open() as f:
        return json.load(f)


def plot_joint_details(metrics: list[dict], violations: list[dict], save_dir: Path):
    """Create detailed plots for each joint"""
    joint_names = list(metrics[0]["joint_pos"].keys())
    timestamps = np.array([m["timestamp"] for m in metrics])

    for joint in joint_names:
        if joint == "floating_base_joint":
            continue
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Position
        positions = [m["joint_pos"][joint] for m in metrics]
        ax1.plot(timestamps, positions, "b-")
        ax1.set_ylabel("Position (rad)")

        # Velocity
        velocities = [m["joint_vel"][joint] for m in metrics]
        ax2.plot(timestamps, velocities, "g-")
        ax2.set_ylabel("Velocity (rad/s)")

        # Torque
        torques = [m["applied_torques"][joint] for m in metrics]
        ax3.plot(timestamps, torques, "r-")

        # Mark violations for this joint
        joint_violations = [v for v in violations if v["joint_name"] == joint]
        for violation in joint_violations:
            if violation["check_type"] == "position":
                ax1.axvline(violation["timestamp"], color="k", linestyle="--", alpha=0.5)
            elif violation["check_type"] == "velocity":
                ax2.axvline(violation["timestamp"], color="k", linestyle="--", alpha=0.5)
            elif violation["check_type"] == "torque":
                ax3.axvline(violation["timestamp"], color="k", linestyle="--", alpha=0.5)

        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Torque (Nm)")

        plt.suptitle(f"Joint: {joint}")
        plt.tight_layout()

        # Save individual joint plot
        joint_plot_path = save_dir / f"joint_{joint}.png"
        plt.savefig(joint_plot_path, dpi=300)
        plt.close()

    print(f"Saved detailed joint plots to {save_dir}")


def plot_base_metrics(metrics: list[dict], violations: list[dict], save_dir: Path):
    """Create detailed plots focusing on base-specific metrics"""
    timestamps = np.array([m["timestamp"] for m in metrics])

    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # 1. Base Linear Position
    ax = axes[0]
    base_pos = np.array([m["base_lin_pos"] for m in metrics])

    for i, label in enumerate(["X", "Y", "Z"]):
        ax.plot(timestamps, base_pos[:, i], label=label)

    ax.set_ylabel("Position (m)")
    ax.set_title("Base Linear Position")
    ax.legend()
    ax.grid(True)

    # 2. Base Orientation (Quaternion Components)
    ax = axes[1]
    quats = np.array([m["base_quat_pos"] for m in metrics])
    for i, label in enumerate(["w", "x", "y", "z"]):
        ax.plot(timestamps, quats[:, i], label=label)

    ax.set_ylabel("Quaternion Value")
    ax.set_title("Base Orientation (Quaternion)")
    ax.legend()
    ax.grid(True)

    # 3. Base Linear and Angular Velocity
    ax = axes[2]
    # Linear velocity
    lin_vel = np.array([m["base_lin_vel"] for m in metrics])
    for i, label in enumerate(["Vx", "Vy", "Vz"]):
        ax.plot(timestamps, lin_vel[:, i], "--", label=f"Lin {label}")

    # Angular velocity
    ang_vel = np.array([m["base_quat_vel"] for m in metrics])
    for i, label in enumerate(["ωx", "ωy", "ωz"]):
        ax.plot(timestamps, ang_vel[:, i], "-", label=f"Ang {label}")

    # Mark velocity violations (if any)
    vel_violations = [v for v in violations if v["check_type"] == "velocity" and v["joint_name"] == "base"]
    for violation in vel_violations:
        ax.axvline(violation["timestamp"], color="r", linestyle="--", alpha=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity")
    ax.set_title("Base Velocity (Linear and Angular)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True)

    plt.tight_layout()

    # Save the figure
    plot_path = save_dir / "base_metrics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved base metrics plot to {plot_path}")
    plt.close()

    # Additional: Plot 3D trajectory if you have position data
    plot_3d_trajectory(base_pos, save_dir)


def plot_3d_trajectory(base_pos: np.ndarray, save_dir: Path):
    """Plot the 3D trajectory of the base"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(base_pos[:, 0], base_pos[:, 1], base_pos[:, 2], "b-", linewidth=2)
    ax.scatter(base_pos[0, 0], base_pos[0, 1], base_pos[0, 2], color="g", s=100, label="Start")
    ax.scatter(base_pos[-1, 0], base_pos[-1, 1], base_pos[-1, 2], color="r", s=100, label="End")

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title("Base 3D Trajectory")
    ax.legend()

    plot_path = save_dir / "base_3d_trajectory.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved 3D trajectory plot to {plot_path}")
    plt.close()


def plot_foot_forces(metrics: list[dict], save_dir: Path):
    """Plot foot contact forces over time"""
    timestamps = np.array([m["timestamp"] for m in metrics])

    plt.figure(figsize=(10, 6))

    for foot in ["left_ankle_roll_link", "right_ankle_roll_link"]:
        forces = [m["foot_contact_forces"][foot] for m in metrics]
        plt.plot(timestamps, forces, label=foot)

    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title("Foot Contact Forces")
    plt.legend()
    plt.grid(True)

    plot_path = save_dir / "foot_forces.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved foot forces plot to {plot_path}")


def plot_joint_reference_tracking(metrics: list[dict], save_dir: Path):
    """Plot joint positions vs their reference commands"""
    joint_names = list(metrics[0]["joint_pos"].keys())
    timestamps = np.array([m["timestamp"] for m in metrics])

    for joint in joint_names:
        if joint not in metrics[0]["desired_q_ref"]:
            continue

        plt.figure(figsize=(10, 6))

        actual_pos = [m["joint_pos"][joint] for m in metrics]
        desired_pos = [m["desired_q_ref"][joint] for m in metrics]

        plt.plot(timestamps, actual_pos, "b-", label="Actual")
        plt.plot(timestamps, desired_pos, "r--", label="Desired")

        plt.xlabel("Time (s)")
        plt.ylabel("Position (rad)")
        plt.title(f"Joint Tracking: {joint}")
        plt.legend()
        plt.grid(True)

        plot_path = save_dir / f"joint_tracking_{joint}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

    print(f"Saved joint tracking plots to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot robot metrics")
    parser.add_argument("log_dir", type=str, help="Directory containing metrics.json and safety_check.json")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save plots (default: log_dir/plots)")

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    save_dir = Path(args.output_dir) if args.output_dir else log_dir / "plots"
    save_dir.mkdir(exist_ok=True)

    metrics = load_metrics(log_dir)
    violations = load_violations(log_dir)

    plot_base_metrics(metrics, violations, save_dir)
    plot_joint_details(metrics, violations, save_dir)
    plot_foot_forces(metrics, save_dir)
    plot_joint_reference_tracking(metrics, save_dir)


if __name__ == "__main__":
    main()
