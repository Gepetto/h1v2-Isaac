import argparse
import logging
import numpy as np
import yaml
from pathlib import Path

from robot_assets.assets import SCENE_PATHS
from robot_deploy.logging.plotter import (
    JointDataPlotter,
    get_joint_limits_from_mujoco,
    load_cmd_log_binary,
    load_state_log_binary,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Per-Joint Plotter for Kinematic Log Files.")
    parser.add_argument("log_dir", type=Path, help="Path to the log directory (containing low_state.bin)")
    parser.add_argument("--config_path", type=Path, default=None, help="Path to config file (for robot model path)")
    parser.add_argument(
        "--show", action="store_true", default=False, help="Set to open all generated plots in separate windows."
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    args = parse_args()
    config_path = args.config_path or Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        err_msg = f"Config file not found at {config_path}. Cannot find robot model path."
        logger.error(err_msg)
        return

    with config_path.open() as file:
        config = yaml.safe_load(file)

    robot_name = config["mujoco"]["robot_name"]
    scene_name = config["mujoco"]["scene_name"]
    try:
        scene_path = SCENE_PATHS[robot_name][scene_name]
    except Exception:
        logger.error(
            "Could not retrieve scene_path from SCENE_PATHS. Please check 'robot_assets.assets' or manually define the scene_path."
        )
        return

    state_file_path = args.log_dir / "low_state.bin"
    cmd_file_path = args.log_dir / "low_cmd.bin"

    if not state_file_path.exists():
        err_msg = f"Log file 'low_state.bin' not found in directory: {args.log_dir}"
        logger.error(err_msg)
        return

    try:
        state_data = load_state_log_binary(state_file_path)
    except Exception as e:
        err_msg = f"Error reading state binary log file: {e}"
        logger.error(err_msg)
        return

    # --- Load Command Data ---
    cmd_data = {}
    if cmd_file_path.exists():
        try:
            # Pass state joint names to ensure alignment
            cmd_data = load_cmd_log_binary(cmd_file_path, state_data["joint_names"])
            info_msg = f"Loaded {len(cmd_data.get('timestamp', []))} command records."
            logger.info(info_msg)
        except Exception as e:
            err_msg = f"Error reading command binary log file: {e}. Command plots will be missing."
            logger.error(err_msg)
            cmd_data = {"timestamp": np.array([]), "q_cmd": np.zeros((0, state_data["nb_motors"]))}
    else:
        logger.warning("Command log file 'low_cmd.bin' not found. Command plots will be missing.")
        cmd_data = {"timestamp": np.array([]), "q_cmd": np.zeros((0, state_data["nb_motors"]))}

    info_msg = f"Loaded {len(state_data['timestamp'])} state frames with {state_data['nb_motors']} joints."
    logger.info(info_msg)

    info_msg = f"Fetching joint limits from MuJoCo model at: {scene_path}"
    logger.info(info_msg)
    joint_limits_map = get_joint_limits_from_mujoco(scene_path, state_data["joint_names"])

    if not joint_limits_map:
        logger.warning("No position limits found or MuJoCo model failed to load. Plots will not show limits.")
    else:
        info_msg = f"Successfully loaded limits for {len(joint_limits_map)} joints."
        logger.info(info_msg)

    output_plots_dir = args.log_dir / "plots"
    # Pass command data to the plotter
    plotter = JointDataPlotter(state_data, cmd_data, joint_limits=joint_limits_map)

    try:
        plotter.plot_all(save_dir=output_plots_dir, show=args.show)
        if not args.show:
            info_msg = f"All plots saved successfully to {output_plots_dir}"
            logger.info(info_msg)
    except Exception as e:
        err_msg = f"Error during plotting: {e}"
        logger.error(err_msg)


if __name__ == "__main__":
    main()
