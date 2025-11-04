import argparse
import logging
import yaml
from pathlib import Path

import mujoco

from robot_assets.assets import SCENE_PATHS
from robot_deploy.logging.replay import ThreadedReplayer, load_state_log_binary

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Kinematic replayer for Unitree logs.")
    parser.add_argument("log_dir", type=Path, help="Path to the log directory (containing low_state.bin)")
    parser.add_argument("--config_path", type=Path, default=None, help="Path to config file")
    parser.add_argument("--render_dt", type=float, default=1.0 / 60.0, help="Render update period in seconds")
    parser.add_argument(
        "--video", action="store_true", help="Record a video to log_dir/video.mp4 instead of launching the GUI."
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    args = parse_args()
    config_path = args.config_path or Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        err_msg = f"Config file not found at {config_path}"
        logger.error(err_msg)
        return

    with config_path.open() as file:
        config = yaml.safe_load(file)

    robot_name = config["mujoco"]["robot_name"]
    scene_name = config["mujoco"]["scene_name"]
    try:
        scene_path = SCENE_PATHS[robot_name][scene_name]
        fix_base = config["mujoco"].get("fix_base", False)
    except KeyError as e:
        err_msg = f"Config file missing required key: {e}"
        logger.error(err_msg)
        return

    log_file_path = args.log_dir / "low_state.bin"
    try:
        state_data = load_state_log_binary(log_file_path)
    except FileNotFoundError:
        return
    except ValueError as e:
        err_msg = f"Error reading binary log file: {e}"
        logger.error(err_msg)
        return

    info_msg = f"Loaded {len(state_data['timestamp'])} log frames with {state_data['nb_motors']} joints."
    logger.info(info_msg)

    try:
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
    except Exception as e:
        err_msg = f"Failed to load MuJoCo model from {scene_path}: {e}"
        logger.error(err_msg)
        return

    if fix_base:
        if model.neq > 0:
            logger.info("Fixing robot base")
            model.eq_active0[0] = 1
        else:
            logger.warning("Config 'fix_base' is True, but model has no equality constraints.")

    replayer_config = {"render_dt": args.render_dt}

    if args.video:
        replayer_config["video_path"] = args.log_dir / "video.mp4"

    try:
        replayer = ThreadedReplayer(model, data, state_data, replayer_config)
    except RuntimeError as e:
        logger.error(e)
        return

    try:
        replayer.start()
    except KeyboardInterrupt:
        logger.info("Replay interrupted by user")
    except Exception as e:
        err_msg = f"Error during replay: {e}"
        logger.error(err_msg)
    finally:
        logger.info("Closing replayer...")
        replayer.close()
        logger.info("Replay completed")


if __name__ == "__main__":
    main()
