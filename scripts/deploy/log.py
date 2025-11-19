import argparse
import logging
import os
import time
import yaml
from datetime import datetime
from pathlib import Path

from robot_deploy.logging.logger import UnitreeLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", type=Path, default=None, help="Path to config file")
    return parser.parse_args()


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Configure root logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    args = parse_args()
    config_path = args.config_path or Path(__file__).parent / "config.yaml"

    log_dir = Path(os.path.join("logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    with config_path.open() as file:
        config = yaml.safe_load(file)

    try:
        logger_instance = UnitreeLogger(config["real"]["channel_id"], config["real"]["net_interface"])

        logger.info("Logging commands and states... Press Ctrl+C to stop.")
        while True:
            time.sleep(10)

    except KeyboardInterrupt:
        logger.info("Stop logging requested by user.")
    except Exception as e:
        err_msg = f"An unexpected error occurred: {e}"
        logger.error(err_msg)
    finally:
        err_msg = f"Saving data to {log_dir}..."
        logger.info(err_msg)
        logger_instance.save_data(log_dir)
        logger_instance.close()
        logger.info("Save complete. Exiting.")
