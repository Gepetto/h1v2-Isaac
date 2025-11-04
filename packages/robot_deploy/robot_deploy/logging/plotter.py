import logging
import matplotlib.pyplot as plt
import numpy as np
import struct
from pathlib import Path

import mujoco

logger = logging.getLogger(__name__)


def load_cmd_log_binary(path: Path, joint_names: list[str]) -> dict:
    """Load command data from binary file and align it to state joint names."""
    path = Path(path)
    if not path.exists():
        err_msg = f"Command log file not found: {path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    with path.open("rb") as f:
        # Read header: num_records, num_joints (Cmd)
        header_data = f.read(8)
        if len(header_data) < 8:
            err_msg = "Invalid command binary file: incomplete header"
            raise ValueError(err_msg)
        num_records_cmd, num_joints_cmd = struct.unpack("II", header_data)

        # Read joint names from the command log
        cmd_joint_names = []
        for _ in range(num_joints_cmd):
            name_len_data = f.read(4)
            name_len = struct.unpack("I", name_len_data)[0]
            name_bytes = f.read(name_len)
            cmd_joint_names.append(name_bytes.decode("utf-8"))

        # Map state joint names to their index in the command log
        state_to_cmd_idx = {name: cmd_joint_names.index(name) for name in joint_names}

        timestamps_cmd = np.zeros(num_records_cmd)
        q_cmd = np.zeros((num_records_cmd, len(joint_names)))

        # Determine the size of one motor command record in bytes (4 floats = 4*4 bytes)
        # LowCmdMotorMetric is q, dq, kp, kd (4 floats)
        motor_cmd_size = 4 * 4

        for i in range(num_records_cmd):
            # Read timestamp (double = 8 bytes)
            timestamp_data = f.read(8)
            timestamps_cmd[i] = struct.unpack("d", timestamp_data)[0]

            # Read all motor commands
            all_motor_cmd_data = f.read(num_joints_cmd * motor_cmd_size)

            # Extract and store only the 'q' command for the joints present in the state log
            for j, state_joint_name in enumerate(joint_names):
                cmd_idx = state_to_cmd_idx[state_joint_name]

                # Calculate the start of the joint's data in the read block
                start_byte = cmd_idx * motor_cmd_size

                # Extract the 'q' command (first float in the 4-float block)
                q_cmd_data = all_motor_cmd_data[start_byte : start_byte + 4]
                q_val = struct.unpack("f", q_cmd_data)[0]
                q_cmd[i, j] = q_val

        return {
            "timestamp": timestamps_cmd,
            "q_cmd": q_cmd,
        }


def load_state_log_binary(path: Path) -> dict:
    """Load state data from binary file."""
    path = Path(path)
    if not path.exists():
        err_msg = f"Log file not found: {path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    with path.open("rb") as f:
        # Read header: num_records, num_joints (State)
        header_data = f.read(8)
        if len(header_data) < 8:
            err_msg = "Invalid binary file: incomplete header"
            raise ValueError(err_msg)
        num_records, num_joints = struct.unpack("II", header_data)

        joint_names = []
        for _ in range(num_joints):
            name_len_data = f.read(4)
            name_len = struct.unpack("I", name_len_data)[0]
            name_bytes = f.read(name_len)
            joint_names.append(name_bytes.decode("utf-8"))

        timestamps = np.zeros(num_records)
        q_pos = np.zeros((num_records, num_joints))
        q_vel = np.zeros((num_records, num_joints))
        tau_est = np.zeros((num_records, num_joints))

        # Determine the size of one motor state record in bytes (3 floats = 3*4 bytes)
        motor_state_size = 3 * 4

        for i in range(num_records):
            timestamp_data = f.read(8)
            timestamps[i] = struct.unpack("d", timestamp_data)[0]

            f.read(4 * 4)  # Skip base_orientation (4 floats)
            f.read(4 * 3)  # Skip base_angular_vel (3 floats)

            # Read all motor states
            all_motor_state_data = f.read(num_joints * motor_state_size)

            for j in range(num_joints):
                # Extract q, dq, tau_est (3 floats)
                start_byte = j * motor_state_size
                motor_data = all_motor_state_data[start_byte : start_byte + motor_state_size]
                q_val, dq_val, tau_val = struct.unpack("fff", motor_data)
                q_pos[i, j] = q_val
                q_vel[i, j] = dq_val
                tau_est[i, j] = tau_val

        return {
            "timestamp": timestamps,
            "q_pos": q_pos,
            "q_vel": q_vel,
            "tau_est": tau_est,
            "joint_names": joint_names,
            "nb_motors": num_joints,
        }


class JointDataPlotter:
    """Generates detailed plots: one figure per joint, and separate figures for latency analysis."""

    def __init__(self, state_data: dict, cmd_data: dict, joint_limits: dict[str, list[float]]):
        # State Data
        self.timestamps_state = state_data["timestamp"]
        self.time_rel = self.timestamps_state - self.timestamps_state[0]
        self.q_pos = state_data["q_pos"]
        self.q_vel = state_data["q_vel"]
        self.tau_est = state_data["tau_est"]
        self.joint_names = state_data["joint_names"]
        self.num_joints = state_data["nb_motors"]
        self.joint_limits = joint_limits

        # Command Data
        self.timestamps_cmd = cmd_data.get("timestamp", np.array([]))
        self.q_cmd = self._align_command_data(
            self.timestamps_cmd, cmd_data.get("q_cmd", np.zeros((0, self.num_joints)))
        )

        # --- Latency Calculation ---
        if len(self.timestamps_state) > 1:
            self.latency_state = np.diff(self.timestamps_state) * 1000  # Convert to ms immediately
            self.frames_state = np.arange(1, len(self.timestamps_state))
        else:
            self.latency_state = np.array([])
            self.frames_state = np.array([])

        if len(self.timestamps_cmd) > 1:
            self.latency_cmd = np.diff(self.timestamps_cmd) * 1000  # Convert to ms immediately
            self.frames_cmd = np.arange(1, len(self.timestamps_cmd))
        else:
            self.latency_cmd = np.array([])
            self.frames_cmd = np.array([])

    import numpy as np
    # Assuming self.q_pos, self.num_joints, and self.timestamps_state are available in the class context.

    def _align_command_data(self, cmd_timestamps: np.ndarray, q_cmd_data: np.ndarray) -> np.ndarray:
        """Interpolates command data onto the state timestamps using zero-order (hold previous) interpolation."""

        if cmd_timestamps.size == 0:
            N_state_steps = self.timestamps_state.size
            N_joints = self.num_joints
            if N_state_steps == 0 or N_joints == 0:
                return np.array([], dtype=np.float32).reshape(0, 0)

            logger.warning("No command data available to align.")
            return np.full((N_state_steps, N_joints), np.nan)
        indices = np.searchsorted(cmd_timestamps, self.timestamps_state, side="right")
        indices = indices - 1
        indices = np.clip(indices, 0, cmd_timestamps.size - 1)
        q_cmd_aligned = q_cmd_data[indices]

        return q_cmd_aligned

    def plot_single_joint(self, joint_idx: int, save_dir: Path, close_fig: bool = True):
        """Generates a single figure with position, velocity, and torque for one joint."""
        joint_name = self.joint_names[joint_idx]

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
        fig.suptitle(f"Joint Analysis: {joint_name}", fontsize=16, fontweight="bold")

        # --- Position Plot (with Command) ---
        axes[0].plot(self.time_rel, self.q_pos[:, joint_idx], label="State Position", color="k")
        axes[0].plot(
            self.time_rel, self.q_cmd[:, joint_idx], label="Command Position", color="orange", drawstyle="steps-post"
        )
        axes[0].set_ylabel("Position (rad)")
        axes[0].set_title("Position (State vs. Command)")

        if joint_name in self.joint_limits:
            min_pos, max_pos = self.joint_limits[joint_name]
            axes[0].axhline(min_pos, color="r", linestyle=":", alpha=0.7, label="Min Limit")
            axes[0].axhline(max_pos, color="r", linestyle=":", alpha=0.7, label="Max Limit")
        axes[0].legend(loc="best", fontsize="small")

        # --- Velocity Plot ---
        axes[1].plot(self.time_rel, self.q_vel[:, joint_idx], label="State Velocity", color="g")
        axes[1].set_ylabel("Velocity (rad/s)")
        axes[1].set_title("Velocity")

        # --- Torque Plot ---
        axes[2].plot(self.time_rel, self.tau_est[:, joint_idx], label="Estimated Torque", color="b")
        axes[2].set_ylabel("Torque (Nm)")
        axes[2].set_title("Torque")
        axes[2].set_xlabel("Time (s)")

        for ax in axes:
            ax.grid(True, linestyle=":", alpha=0.7)

        safe_name = joint_name.replace("/", "_").replace(":", "")
        save_path = save_dir / f"joint_{safe_name}_analysis.pdf"
        plt.savefig(save_path)

        if close_fig:
            plt.close(fig)

    # REMOVED plot_latency (timeseries plot)

    def _plot_latency_histogram(
        self, latencies_ms: np.ndarray, log_type: str, color: str, save_dir: Path, close_fig: bool = True
    ):
        """Internal helper to generate a single latency histogram."""
        if latencies_ms.size < 10:
            warn_msg = f"Not enough data points ({latencies_ms.size}) to plot {log_type} latency histogram."
            logger.warning(warn_msg)
            return

        median_latency_ms = np.median(latencies_ms)
        avg_frequency_hz = 1000 / median_latency_ms

        # Determine robust binning
        q1, q3 = np.percentile(latencies_ms, [25, 75])
        iqr = q3 - q1
        bin_width = 0.5

        hist_min = max(0, median_latency_ms - 5 * iqr)
        hist_max = median_latency_ms + 5 * iqr

        if hist_max - hist_min < 2 * bin_width:
            hist_min = max(0, median_latency_ms - 5)
            hist_max = median_latency_ms + 5

        bins = np.arange(hist_min, hist_max, bin_width)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

        ax.hist(latencies_ms, bins=bins, color=color, edgecolor="black", zorder=2, alpha=0.7)

        ax.axvline(
            median_latency_ms,
            color="red",
            linestyle="--",
            linewidth=2,
            zorder=3,
            label=f"Median Latency: {median_latency_ms:.2f} ms\nAvg Frequency: {avg_frequency_hz:.1f} Hz",
        )

        ax.set_title(f"{log_type} Log Latency Histogram (Timestamp Difference)", fontsize=14)
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count (Number of Frames)")
        ax.grid(axis="y", linestyle=":", alpha=0.7, zorder=1)
        ax.legend(loc="upper right", fontsize="small")
        ax.set_xlim(min(bins), max(bins))

        save_path = save_dir / f"{log_type.lower()}_latency_histogram_analysis.pdf"
        plt.savefig(save_path)

        if close_fig:
            plt.close(fig)

    # MODIFIED: Split into two calls
    def plot_latency_histograms(self, save_dir: Path, close_fig: bool = True):
        """Generates separate histograms for State and Command latencies."""

        # 1. State Latency Histogram
        self._plot_latency_histogram(
            latencies_ms=self.latency_state, log_type="State", color="skyblue", save_dir=save_dir, close_fig=close_fig
        )

        # 2. Command Latency Histogram
        self._plot_latency_histogram(
            latencies_ms=self.latency_cmd,
            log_type="Command",
            color="lightgreen",
            save_dir=save_dir,
            close_fig=close_fig,
        )

    # MODIFIED: Removed plot_latency call and updated plot_latency_histogram call
    def plot_all(self, save_dir: Path, show: bool = False):
        """Iterates through all joints to generate individual plots and plots latency."""
        save_dir.mkdir(parents=True, exist_ok=True)
        err_msg = f"Saving plots to directory: {save_dir}"
        logger.info(err_msg)

        close_figures = not show

        for i in range(self.num_joints):
            self.plot_single_joint(i, save_dir=save_dir, close_fig=close_figures)

        # Call the new function to plot separate histograms
        self.plot_latency_histograms(save_dir=save_dir, close_fig=close_figures)

        if show:
            logger.info("All plots generated successfully. Opening figures now...")
            plt.show()
        else:
            logger.info("Plotting complete. Please view the generated PDF files in the 'plots' subdirectory.")


def get_joint_limits_from_mujoco(model_path: str, log_joint_names: list[str]) -> dict[str, list[float]]:
    """Loads MuJoCo model and extracts position limits for the logged joints."""
    joint_limits_map = {}

    try:
        model = mujoco.MjModel.from_xml_path(model_path)

        for name in log_joint_names:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

            if jnt_id == -1:
                warn_msg = f"Joint '{name}' not found in MuJoCo model. Skipping limits."
                logger.warning(warn_msg)
                continue

            if model.jnt_type[jnt_id] in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                if model.jnt_limited[jnt_id]:
                    min_lim = model.jnt_range[jnt_id, 0]
                    max_lim = model.jnt_range[jnt_id, 1]
                    joint_limits_map[name] = [min_lim, max_lim]
                else:
                    info_msg = f"Joint '{name}' found, but not position-limited in the model."
                    logger.info(info_msg)
            else:
                info_msg = f"Joint '{name}' is a free/ball joint, no simple position limit to plot."
                logger.info(info_msg)

    except Exception as e:
        err_msg = f"Failed to load MuJoCo model from {model_path} or get limits: {e}"
        logger.error(err_msg)
        return {}

    return joint_limits_map
