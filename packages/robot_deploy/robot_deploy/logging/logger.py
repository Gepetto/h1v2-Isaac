import logging
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

from robot_deploy.robots.h12_real import H12Real


@dataclass
class MotorCmdMetric:
    q: float
    dq: float
    kp: float
    kd: float


@dataclass
class LowCmdMetrics:
    timestamp: float
    motor_cmd: dict[str, MotorCmdMetric]


@dataclass
class MotorStateMetric:
    q: float
    dq: float
    tau: float


@dataclass
class LowStateMetrics:
    timestamp: float
    base_orientation: list[float]  # quaternion [w, x, y, z]
    base_angular_vel: list[float]  # gyroscope [x, y, z]
    motor_state: dict[str, MotorStateMetric]


class UnitreeLogger:
    """
    Logs Unitree LowCmd and LowState messages to binary files.

    This class subscribes to the Unitree DDS topics and appends
    data to in-memory lists in the DDS callback threads.
    This is very fast and avoids blocking the real-time callbacks.

    Data is serialized to binary format when the save_data() method is called.
    """

    def __init__(self, net_interface: str, log_level: int = logging.INFO):
        """
        Initialize the logger and subscribers.

        Args:
            net_interface: The net interface on which to subscribe via DDS
            log_level: The logging level for the logger.
        """
        self._setup_logging(log_level)

        self.joint_names = H12Real.REAL_JOINT_NAME_ORDER

        # --- Data Storage ---
        self.cmd_data: list[LowCmdMetrics] = []
        self.state_data: list[LowStateMetrics] = []
        self.cmd_lock = threading.Lock()
        self.state_lock = threading.Lock()

        # --- Initialize Unitree SDK ---
        try:
            ChannelFactoryInitialize(0, net_interface)
            self._log.info(f"ChannelFactory initialized on interface {net_interface}")
        except Exception as e:
            self._log.error(f"Failed to initialize ChannelFactory: {e}")
            raise

        # --- Command Subscriber ---
        self.lowcmd_subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.lowcmd_subscriber.Init(self._low_cmd_handler, 10)
        self._log.info("Subscribed to rt/lowcmd")

        # --- State Subscriber ---
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)
        self._log.info("Subscribed to rt/lowstate")

    def _setup_logging(self, log_level: int):
        """Setup logging configuration for this instance."""
        self._log = logging.getLogger(f"{__name__}.UnitreeLogger")
        self._log.setLevel(log_level)

    def _low_cmd_handler(self, msg: LowCmd_):
        """
        Fast callback for LowCmd messages.
        Converts message to a dataclass and appends to a list.
        """
        now = time.perf_counter()

        motor_cmds: dict[str, MotorCmdMetric] = {}
        for i in range(len(self.joint_names)):
            cmd = msg.motor_cmd[i]
            motor_cmds[self.joint_names[i]] = MotorCmdMetric(q=cmd.q, dq=cmd.dq, kp=cmd.kp, kd=cmd.kd)

        metrics = LowCmdMetrics(timestamp=now, motor_cmd=motor_cmds)

        with self.cmd_lock:
            self.cmd_data.append(metrics)

    def _low_state_handler(self, msg: LowState_):
        """
        Fast callback for LowState messages.
        Converts message to a dataclass and appends to a list.
        """
        now = time.perf_counter()

        motor_states: dict[str, MotorStateMetric] = {}
        for i in range(len(self.joint_names)):
            state = msg.motor_state[i]
            motor_states[self.joint_names[i]] = MotorStateMetric(q=state.q, dq=state.dq, tau=state.tau_est)

        metrics = LowStateMetrics(
            timestamp=now,
            base_orientation=list(msg.imu_state.quaternion),
            base_angular_vel=list(msg.imu_state.gyroscope),
            motor_state=motor_states,
        )

        with self.state_lock:
            self.state_data.append(metrics)

    def _save_cmd_data_binary(self, file_path: Path) -> None:
        with self.cmd_lock:
            num_records = len(self.cmd_data)
            if num_records == 0:
                self._log.warning("No command data to save.")
                return

            with file_path.open("wb") as f:
                # Write header: number of records and joints
                f.write(struct.pack("II", num_records, len(self.joint_names)))

                # Write joint names (for reconstruction)
                for joint_name in self.joint_names:
                    name_bytes = joint_name.encode("utf-8")
                    f.write(struct.pack("I", len(name_bytes)))
                    f.write(name_bytes)

                # Write data records
                for record in self.cmd_data:
                    f.write(struct.pack("d", record.timestamp))
                    for joint_name in self.joint_names:
                        cmd = record.motor_cmd[joint_name]
                        f.write(struct.pack("ffff", cmd.q, cmd.dq, cmd.kp, cmd.kd))

    def _save_state_data_binary(self, file_path: Path) -> None:
        with self.state_lock:
            num_records = len(self.state_data)
            if num_records == 0:
                self._log.warning("No state data to save.")
                return

            with file_path.open("wb") as f:
                # Write header: number of records and joints
                f.write(struct.pack("II", num_records, len(self.joint_names)))

                # Write joint names (for reconstruction)
                for joint_name in self.joint_names:
                    name_bytes = joint_name.encode("utf-8")
                    f.write(struct.pack("I", len(name_bytes)))
                    f.write(name_bytes)

                # Write data records
                for record in self.state_data:
                    f.write(struct.pack("d", record.timestamp))
                    f.write(struct.pack("ffff", *record.base_orientation))
                    f.write(struct.pack("fff", *record.base_angular_vel))
                    for joint_name in self.joint_names:
                        state = record.motor_state[joint_name]
                        f.write(struct.pack("fff", state.q, state.dq, state.tau))

    def save_data(self, log_dir: Path) -> None:
        """
        Save all recorded data to binary files.
        This method performs the I/O operations.
        """
        try:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            cmd_path = log_dir / "low_cmd.bin"
            state_path = log_dir / "low_state.bin"

            # --- Save Command Data ---
            self._log.info(f"Saving {len(self.cmd_data)} command records to {cmd_path}...")
            self._save_cmd_data_binary(cmd_path)
            self._log.info("Command data saved.")

            # --- Save State Data ---
            self._log.info(f"Saving {len(self.state_data)} state records to {state_path}...")
            self._save_state_data_binary(state_path)
            self._log.info("State data saved.")

        except Exception as e:
            self._log.error(f"Error saving data to {log_dir}: {e}")
            raise

    def close(self):
        """Cleans up resources (if any)."""
        self._log.info("Closing logger")
