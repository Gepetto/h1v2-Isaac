import logging
import numpy as np
import struct
import threading
import time
from pathlib import Path
from tqdm import tqdm

import mujoco
import mujoco.viewer

try:
    import imageio.v3 as iio
except ImportError:
    iio = None
    logging.warning("imageio not found. Video recording will fail. Please install: pip install imageio imageio-ffmpeg")

logger = logging.getLogger(__name__)


def load_state_log_binary(path: Path) -> dict:
    """
    Load state data from binary file.

    Binary format structure:
    - Header: num_records (uint32), num_joints (uint32)
    - Joint names: for each joint: length (uint32) + name_bytes
    - Data records: for each record:
        - timestamp (double)
        - base_orientation (4 floats)
        - base_angular_vel (3 floats)
        - for each joint: q, dq, tau (3 floats each)
    """
    path = Path(path)
    if not path.exists():
        err_msg = f"Log file not found: {path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    with path.open("rb") as f:
        header_data = f.read(8)
        if len(header_data) < 8:
            err_msg = "Invalid binary file: incomplete header"
            raise ValueError(err_msg)

        num_records, num_joints = struct.unpack("II", header_data)

        joint_names = []
        for _ in range(num_joints):
            name_len_data = f.read(4)
            if len(name_len_data) < 4:
                err_msg = "Invalid binary file: incomplete joint name length"
                raise ValueError(err_msg)
            name_len = struct.unpack("I", name_len_data)[0]

            name_bytes = f.read(name_len)
            if len(name_bytes) < name_len:
                err_msg = "Invalid binary file: incomplete joint name"
                raise ValueError(err_msg)
            joint_names.append(name_bytes.decode("utf-8"))

        timestamps = np.zeros(num_records)
        base_orientations = np.zeros((num_records, 4))
        base_ang_vels = np.zeros((num_records, 3))
        q_pos = np.zeros((num_records, num_joints))
        q_vel = np.zeros((num_records, num_joints))
        tau_est = np.zeros((num_records, num_joints))

        for i in range(num_records):
            timestamp_data = f.read(8)
            if len(timestamp_data) < 8:
                err_msg = f"Invalid binary file: incomplete timestamp at record {i}"
                raise ValueError(err_msg)
            timestamps[i] = struct.unpack("d", timestamp_data)[0]

            orientation_data = f.read(4 * 4)
            if len(orientation_data) < 16:
                err_msg = f"Invalid binary file: incomplete orientation at record {i}"
                raise ValueError(err_msg)
            base_orientations[i] = struct.unpack("4f", orientation_data)

            ang_vel_data = f.read(4 * 3)
            if len(ang_vel_data) < 12:
                err_msg = f"Invalid binary file: incomplete angular velocity at record {i}"
                raise ValueError(err_msg)
            base_ang_vels[i] = struct.unpack("3f", ang_vel_data)

            for j in range(num_joints):
                motor_data = f.read(4 * 3)
                if len(motor_data) < 12:
                    err_msg = f"Invalid binary file: incomplete motor data at record {i}, joint {j}"
                    raise ValueError(err_msg)
                q_val, dq_val, tau_val = struct.unpack("3f", motor_data)
                q_pos[i, j] = q_val
                q_vel[i, j] = dq_val
                tau_est[i, j] = tau_val

        remaining_data = f.read()
        if remaining_data:
            warn_msg = f"Unexpected extra data at end of file: {len(remaining_data)} bytes"
            logger.warning(warn_msg)

        return {
            "timestamp": timestamps,
            "base_orientation": base_orientations,
            "base_angular_vel": base_ang_vels,
            "q_pos": q_pos,
            "q_vel": q_vel,
            "tau_est": tau_est,
            "joint_names": joint_names,
            "nb_motors": num_joints,
        }


class ThreadedReplayer:
    def __init__(self, model, data, state_data, config):
        self.model = model
        self.data = data
        self.state_data = state_data
        self.render_dt = config.get("render_dt", 1.0 / 60.0)

        self.video_path = config.get("video_path")
        self.is_video_mode = self.video_path is not None

        timestamps = self.state_data["timestamp"]
        if len(timestamps) > 1:
            log_dts = np.diff(timestamps)
            avg_log_dt = np.median(log_dts)
        elif len(timestamps) == 1:
            logger.warning("Only one log frame, assuming log frequency of 50Hz (dt=0.02s).")
            avg_log_dt = 0.02
        else:
            avg_log_dt = self.render_dt
            logger.warning("No log frames, using render_dt for frame rate.")

        self.log_frequency = 1.0 / avg_log_dt
        self.frame_rate = round(1.0 / self.render_dt)

        if self.is_video_mode:
            info_msg = f"Video will be saved at the recorded state frequency: {self.frame_rate} FPS (Avg Log DT: {avg_log_dt:.4f}s)"
            logger.info(info_msg)

        self.renderer = None
        if self.is_video_mode:
            if iio is None:
                err_msg = "Cannot use --video mode: imageio library is not installed."
                raise RuntimeError(err_msg)
            self.renderer = mujoco.Renderer(model)
            self.frames = []

        self.sim_lock = threading.Lock()
        self.close_event = threading.Event()

        self._setup_joint_mapping()

        self.viewer_thread = None
        self.replay_thread = None

    def _setup_joint_mapping(self):
        log_joint_names = self.state_data["joint_names"]

        self.model_qpos_indices = []
        self.model_qvel_indices = []

        for name in log_joint_names:
            try:
                jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jnt_id == -1:
                    err_msg = f"Joint '{name}' not found"
                    raise ValueError(err_msg)

                qpos_adr = self.model.jnt_qposadr[jnt_id]
                dof_adr = self.model.jnt_dofadr[jnt_id]

                self.model_qpos_indices.append(qpos_adr)
                self.model_qvel_indices.append(dof_adr)

            except ValueError as e:
                warn_msg = f"Skipping joint: {e}"
                logger.warning(warn_msg)

        self.model_qpos_indices = np.array(self.model_qpos_indices)
        self.model_qvel_indices = np.array(self.model_qvel_indices)

    def run_render(self):
        try:
            if not self.is_video_mode:
                with self.sim_lock:
                    viewer = mujoco.viewer.launch_passive(self.model, self.data)

                logger.info("Viewer thread started (GUI)")
                while viewer.is_running() and not self.close_event.is_set():
                    with self.sim_lock:
                        viewer.sync()
                    time.sleep(self.render_dt)

                viewer.close()
                logger.info("Viewer thread finished")
            else:
                logger.info("Video Writer thread started (Off-Screen)")

                while not self.close_event.is_set():
                    time.sleep(0.1)

                info_msg = f"Replay complete. Writing {len(self.frames)} frames to video..."
                logger.info(info_msg)

                if self.frames:
                    iio.imwrite(self.video_path, self.frames, fps=self.frame_rate, codec="libx264", quality=8)
                    info_msg = f"Video saved successfully to {self.video_path}"
                    logger.info(info_msg)
                else:
                    logger.warning("No frames recorded for video.")

        except Exception as e:
            err_msg = f"Error in viewer/renderer thread: {e}"
            logger.error(err_msg)

    def run_replay(self):
        try:
            timestamps = self.state_data["timestamp"]
            num_frames = len(timestamps)
            start_time = time.perf_counter()

            info_msg = f"Starting replay of {num_frames} frames (render every {self.render_dt:.3f}s)..."
            logger.info(info_msg)

            next_render_time = timestamps[0]  # Next time we should render
            render_count = 0

            with tqdm(total=num_frames, desc="Replay Progress", unit="frame") as pbar:
                frame_idx = 0

                while not self.close_event.is_set() and frame_idx < num_frames:
                    log_time = timestamps[frame_idx]
                    log_time_elapsed = log_time - timestamps[0]
                    real_time_elapsed = time.perf_counter() - start_time
                    time_to_wait = log_time_elapsed - real_time_elapsed

                    if time_to_wait > 0:
                        time.sleep(time_to_wait)

                    # --- Update simulation state ---
                    with self.sim_lock:
                        self.data.qpos[3:7] = self.state_data["base_orientation"][frame_idx]
                        self.data.qpos[self.model_qpos_indices] = self.state_data["q_pos"][frame_idx]

                        self.data.qvel[0:3] = [0, 0, 0]
                        self.data.qvel[3:6] = self.state_data["base_angular_vel"][frame_idx]
                        self.data.qvel[self.model_qvel_indices] = self.state_data["q_vel"][frame_idx]

                        mujoco.mj_forward(self.model, self.data)

                        # --- Render only at fixed intervals ---
                        if self.is_video_mode and self.renderer and log_time >= next_render_time:
                            self.renderer.update_scene(self.data)
                            frame = self.renderer.render()
                            self.frames.append(frame)
                            render_count += 1
                            next_render_time += self.render_dt  # Schedule next render

                    frame_idx += 1
                    pbar.update(1)

            info_msg = f"Replay thread finished. Rendered {render_count} frames (target dt={self.render_dt:.3f}s)"
            logger.info(info_msg)

        except Exception as e:
            err_msg = f"Error in replay thread: {e}"
            logger.error(err_msg)

    def start(self):
        self.viewer_thread = threading.Thread(target=self.run_render)
        self.viewer_thread.daemon = True
        self.viewer_thread.start()

        time.sleep(0.5)
        self.run_replay()

    def close(self):
        self.close_event.set()
        if self.viewer_thread and self.viewer_thread.is_alive():
            self.viewer_thread.join(timeout=5.0)

        if self.renderer:
            self.renderer.close()
