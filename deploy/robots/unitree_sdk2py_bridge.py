import threading
import time

import mujoco
import mujoco.viewer
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"


class UnitreeSdk2Bridge:
    def __init__(self, scene_path, dt):
        self.mj_model = mujoco.MjModel.from_xml_path(scene_path)
        self.mj_model.opt.integrator = 3
        self.mj_model.opt.timestep = dt
        self.mj_data = mujoco.MjData(self.mj_model)

        # self.num_motor = self.mj_model.nu
        self.num_motor = len(self.mj_data.ctrl)
        self.dt = self.mj_model.opt.timestep

        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state.tick = 1
        self.low_state_puber = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_puber.Init()

        self.low_cmd_suber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_suber.Init(self.low_cmd_handler, 10)

        self.sim_lock = threading.Lock()
        self.close_event = threading.Event()
        self.sim_thread = threading.Thread(target=self.run_sim, args=(self.close_event,))
        self.viewer_thread = threading.Thread(target=self.run_render, args=(self.close_event,))
        self.sim_thread.start()
        self.viewer_thread.start()

    def low_cmd_handler(self, msg: LowCmd_):
        if self.mj_data is None:
            return
        for i in range(self.num_motor):
            self.mj_data.ctrl[i] = (
                msg.motor_cmd[i].tau
                + msg.motor_cmd[i].kp * (msg.motor_cmd[i].q - self.mj_data.qpos[7 + i])
                + msg.motor_cmd[i].kd * (msg.motor_cmd[i].dq - self.mj_data.qvel[6 + i])
            )

    def publish_low_state(self):
        if self.mj_data is None:
            return
        for i in range(self.num_motor):
            self.low_state.motor_state[i].q = self.mj_data.qpos[7 + i]
            self.low_state.motor_state[i].dq = self.mj_data.qvel[6 + i]
            self.low_state.motor_state[i].tau_est = 0.0

        self.low_state.imu_state.quaternion = self.mj_data.qpos[3:7]
        self.low_state.imu_state.gyroscope = self.mj_data.qvel[3:6]

        self.low_state_puber.Write(self.low_state)

    def run_sim(self, close_event):
        while not close_event.is_set():
            with self.sim_lock:
                mujoco.mj_step(self.mj_model, self.mj_data)
                self.publish_low_state()
            time.sleep(self.dt)
        close_event.set()

    def run_render(self, close_event):
        viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        while viewer.is_running() and not close_event.is_set():
            with self.sim_lock:
                viewer.sync()
            time.sleep(0.02)  # 50 Hz
        close_event.set()
        viewer.close()

    def close(self):
        self.close_event.set()
        self.sim_thread.join()
        self.viewer_thread.join()
