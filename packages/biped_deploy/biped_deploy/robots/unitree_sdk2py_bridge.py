import threading

from simulator.sim_mujoco import MujocoSim
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.thread import RecurrentThread

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"


class UnitreeSdk2Bridge:
    def __init__(self, config):
        scene_path = config["mujoco"]["scene_path"]
        config["mujoco"]["real_time"] = True

        self.simulator = MujocoSim(scene_path, config)

        self.num_motor = len(config["joints"])
        self.torques = [0.0] * self.num_motor

        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state.tick = 1
        self.low_state_puber = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_puber.Init()

        self.low_cmd_suber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_suber.Init(self.low_cmd_handler, 10)

        self.torques_lock = threading.Lock()
        self.close_event = threading.Event()
        self.sim_thread = threading.Thread(target=self.run_sim, args=(self.close_event,))

        self.state_lock = threading.Lock()
        self.state_thread = RecurrentThread(interval=config["control_dt"] / 5, target=self.publish_low_state)

        self.sim_thread.start()
        self.state_thread.Start()

    def get_controller_command(self):
        return self.simulator.get_controller_command()

    def low_cmd_handler(self, msg: LowCmd_):
        state = self.simulator.get_robot_sim_state()
        qpos = state["qpos"]
        qvel = state["qvel"]

        with self.torques_lock:
            for i in range(self.num_motor):
                motor = msg.motor_cmd[i]
                self.torques[i] = motor.tau + motor.kp * (motor.q - qpos[i]) + motor.kd * (motor.dq - qvel[i])

    def publish_low_state(self):
        state = self.simulator.get_robot_sim_state()
        qpos = state["qpos"]
        qvel = state["qvel"]

        with self.state_lock:
            for i in range(self.num_motor):
                motor_state = self.low_state.motor_state[i]
                motor_state.q = qpos[i]
                motor_state.dq = qvel[i]
                motor_state.tau_est = 0.0

            self.low_state.imu_state.quaternion = state["base_orientation"]
            self.low_state.imu_state.gyroscope = state["base_angular_vel"]

            self.low_state_puber.Write(self.low_state)

    def run_sim(self, close_event):
        while not close_event.is_set():
            with self.torques_lock:
                torques = self.torques.copy()

            self.simulator.sim_step(torques)

    def close(self, log_dir=None):
        self.simulator.close(log_dir)
        self.close_event.set()
        self.sim_thread.join()
