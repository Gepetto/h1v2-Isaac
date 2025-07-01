from simulator.sim_mujoco import MujocoSim
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"


class UnitreeSdk2Bridge:
    def __init__(self, scene_path, config):
        self.mujoco = MujocoSim(scene_path, config)

        self.decimation = config["decimation"]
        self.num_motor = len(self.mujoco.data.ctrl)

        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state.tick = 1
        self.low_state_puber = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_puber.Init()

        self.low_cmd_suber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_suber.Init(self.low_cmd_handler, 10)

        self.publish_low_state()

    def low_cmd_handler(self, msg: LowCmd_):
        for _ in range(self.decimation):
            state = self.mujoco.get_robot_state()
            qpos = state["q_pos"]
            qvel = state["q_vel"]
            torques = [
                msg.motor_cmd[i].tau
                + msg.motor_cmd[i].kp * (msg.motor_cmd[i].q - qpos[i])
                + msg.motor_cmd[i].kd * (msg.motor_cmd[i].dq - qvel[i])
                for i in range(self.num_motor)
            ]
            self.mujoco.sim_step(torques)
        self.publish_low_state()

    def publish_low_state(self):
        state = self.mujoco.get_robot_state()
        qpos = state["q_pos"]
        qvel = state["q_vel"]
        for i in range(self.num_motor):
            self.low_state.motor_state[i].q = qpos[i]
            self.low_state.motor_state[i].dq = qvel[i]
            self.low_state.motor_state[i].tau_est = 0.0

        self.low_state.imu_state.quaternion = state["base_orientation"]
        self.low_state.imu_state.gyroscope = state["base_angular_vel"]

        self.low_state_puber.Write(self.low_state)

    def close(self):
        self.mujoco.close()
