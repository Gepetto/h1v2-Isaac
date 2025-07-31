# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

import biped_tasks.utils.mdp.commands as commands
import biped_tasks.utils.mdp.events as events

from biped_assets.robots.h12 import H12_12DOF_IDEAL as ROBOT_CFG  # isort: skip
from biped_tasks.utils.mdp.terrains import ROUGH_TERRAINS_CFG  # isort: skip

MAX_CURRICULUM_ITERATIONS = 5000

# ========================================================
# Scene Configuration
# ========================================================
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        #terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=750.0),
    )

# ========================================================
# Commands Configuration
# ========================================================
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = commands.UniformVelocityCommandWithDeadzoneCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness= 1.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
        ),
        velocity_deadzone=0.0
    )

# ========================================================
# Actions Configuration
# ========================================================
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[   "left_hip_yaw_joint",
                        "left_hip_pitch_joint",
                        "left_hip_roll_joint",
                        "left_knee_joint",
                        "left_ankle_pitch_joint",
                        "left_ankle_roll_joint",
                        "right_hip_yaw_joint",
                        "right_hip_pitch_joint",
                        "right_hip_roll_joint",
                        "right_knee_joint",
                        "right_ankle_pitch_joint",
                        "right_ankle_roll_joint",],
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )

# ========================================================
# Observations Configuration
# ========================================================
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            noise=Unoise(n_min=-0.01, n_max=0.01), 
                            params={"asset_cfg": SceneEntityCfg("robot", 
                                                                joint_names=[   "left_hip_yaw_joint",
                                                                                "left_hip_pitch_joint",
                                                                                "left_hip_roll_joint",
                                                                                "left_knee_joint",
                                                                                "left_ankle_pitch_joint",
                                                                                "left_ankle_roll_joint",
                                                                                "right_hip_yaw_joint",
                                                                                "right_hip_pitch_joint",
                                                                                "right_hip_roll_joint",
                                                                                "right_knee_joint",
                                                                                "right_ankle_pitch_joint",
                                                                                "right_ankle_roll_joint",],
                                                                preserve_order=True)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            noise=Unoise(n_min=-1.5, n_max=1.5), 
                            params={"asset_cfg": SceneEntityCfg("robot", 
                                                                joint_names=[   "left_hip_yaw_joint",
                                                                                "left_hip_pitch_joint",
                                                                                "left_hip_roll_joint",
                                                                                "left_knee_joint",
                                                                                "left_ankle_pitch_joint",
                                                                                "left_ankle_roll_joint",
                                                                                "right_hip_yaw_joint",
                                                                                "right_hip_pitch_joint",
                                                                                "right_hip_roll_joint",
                                                                                "right_knee_joint",
                                                                                "right_ankle_pitch_joint",
                                                                                "right_ankle_roll_joint",],
                                                                preserve_order=True)},
                                                                scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
       
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 6

    # observation groups
    policy: PolicyCfg = PolicyCfg()

# ========================================================
# Events Configuration
# ========================================================
@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 1.25),
            "dynamic_friction_range": (0.1, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={"velocity_range": {"x": (-1.0, 1.0), 
                                   "y": (-1.0, 1.0),
                                }},
    )

# ========================================================
# Rewards Configuration
# ========================================================
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Velocity-tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # Locomotion rewards
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces",
                body_names=["left_ankle_roll_link",
                            "right_ankle_roll_link"]),
            "command_name": "base_velocity",
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces",
                body_names=["left_ankle_roll_link",
                            "right_ankle_roll_link"]),
            "asset_cfg": SceneEntityCfg("robot",
                body_names=["left_ankle_roll_link",
                            "right_ankle_roll_link"]),
        },
    )

    # Root penalties
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.2,
        params={
            "target_height":1.0,
        }
    )

    # Joint penalties
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
    )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-3,
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-07,
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot",
                joint_names=["left_hip_yaw_joint",
                             "right_hip_yaw_joint",
                             "left_hip_roll_joint",
                             "right_hip_roll_joint"]),
        },
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot",
                joint_names=["left_ankle_roll_joint",
                             "right_ankle_roll_joint",
                             "left_ankle_pitch_joint",
                             "right_ankle_pitch_joint"]),
        },
    )
    joint_pos_limits_ankle = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot",
                joint_names=[".*_ankle_roll_joint",
                             ".*_ankle_pitch_joint"]),
        },
    )
    joint_pos_limits_hip = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot",
                joint_names=[".*_hip_yaw_joint",
                             ".*_hip_roll_joint"]),
        },
    )

    # Action penalties
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # Contact sensor
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-1.0e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces",
                body_names=["left_ankle_roll_link",
                            "right_ankle_roll_link"]),
            "threshold": 800.0,
        },
    )

    # General
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0
    )

# ========================================================
# Terminations Configuration
# ========================================================
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
                                             ".*_hip_yaw_link",
                                             ".*_hip_roll_link",
                                             ".*_hip_pitch_link",
                                             ".*_knee_link",
                                             "torso_link",
                                             "pelvis",
                                             ".*_shoulder_pitch_link",
                                             ".*_shoulder_roll_link",
                                             ".*_shoulder_yaw_link",
                                             ".*_elbow_link",
                                             ".*_wrist_yaw_link",
                                             ".*_wrist_roll_link",
                                             ".*_wrist_pitch_link"]
            ),
            "threshold": 1.0,
        },
    )

# ========================================================
# Curriculum Configuration
# ========================================================
@configclass
class CurriculumCfg:
    # Soft constraints
    flat_orientation = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "flat_orientation",
                "weight":-1.0,
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS},
    )
    joint_torques_l2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_torques_l2",
                "weight":-1.0e-5,
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS},
    )
    joint_vel_l2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel_l2",
                "weight":-1.0e-3,
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS},
    )
    dof_acc_l2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "dof_acc_l2",
                "weight":-1.0e-07,
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS},
    )
    joint_deviation_hip = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_deviation_hip",
                "weight":-0.2,
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS},
    )
    joint_deviation_ankle = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_deviation_ankle",
                "weight":-0.2,
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS},
    )
    joint_pos_limits_ankle = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_pos_limits_ankle",
                "weight":-0.2,
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS},
    )
    joint_pos_limits_hip = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_pos_limits_hip",
                "weight":-0.2,
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS},
    )
    contact_forces = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "contact_forces",
                "weight":-1.0e-3,
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS},
    )
    feet_air_time = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "feet_air_time",
                "weight":0.75,
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS},
    )
    feet_slide = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "feet_slide",
                "weight":-0.25,
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS},
    )
    base_height_l2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "base_height_l2",
                "weight":-0.2,
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS},
    )

# ========================================================
# Environment Configuration
# ========================================================
@configclass
class H12_12dof_EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

class H12_12dof_EnvCfg_PLAY(H12_12dof_EnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 100
        self.scene.env_spacing = 2.5
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        # disable randomization for play
        self.events.push_robot = self.events.physics_material = self.events.scale_mass = self.events.move_base_com = self.events.randomize_joint_parameters = self.events.base_external_force_torque = None
        self.observations.policy.enable_corruption = False
        # set velocity command
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)