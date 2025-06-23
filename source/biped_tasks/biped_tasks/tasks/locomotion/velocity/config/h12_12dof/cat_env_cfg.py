# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
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

from biped_tasks.utils.history.manager_term_cfg import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
)

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from biped_tasks.utils.cat.manager_constraint_cfg import ConstraintTermCfg as ConstraintTerm
import biped_tasks.utils.cat.constraints as constraints
import biped_tasks.utils.cat.curriculums as curriculums
import biped_tasks.utils.mdp.commands as commands
import biped_tasks.utils.mdp.events as events

from biped_assets.robots.h12 import H12_12DOF as ROBOT_CFG  # isort: skip
from biped_tasks.utils.mdp.terrains import ROUGH_TERRAINS_CFG  # isort: skip


VELOCITY_DEADZONE = 0.2
MAX_CURRICULUM_ITERATIONS = 1000


# ========================================================
# Scene Configuration
# ========================================================
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="exts/cat_envs/cat_envs/assets/materials/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"exts/cat_envs/cat_envs/assets/materials/kloofendal_43d_clear_puresky_4k.hdr",
        ),
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
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.5, 0.5)
        ),
        velocity_deadzone=VELOCITY_DEADZONE
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
        scale=0.5,
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
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
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
                                                                preserve_order=True)})
        actions = ObsTerm(func=mdp.last_action)
       
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 6
            self.history_step = 1

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
    scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "recompute_inertia": False,
        },
    )
    move_base_com = EventTerm(
        func=events.randomize_body_coms,
        mode="startup",
        params={
            "max_displacement": 0.03,
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )
    randomize_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.01, 0.1),
            "operation": "abs",
            "distribution": "uniform",
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
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.8, 1.2),
        },
    )
    
    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={"velocity_range": {"x": (-1.0, 1.0), 
                                   "y": (-1.0, 1.0),
                                   "z": (-0.2, 0.2),
                                   "yaw": (-0.5, 0.5), 
                                   "pitch": (-0.5, 0.5), 
                                   "roll": (-0.5, 0.5)}},
    )


# ========================================================
# Rewards Configuration
# ========================================================
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )


# ========================================================
# Constraints Configuration
# ========================================================
@configclass
class ConstraintsCfg:
    # Safety Hard constraints
    contact = ConstraintTerm(
        func=constraints.contact,
        max_p=1.0,
        params={"names": [".*_hip_yaw_link",
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
                            ".*_wrist_pitch_link"]},
    )

    # Safety Soft constraints
    foot_contact_force = ConstraintTerm(
        func=constraints.foot_contact_force,
        max_p=0.25,
        params={"limit": 700.0, "names": [".*_ankle_roll_link"]},
    )
    hip_joint_torque = ConstraintTerm(
        func=constraints.joint_torque,
        max_p=0.25,
        params={"limit": 220.0, "names": [".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint"]},
    )
    knee_joint_torque = ConstraintTerm(
        func=constraints.joint_torque,
        max_p=0.25,
        params={"limit": 360.0, "names": [".*_knee_joint"]},
    )
    ankle_joint_torque = ConstraintTerm(
        func=constraints.joint_torque,
        max_p=0.25,
        params={"limit": 45.0, "names": [".*_ankle_pitch_joint", ".*_ankle_roll_joint"]},
    )
    joint_velocity = ConstraintTerm(
        func=constraints.joint_velocity,
        max_p=0.25,
        params={"limit": 16.0, "names": [".*"]},
    )
    joint_acceleration = ConstraintTerm(
        func=constraints.joint_acceleration,
        max_p=0.25,
        params={"limit": 800.0, "names": [".*"]},
    )
    action_rate = ConstraintTerm(
        func=constraints.action_rate,
        max_p=0.25,
        params={"limit": 90.0, "names": [".*"]},
    )

    # Style constraints
    base_orientation = ConstraintTerm(
        func=constraints.base_orientation, 
        max_p=0.25, 
        params={"limit": 0.2}
    )
    foot_contact = ConstraintTerm(
        func=constraints.foot_contact,
        max_p=0.25,
        params={"names": [".*_ankle_roll_link"]},
    )
    air_time = ConstraintTerm(
        func=constraints.air_time,
        max_p=0.25,
        params={"limit": 0.5, 
                "names": [".*_ankle_roll_link"], 
                "velocity_deadzone": VELOCITY_DEADZONE},
    )
    no_move = ConstraintTerm(
        func=constraints.no_move,
        max_p=0.25,
        params={
            "names": [".*"],
            "velocity_deadzone": VELOCITY_DEADZONE,
            "joint_vel_limit": 1.0,
        },
    )
    hip_roll_position = ConstraintTerm(
        func=constraints.joint_range,
        max_p=0.25,
        params={"limit": 0.1, "names": [".*_hip_roll_joint"]},
    )
    hip_yaw_position = ConstraintTerm(
        func=constraints.joint_range,
        max_p=0.25,
        params={"limit": 0.1, "names": [".*_hip_yaw_joint"]},
    )
    knee_position = ConstraintTerm(
        func=constraints.joint_range,
        max_p=0.25,
        params={"limit": 1.0, "names": [".*_knee_joint"]},
    )
    ankle_roll_position = ConstraintTerm(
        func=constraints.joint_range,
        max_p=0.25,
        params={"limit": 0.1, "names": [".*_ankle_roll_joint"]},
    )
    ankle_pitch_position = ConstraintTerm(
        func=constraints.joint_range,
        max_p=0.25,
        params={"limit": 0.5, "names": [".*_ankle_pitch_joint"]},
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
    foot_contact_force = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={"term_name": "foot_contact_force", 
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS, 
                "init_max_p": 0.25},
    )
    hip_joint_torque = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "hip_joint_torque",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    knee_joint_torque = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "knee_joint_torque",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    ankle_joint_torque = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "ankle_joint_torque",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    joint_velocity = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_velocity",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    joint_acceleration = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_acceleration",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    action_rate = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={"term_name": "action_rate", 
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS, 
                "init_max_p": 0.25},
    )

    # Style constraints
    base_orientation = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "base_orientation",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    foot_contact = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={"term_name": "foot_contact", 
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS, 
                "init_max_p": 0.25},
    )
    air_time = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={"term_name": "air_time", 
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS, 
                "init_max_p": 0.25},
    )
    no_move = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "no_move",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    hip_roll_position = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "hip_roll_position",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    hip_yaw_position = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "hip_yaw_position",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    knee_position = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "knee_position",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    ankle_roll_position = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "ankle_roll_position",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    ankle_pitch_position = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "ankle_pitch_position",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
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
    constraints: ConstraintsCfg = ConstraintsCfg()
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
        # self.events.push_robot = self.events.physics_material = self.events.scale_mass = self.events.move_base_com = self.events.randomize_joint_parameters = self.events.base_external_force_torque = None
        # self.observations.policy.enable_corruption = False
        # set velocity command
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, .0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0., 0.)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

