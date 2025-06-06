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
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm
)

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import (
    AdditiveUniformNoiseCfg as Unoise,
    GaussianNoiseCfg,
    NoiseModelWithAdditiveBiasCfg,
)

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from cat_envs.tasks.utils.cat.manager_constraint_cfg import ConstraintTermCfg as ConstraintTerm
import h1_extension.utils.cat.constraints as constraints
import h1_extension.utils.cat.curriculums as curriculums
import h1_extension.utils.mdp.observations as observations
import h1_extension.utils.mdp.events as events

from h1_assets.robots.h1v2 import H1_2_12DOF as ROBOT_CFG  # isort: skip

# ========================================================
# Scene Configuration
# ========================================================
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
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

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
        ),
    )


# ========================================================
# Actions Configuration
# ========================================================
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
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
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
       
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

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
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
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
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )


# ========================================================
# Constraints Configuration
# ========================================================
@configclass
class ConstraintsCfg:
    # Safety Hard constraints
    # upsidedown = ConstraintTerm(
    #     func=constraints.upsidedown, 
    #     max_p=1.0, 
    #     params={"limit": 0.0}
    # )
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
    # foot_contact_force = ConstraintTerm(
    #     func=constraints.foot_contact_force,
    #     max_p=1.0,
    #     params={"limit": 50.0, "names": [".*_FOOT"]},
    # )

    # Safety Soft constraints
    # joint_torque = ConstraintTerm(
    #     func=constraints.joint_torque,
    #     max_p=0.25,
    #     params={"limit": 4.0, "names": [".*_HAA", ".*_HFE", ".*_KFE"]},
    # )
    # joint_velocity = ConstraintTerm(
    #     func=constraints.joint_velocity,
    #     max_p=0.25,
    #     params={"limit": 16.0, "names": [".*_HAA", ".*_HFE", ".*_KFE"]},
    # )
    # joint_acceleration = ConstraintTerm(
    #     func=constraints.joint_acceleration,
    #     max_p=0.25,
    #     params={"limit": 800.0, "names": [".*_HAA", ".*_HFE", ".*_KFE"]},
    # )
    # action_rate = ConstraintTerm(
    #     func=constraints.action_rate,
    #     max_p=0.25,
    #     params={"limit": 90.0, "names": [".*_HAA", ".*_HFE", ".*_KFE"]},
    # )

    # Style constraints
    # base_orientation = ConstraintTerm(
    #     func=constraints.base_orientation, 
    #     max_p=0.25, 
    #     params={"limit": 0.1}
    # )
    # air_time = ConstraintTerm(
    #     func=constraints.air_time,
    #     max_p=0.25,
    #     params={"limit": 0.25, "names": [".*_FOOT"], "velocity_deadzone": 0.0},
    # )
    # switch these two constraints to change the robot's gait. 
    # The first is for walking, the second for jumping.
    # one_foot_contact = ConstraintTerm(
    #     func=constraints.n_foot_contact,
    #     max_p=0.25,
    #     params={
    #         "names": [".*_FOOT"],
    #         "number_of_desired_feet": 1,
    #         "min_command_value": 0.0,
    #     },
    # )
    # mod_n_foot_contact = ConstraintTerm(
    #     func=constraints.mod_n_foot_contact,
    #     max_p=0.25,
    #     params={
    #         "names": [".*_FOOT"],
    #         "number_of_desired_feet": 2,
    #         "min_command_value": 0.0,
    #     },
    # )
    # HAA_position = ConstraintTerm(
    #     func=constraints.joint_range,
    #     max_p=0.25,
    #     params={"limit": 0.3, "names": [".*_HAA"]},
    # )
    # HFE_position = ConstraintTerm(
    #     func=constraints.joint_range,
    #     max_p=0.25,
    #     params={"limit": 0.5, "names": [".*_HFE"]},
    # )
    # KFE_position = ConstraintTerm(
    #     func=constraints.joint_range,
    #     max_p=0.25,
    #     params={"limit": 0.5, "names": [".*_KFE"]},
    # )


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
    pass
    # Soft constraints
    # joint_torque = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "joint_torque",
    #         "num_steps": 24 * 1000,
    #         "init_max_p": 0.25,
    #     },
    # )
    # joint_velocity = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "joint_velocity",
    #         "num_steps": 24 * 1000,
    #         "init_max_p": 0.25,
    #     },
    # )
    # joint_acceleration = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "joint_acceleration",
    #         "num_steps": 24 * 1000,
    #         "init_max_p": 0.25,
    #     },
    # )
    # action_rate = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={"term_name": "action_rate", "num_steps": 24 * 1000, "init_max_p": 0.25},
    # )

    # # Style constraints
    # base_orientation = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "base_orientation",
    #         "num_steps": 24 * 1000,
    #         "init_max_p": 0.25,
    #     },
    # )
    # air_time = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={"term_name": "air_time", 
    #             "num_steps": 24 * 1000, 
    #             "init_max_p": 0.25},
    # )
    # # switch these two curriculums to change the robot's gait. 
    # # The first is for walking, the second for jumping.
    # one_foot_contact = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "one_foot_contact",
    #         "num_steps": 24 * 1000,
    #         "init_max_p": 0.25,
    #     },
    # )
    # # mod_n_foot_contact = CurrTerm(
    # #     func=curriculums.modify_constraint_p,
    # #     params={
    # #         "term_name": "mod_n_foot_contact",
    # #         "num_steps": 24 * 1000,
    # #         "init_max_p": 0.25,
    # #     },
    # # )
    # HAA_position = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "HAA_position",
    #         "num_steps": 24 * 1000,
    #         "init_max_p": 0.25,
    #     },
    # )
    # HFE_position = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "HFE_position",
    #         "num_steps": 24 * 1000,
    #         "init_max_p": 0.25,
    #     },
    # )
    # KFE_position = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "KFE_position",
    #         "num_steps": 24 * 1000,
    #         "init_max_p": 0.25,
    #     },
    # )


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
        self.observations.policy.enable_corruption = False
        # set velocity command
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)