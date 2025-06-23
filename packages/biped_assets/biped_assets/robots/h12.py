"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`H12_CFG`: H12 humanoid robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from ..assets import USD_PATHS

##
# Configuration
##

H12_12DOF = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATHS["h12"]["h12_12dof"],
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            # preserved order for sim2sim
            "left_hip_yaw_joint": 0.0,
            "left_hip_pitch_joint": -0.16,  # -0.28 -16 degrees
            "left_hip_roll_joint": 0.0,
            "left_knee_joint": 0.36,  # 0.79 45 degrees
            "left_ankle_pitch_joint": -0.2,  # -0.52 -30 degrees
            "left_ankle_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_pitch_joint": -0.16,  # -0.28 -16 degrees
            "right_hip_roll_joint": 0.0,
            "right_knee_joint": 0.36,  # 0.79 45 degrees
            "right_ankle_pitch_joint": -0.2,  # -0.52 -30 degrees
            "right_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
            ],
            effort_limit=220,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
            },
            armature=0.01,
            friction=0.0,
            min_delay=0,
            max_delay=5,
        ),
        "knees": DelayedPDActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit=360,
            velocity_limit=100.0,
            stiffness={
                ".*_knee_joint": 300.0,
            },
            damping={
                ".*_knee_joint": 4.0,
            },
            armature=0.01,
            friction=0.0,
            min_delay=0,
            max_delay=5,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=45,
            velocity_limit=100.0,
            stiffness={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 40.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 2.0,
            },
            armature=0.01,
            friction=0.0,
            min_delay=0,
            max_delay=5,
        ),
    },
)
H12_12DOF_MINIMAL = H12_12DOF.copy()

H12_12DOF_IDEAL = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATHS["h12"]["h12_12dof"],
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.16,  # -0.28 -16 degrees
            ".*_knee_joint": 0.36,  # 0.79 45 degrees
            ".*_ankle_pitch_joint": -0.2,  # -0.52 -30 degrees
            ".*_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
            ],
            effort_limit=220,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
            },
            armature=0.1,
        ),
        "knees": IdealPDActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit=360,
            velocity_limit=100.0,
            stiffness={
                ".*_knee_joint": 300.0,
            },
            damping={
                ".*_knee_joint": 4.0,
            },
            armature=0.1,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=45,
            velocity_limit=100.0,
            stiffness={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 40.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 2.0,
            },
            armature=0.1,
        ),
    },
)
H12_12DOF_MINIMAL = H12_12DOF_IDEAL.copy()


H12_27DOF = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATHS["h12"]["h12_27dof"],
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.16,  # -0.28 -16 degrees
            ".*_knee_joint": 0.36,  # 0.79 45 degrees
            ".*_ankle_pitch_joint": -0.2,  # -0.52 -30 degrees
            ".*_ankle_roll_joint": 0.0,
            "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.0,  # 0.28
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.0,  # 0.52
            ".*_wrist_yaw_joint": 0.0,
            ".*_wrist_roll_joint": 0.0,
            ".*_wrist_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                "torso_joint",
            ],
            effort_limit=220,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5,
                ".*_hip_roll_joint": 5,
                ".*_hip_pitch_joint": 5,
                "torso_joint": 5.0,
            },
            armature=0.001,
            friction=0.01,
        ),
        "knees": IdealPDActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit=360,
            velocity_limit=100.0,
            stiffness={
                ".*_knee_joint": 300.0,
            },
            damping={
                ".*_knee_joint": 10.0,
            },
            armature=0.001,
            friction=0.01,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=45,
            velocity_limit=100.0,
            stiffness={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 40.0,
            },
            damping={
                ".*_ankle_pitch_joint": 5.0,
                ".*_ankle_roll_joint": 5.0,
            },
            armature=0.001,
            friction=0.01,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
            ],
            effort_limit=75,
            velocity_limit=100.0,
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_joint": 40.0,
                ".*_wrist_yaw_joint": 40.0,
                ".*_wrist_roll_joint": 40.0,
                ".*_wrist_pitch_joint": 40.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 10.0,
                ".*_shoulder_roll_joint": 10.0,
                ".*_shoulder_yaw_joint": 10.0,
                ".*_elbow_joint": 10.0,
                ".*_wrist_yaw_joint": 10.0,
                ".*_wrist_roll_joint": 10.0,
                ".*_wrist_pitch_joint": 10.0,
            },
            armature=0.001,
            friction=0.01,
        ),
    },
)
