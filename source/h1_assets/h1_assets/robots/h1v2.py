"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`H1_2_CFG`: H1 humanoid robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

# THIS CONFIG IS UNSTABLE, DO NOT USE
H1_2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/cperrot/h1v2-Isaac/source/h1_assets/h1_assets/H12_handless_simplified_feet.usd",
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
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            # solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            # Legs:
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_pitch_joint": -0.28,  # -16 degrees
            ".*_hip_roll_joint": 0.0,
            ".*_knee_joint": 0.79,  # 45 degrees
            ".*_ankle_pitch_joint": -0.52,  # -30 degrees
            ".*_ankle_roll_joint": 0.0,
            "torso_joint": 0.0,
            # Arms:
            ".*_shoulder_pitch_joint": 0.28,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.52,
            ".*_wrist_roll_joint": 0.0,
            ".*_wrist_pitch_joint": 0.0,
            ".*_wrist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_knee_joint",
                "torso_joint",
            ],
            # Using a fallback effort limit (e.g., max value among the joints)
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 250.0,
                ".*_hip_pitch_joint": 250.0,
                ".*_hip_roll_joint": 250.0,
                ".*_knee_joint": 400.0,
                "torso_joint": 600.0,
            },
            damping={
                ".*_hip_yaw_joint": 6.5,
                ".*_hip_pitch_joint": 6.5,
                ".*_hip_roll_joint": 6.5,
                ".*_knee_joint": 10.0,
                "torso_joint": 6.0,
            },
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=100,  # Fallback effort limit
            velocity_limit=100.0,
            stiffness={
                ".*_ankle_pitch_joint": 80.0,
                ".*_ankle_roll_joint": 40.0,
            },
            damping={
                ".*_ankle_pitch_joint": 1.5,
                ".*_ankle_roll_joint": 0.4,
            },
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit=300,  # Fallback effort limit
            velocity_limit=100.0,
            stiffness={
                ".*_shoulder_pitch_joint": 250.0,
                ".*_shoulder_roll_joint": 400.0,
                ".*_shoulder_yaw_joint": 80.0,
                ".*_elbow_joint": 80.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 6.0,
                ".*_shoulder_roll_joint": 10.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
            },
        ),
        "wrists": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit=300,  # Fallback effort limit
            velocity_limit=100.0,
            stiffness={
                ".*_wrist_roll_joint": 20.0,
                ".*_wrist_pitch_joint": 20.0,
                ".*_wrist_yaw_joint": 20.0,
            },
            damping={
                ".*_wrist_roll_joint": 0.5,
                ".*_wrist_pitch_joint": 0.5,
                ".*_wrist_yaw_joint": 0.5,
            },
        ),
    },
)

# THIS CONFIG MAKE NOT SENSE, DO NOT USE
H1_2_SHADOW_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/cperrot/h1v2-Isaac/source/h1_assets/h1_assets/H12_handless_simplified_feet.usd",
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
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            # solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            # Legs:
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_pitch_joint": -0.28,  # -16 degrees
            ".*_hip_roll_joint": 0.0,
            ".*_knee_joint": 0.79,  # 45 degrees
            ".*_ankle_pitch_joint": -0.52,  # -30 degrees
            ".*_ankle_roll_joint": 0.0,
            "torso_joint": 0.0,
            # Arms:
            ".*_shoulder_pitch_joint": 0.28,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.52,
            ".*_wrist_roll_joint": 0.0,
            ".*_wrist_pitch_joint": 0.0,
            ".*_wrist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            # For H1_2 the leg actuator controls the hip and knee joints along with torso rotation.
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_knee_joint",
                "torso_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_knee_joint": 300.0,
                "torso_joint": 200.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 23.0,
                ".*_hip_pitch_joint": 23.0,
                ".*_hip_roll_joint": 23.0,
                ".*_knee_joint": 14.0,
                "torso_joint": 23.0,
            },
            stiffness=0,
            damping=0,
        ),
        "feet": IdealPDActuatorCfg(
            # H1_2 now has two ankle degrees of freedom.
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit={
                ".*_ankle_pitch_joint": 60.0,
                ".*_ankle_roll_joint": 40.0,
            },
            velocity_limit={
                ".*_ankle_pitch_joint": 9.0,
                ".*_ankle_roll_joint": 9.0,
            },
            stiffness=0,
            damping=0,
        ),
        "arms": IdealPDActuatorCfg(
            # Arms now include additional wrist joints.
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 18.0,
                ".*_elbow_joint": 18.0,
            },
            velocity_limit={
                ".*_shoulder_pitch_joint": 9.0,
                ".*_shoulder_roll_joint": 9.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*_elbow_joint": 20.0,
            },
            stiffness=0,
            damping=0,
        ),
        "wrists": IdealPDActuatorCfg(
            # Arms now include additional wrist joints.
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit={
                ".*_wrist_roll_joint": 19.0,
                ".*_wrist_pitch_joint": 19.0,
                ".*_wrist_yaw_joint": 19.0,
            },
            velocity_limit={
                ".*_wrist_roll_joint": 31.4,
                ".*_wrist_pitch_joint": 31.4,
                ".*_wrist_yaw_joint": 31.4,
            },
            stiffness=0,
            damping=0,
            armature=0.01,
            friction=0.1,
        ),
    },
)


H1_2_12DOF = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/cperrot/h1v2-Isaac/source/h1_assets/h1_assets/H12_handless_simplified_feet.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
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
            # "torso_joint": 0.0,
            # ".*_shoulder_pitch_joint": 0.0, # 0.28
            # ".*_shoulder_roll_joint": 0.0,
            # ".*_shoulder_yaw_joint": 0.0,
            # ".*_elbow_joint": 0.0, # 0.52
            # ".*_wrist_yaw_joint": 0.0,
            # ".*_wrist_roll_joint": 0.0,
            # ".*_wrist_pitch_joint": 0.0,
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
                # "torso_joint",
            ],
            effort_limit=220,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                # "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
                # "torso_joint": 5.0,
            },
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
        ),
        # "arms": ImplicitActuatorCfg(
        # "arms": DelayedPDActuatorCfg(
        #     joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_yaw_joint", ".*_wrist_roll_joint", ".*_wrist_pitch_joint"],
        #     effort_limit=75,
        #     velocity_limit=100.0,
        #     stiffness={
        #         ".*_shoulder_pitch_joint": 40.0,
        #         ".*_shoulder_roll_joint": 40.0,
        #         ".*_shoulder_yaw_joint": 40.0,
        #         ".*_elbow_joint": 40.0,
        #         ".*_wrist_yaw_joint": 40.0,
        #         ".*_wrist_roll_joint": 40.0,
        #         ".*_wrist_pitch_joint": 40.0,
        #     },
        #     damping={
        #         ".*_shoulder_pitch_joint": 10.0,
        #         ".*_shoulder_roll_joint": 10.0,
        #         ".*_shoulder_yaw_joint": 10.0,
        #         ".*_elbow_joint": 10.0,
        #         ".*_wrist_yaw_joint": 10.0,
        #         ".*_wrist_roll_joint": 10.0,
        #         ".*_wrist_pitch_joint": 10.0,
        #     },
        # ),
    },
)
H1_2_12DOF_MINIMAL = H1_2_12DOF.copy()


H1_2_27DOF = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/cperrot/h1v2-Isaac/source/h1_assets/h1_assets/h1_2.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
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
                # "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5,
                ".*_hip_roll_joint": 5,
                ".*_hip_pitch_joint": 5,
                "torso_joint": 5.0,
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
                ".*_knee_joint": 10.0,
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
                ".*_ankle_pitch_joint": 5.0,
                ".*_ankle_roll_joint": 5.0,
            },
            armature=0.1,
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
            armature=0.1,
        ),
    },
)

H1v2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/atrovatell/ws/IsaacLab/v1.0/h1_description/usd_unitreeros_h1v2/h1v2.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
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
                ".*_knee_joint": 10.0,
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
                ".*_ankle_pitch_joint": 5.0,
                ".*_ankle_roll_joint": 5.0,
            },
            armature=0.1,
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
            armature=0.1,
        ),
    },
)
