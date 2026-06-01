"""
Booster T1 29-DOF robot configuration for SONIC (GR00T-WholeBodyControl).

Hardware facts (from GMR/assets/booster_t1_29dof/t1_mocap.xml):
  nbody = 33  (world + 32 links)
  nu    = 27  (actuated DOFs — name "29dof" is a GMR convention, not MuJoCo nu)
  nq    = 34  (7 free-joint + 27 revolute)

MuJoCo actuator order (= DOF order in PKLs):
  [ 0] Left_Shoulder_Pitch   [ 1] Left_Shoulder_Roll    [ 2] Left_Elbow_Pitch
  [ 3] Left_Elbow_Yaw        [ 4] Left_Wrist_Pitch      [ 5] Left_Wrist_Yaw
  [ 6] Left_Hand_Roll        [ 7] Right_Shoulder_Pitch  [ 8] Right_Shoulder_Roll
  [ 9] Right_Elbow_Pitch     [10] Right_Elbow_Yaw       [11] Right_Wrist_Pitch
  [12] Right_Wrist_Yaw       [13] Right_Hand_Roll       [14] Waist
  [15] Left_Hip_Pitch        [16] Left_Hip_Roll         [17] Left_Hip_Yaw
  [18] Left_Knee_Pitch       [19] Left_Ankle_Pitch      [20] Left_Ankle_Roll
  [21] Right_Hip_Pitch       [22] Right_Hip_Roll        [23] Right_Hip_Yaw
  [24] Right_Knee_Pitch      [25] Right_Ankle_Pitch     [26] Right_Ankle_Roll

Isaac Lab uses URDF BFS order (NOT MJCF declaration order) → non-identity mapping.
Confirmed from joint list in IL error log (BFS produces interleaved L/R limbs).

KP/KD: no official Booster armature datasheet. Derived from MJCF torque limits
using the same natural-frequency formula as h2.py (10 Hz, damping_ratio=2):
  KP = armature * (2π * 10)²
  KD = 2 * damping_ratio * armature * (2π * 10)

NOTE: Isaac Lab requires the URDF extended to 29-DOF (Phase B1). Until the URDF
is created, the MJCF can be used for offline MuJoCo smoke tests only.
"""

# ruff: noqa: F401
try:
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets.articulation import ArticulationCfg
    import isaaclab.sim as sim_utils
    _ISAACLAB_AVAILABLE = True
except ImportError:
    _ISAACLAB_AVAILABLE = False

import math

ASSET_DIR = "gear_sonic/data/assets"

# ---------------------------------------------------------------------------
# KP/KD — estimated from torque limits via natural-frequency formula
# ---------------------------------------------------------------------------
_NATURAL_FREQ = 10 * 2.0 * math.pi  # 10 Hz
_DAMPING_RATIO = 2.0


def _kp(armature: float) -> float:
    return armature * _NATURAL_FREQ ** 2


def _kd(armature: float) -> float:
    return 2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ


# Armature estimates (no datasheet — iterate after F2 smoke test)
#   leg_strong  (hip pitch/roll, knee):  45–65 Nm  → armature ≈ 0.010
#   leg_yaw     (hip yaw):               30 Nm     → armature ≈ 0.006
#   ankle       (ankle pitch/roll):      15–24 Nm  → armature ≈ 0.004
#   waist       (waist yaw):             30 Nm     → armature ≈ 0.006
#   arms        (shoulder/elbow/wrist):  18 Nm     → armature ≈ 0.003
ARM_LEG_STRONG = 0.010
ARM_LEG_YAW    = 0.006
ARM_ANKLE      = 0.004
ARM_WAIST      = 0.006
ARM_ARMS       = 0.003

# ---------------------------------------------------------------------------
# Body / DOF order mappings
# Isaac Lab (MJCF) body order == MuJoCo declaration order → identity mappings
# ---------------------------------------------------------------------------

# 30 bodies in IsaacLab BFS traversal order (URDF has 30 links, no toe links).
# MJCF has 32 bodies (includes left_toe_link=24, right_toe_link=31).
# Derived from IL BFS joint order confirmed in F2 error log.
T1_ISAACLAB_JOINTS = [
    "Trunk",                                         # IL  0 → MJ  0
    "H1", "AL1", "AR1", "Waist",                     # IL 1-4
    "H2", "AL2", "AR2",                              # IL 5-7
    "Hip_Pitch_Left", "Hip_Pitch_Right",              # IL 8-9
    "AL3", "AR3", "Hip_Roll_Left", "Hip_Roll_Right",  # IL 10-13
    "AL4", "AR4", "Hip_Yaw_Left", "Hip_Yaw_Right",   # IL 14-17
    "AL5", "AR5", "Shank_Left", "Shank_Right",       # IL 18-21
    "AL6", "AR6",                                     # IL 22-23
    "Ankle_Cross_Left", "Ankle_Cross_Right",          # IL 24-25
    "left_hand_link", "right_hand_link",              # IL 26-27
    "left_foot_link", "right_foot_link",              # IL 28-29
]  # len == 30

# DOF mapping: IL BFS order → MJCF actuator order
# IL BFS (27 body joints, excl. head): Left_Shoulder_Pitch, Right_Shoulder_Pitch,
#   Waist, Left_Shoulder_Roll, Right_Shoulder_Roll, Left_Hip_Pitch, ...
# MJCF: [0-6] L-arm, [7-13] R-arm, [14] Waist, [15-20] L-leg, [21-26] R-leg
# Derived from error-log joint list (log showed exact BFS order from Isaac Lab).
T1_ISAACLAB_TO_MUJOCO_DOF = [
    0, 7, 14, 1, 8, 15, 21, 2, 9, 16, 22, 3, 10, 17, 23,
    4, 11, 18, 24, 5, 12, 19, 25, 6, 13, 20, 26,
]
T1_MUJOCO_TO_ISAACLAB_DOF = [
    0, 3, 7, 11, 15, 19, 23, 1, 4, 8, 12, 16, 20, 24, 2,
    5, 9, 13, 17, 21, 25, 6, 10, 14, 18, 22, 26,
]

# Body mapping: IL BFS (30 bodies) ↔ MJCF (32 bodies).
# MJCF bodies 24 (left_toe_link) and 31 (right_toe_link) have no IL equivalent.
T1_ISAACLAB_TO_MUJOCO_BODY = [
    0, 1, 3, 10, 17, 2, 4, 11, 18, 25,   # IL 0-9
    5, 12, 19, 26, 6, 13, 20, 27,         # IL 10-17
    7, 14, 21, 28, 8, 15, 22, 29,         # IL 18-25
    9, 16, 23, 30,                         # IL 26-29
]  # len == 30
T1_MUJOCO_TO_ISAACLAB_BODY = [
    0, 1, 5, 2, 6, 10, 14, 18, 22, 26,   # MJ 0-9
    3, 7, 11, 15, 19, 23, 27, 4,          # MJ 10-17
    8, 12, 16, 20, 24, 28, -1, 9,         # MJ 18-24 (-1 = left_toe_link, no IL)
    13, 17, 21, 25, 29, -1,               # MJ 25-31 (-1 = right_toe_link, no IL)
]  # len == 32

T1_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": T1_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": T1_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": T1_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": T1_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": T1_MUJOCO_TO_ISAACLAB_BODY,
}

# ---------------------------------------------------------------------------
# ArticulationCfg — requires Isaac Lab
# NOTE: Phase B1 (URDF extension 21→29 DOF) must be done before IL smoke test.
#       Until then, the MJCF at mjcf/booster_t1_29dof.xml can be used for
#       offline MuJoCo validation only.
# ---------------------------------------------------------------------------
if _ISAACLAB_AVAILABLE:
    T1_CFG = ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            replace_cylinders_with_capsules=True,
            # B1: create this URDF by extending htwk-gym T1_serial.urdf to 29-DOF
            asset_path=f"{ASSET_DIR}/robot_description/urdf/booster_t1_29dof/booster_t1_29dof.urdf",
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
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0, damping=0
                )
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.75),  # T1 initial height ~0.75 m (slightly above ground)
            joint_pos={
                # Slight knee bend and ankle compensation for natural standing
                "Left_Hip_Pitch":   -0.2,
                "Right_Hip_Pitch":  -0.2,
                "Left_Knee_Pitch":   0.4,
                "Right_Knee_Pitch":  0.4,
                "Left_Ankle_Pitch": -0.2,
                "Right_Ankle_Pitch":-0.2,
                # Arms relaxed
                "Left_Shoulder_Pitch":  0.2,
                "Right_Shoulder_Pitch": 0.2,
                "Left_Elbow_Pitch":     0.5,
                "Right_Elbow_Pitch":    0.5,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "leg_strong": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_Hip_Pitch",
                    ".*_Hip_Roll",
                    ".*_Knee_Pitch",
                ],
                effort_limit_sim={
                    ".*_Hip_Pitch": 45.0,
                    ".*_Hip_Roll":  45.0,
                    ".*_Knee_Pitch": 65.0,
                },
                velocity_limit_sim=20.0,
                stiffness=_kp(ARM_LEG_STRONG),
                damping=_kd(ARM_LEG_STRONG),
                armature=ARM_LEG_STRONG,
            ),
            "leg_yaw": ImplicitActuatorCfg(
                joint_names_expr=[".*_Hip_Yaw"],
                effort_limit_sim=30.0,
                velocity_limit_sim=20.0,
                stiffness=_kp(ARM_LEG_YAW),
                damping=_kd(ARM_LEG_YAW),
                armature=ARM_LEG_YAW,
            ),
            "ankle": ImplicitActuatorCfg(
                joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
                effort_limit_sim={
                    ".*_Ankle_Pitch": 24.0,
                    ".*_Ankle_Roll":  15.0,
                },
                velocity_limit_sim=20.0,
                stiffness=_kp(ARM_ANKLE),
                damping=_kd(ARM_ANKLE),
                armature=ARM_ANKLE,
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["Waist"],
                effort_limit_sim=30.0,
                velocity_limit_sim=20.0,
                stiffness=_kp(ARM_WAIST),
                damping=_kd(ARM_WAIST),
                armature=ARM_WAIST,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_Shoulder_Pitch",
                    ".*_Shoulder_Roll",
                    ".*_Elbow_Pitch",
                    ".*_Elbow_Yaw",
                    ".*_Wrist_Pitch",
                    ".*_Wrist_Yaw",
                    ".*_Hand_Roll",
                ],
                effort_limit_sim=18.0,
                velocity_limit_sim=22.0,
                stiffness=_kp(ARM_ARMS),
                damping=_kd(ARM_ARMS),
                armature=ARM_ARMS,
            ),
        },
    )

    # Action scale: 0.25 * effort_limit / stiffness (same formula as H2)
    T1_ACTION_SCALE = {}
    for group_name, act in T1_CFG.actuators.items():
        e = act.effort_limit_sim
        s = act.stiffness
        names = act.joint_names_expr
        if not isinstance(e, dict):
            e = {n: e for n in names}
        if not isinstance(s, dict):
            s = {n: s for n in names}
        for n in names:
            if n in e and n in s and s[n]:
                T1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
else:
    T1_CFG = None
    T1_ACTION_SCALE = {}
